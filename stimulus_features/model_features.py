"""Language- and audio-model features of the CantiSong songs, and GPT-2 medium surprisal for the robustness analysis.

* GPT-2 (124 M) hidden states per lyric word at layers 0, 4, 8 and 12 (mean over the word's tokens; context: the
  preceding lyrics of the song); principal component analysis to 16 dimensions per layer over all lyric words.
* HuBERT-base (facebook/hubert-base-ls960), layers 3, 6, 9 and 12: 50 Hz frames, 20 s chunks.
* AST (MIT/ast-finetuned-audioset-10-10-0.4593), layers 3, 6, 9 and 12: 10.24 s windows, patch tokens averaged over
  frequency (about 10 Hz frames).
  Audio features for every version (spoken, sung, hummed); principal component analysis to 16 dimensions per model and
  layer over all versions (fitted on 30,000 random frames).
* GPT-2 medium surprisal of the CantiSong lyrics (word-table order) and of the Alice chapter (sliding windows).

Requires torch, transformers, librosa and scikit-learn; a GPU is used when available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stimulus_features.lyric_alignment import surprisal, words_of
from stimulus_features.surprisal import surprisal_long

GPT_LAYERS, AUDIO_LAYERS, NPC = (0, 4, 8, 12), (3, 6, 9, 12), 16
VERSIONS = ("speech", "song", "melody")
SEED = None                               # seed of the PCA and of the frame subsample (an integer; set before use)


def _device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def lyric_text(stim: Path, s: str) -> tuple[str, list]:
    text = (stim / "lyrics" / f"{s}.txt").read_text(encoding="utf-8").replace("’", "'")
    return text, words_of(text)


def gpt2_word_states(stim: Path, songs: list[str], out: Path):
    """gpt2_L<layer>_<song>.npy: words x 16 principal components."""
    import torch
    from sklearn.decomposition import PCA
    from transformers import GPT2Model, GPT2TokenizerFast
    dev = _device()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").to(dev).eval()
    per_song = {}
    for s in songs:
        text, ws = lyric_text(stim, s)
        enc = tok(text, return_offsets_mapping=True)
        with torch.no_grad():
            hs = model(torch.tensor([enc["input_ids"]], device=dev), output_hidden_states=True).hidden_states
        offs = enc["offset_mapping"]
        vec = {}
        for L in GPT_LAYERS:
            H = hs[L][0].cpu().numpy()
            vec[L] = np.stack([H[[k for k, (o0, o1) in enumerate(offs) if o1 > a and o0 < b]].mean(0) for _, a, b in ws])
        per_song[s] = vec
    for L in GPT_LAYERS:
        allv = np.vstack([per_song[s][L] for s in songs])
        pca = PCA(NPC, random_state=SEED).fit(allv)
        for s in songs:
            np.save(out / f"gpt2_L{L}_{s}.npy", pca.transform(per_song[s][L]).astype(np.float32))


def hubert_frames(y: np.ndarray, model, fe) -> dict[int, np.ndarray]:
    import torch
    dev = next(model.parameters()).device
    chunks = {L: [] for L in AUDIO_LAYERS}
    step = 20 * 16000
    for i in range(0, len(y), step):
        seg = y[i:i + step]
        if len(seg) < 800:
            continue
        inp = fe(seg, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        with torch.no_grad():
            hs = model(inp, output_hidden_states=True).hidden_states
        for L in AUDIO_LAYERS:
            chunks[L].append(hs[L][0].cpu().numpy())
    return {L: np.vstack(v) for L, v in chunks.items()}                    # 50 Hz


def ast_frames(y: np.ndarray, model, fe) -> dict[int, np.ndarray]:
    import torch
    dev = next(model.parameters()).device
    chunks = {L: [] for L in AUDIO_LAYERS}
    win = int(10.24 * 16000)
    for i in range(0, len(y), win):
        seg = y[i:i + win]
        if len(seg) < 16000:
            continue
        inp = fe(seg, sampling_rate=16000, return_tensors="pt")["input_values"].to(dev)
        with torch.no_grad():
            hs = model(inp, output_hidden_states=True).hidden_states
        f_dim, t_dim = model.embeddings.get_shape(model.config)
        n_keep = int(np.ceil(t_dim * len(seg) / win))
        for L in AUDIO_LAYERS:
            H = hs[L][0, 2:].cpu().numpy().reshape(f_dim, t_dim, -1).mean(0)
            chunks[L].append(H[:n_keep])
    return {L: np.vstack(v) for L, v in chunks.items()}                    # t_dim frames per 10.24 s


def audio_states(stim: Path, songs: list[str], out: Path):
    """<model>_L<layer>_<song>_<version>.npz: frames x 16 principal components (x) and the frame rate (rate)."""
    import librosa
    import torch
    from sklearn.decomposition import PCA
    from transformers import ASTFeatureExtractor, ASTModel, HubertModel, Wav2Vec2FeatureExtractor
    dev = _device()
    specs = {"hubert": (HubertModel.from_pretrained("facebook/hubert-base-ls960").to(dev).eval(),
                        Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960"), hubert_frames),
             "ast": (ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval(),
                     ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"), ast_frames)}
    for name, (model, fe, fn) in specs.items():
        feats, rates = {}, {}
        for s in songs:
            for v in VERSIONS:
                y, _ = librosa.load(stim / "sounds" / f"{s}_{v}.wav", sr=16000, mono=True)
                feats[(s, v)] = fn(y, model, fe)
                rates[(s, v)] = feats[(s, v)][AUDIO_LAYERS[0]].shape[0] / (len(y) / 16000)
        for L in AUDIO_LAYERS:
            allf = np.vstack([f[L] for f in feats.values()])
            sub = allf[np.random.default_rng(SEED).choice(len(allf), min(30000, len(allf)), replace=False)]
            pca = PCA(NPC, random_state=SEED).fit(sub)
            for (s, v), f in feats.items():
                np.savez(out / f"{name}_L{L}_{s}_{v}.npz", x=pca.transform(f[L]).astype(np.float32), rate=rates[(s, v)])
        del model
        torch.cuda.empty_cache()


def medium_surprisal(stim: Path, songs: list[str], words_dir: Path, out: Path, alice_csv: Path):
    """<song>_words_medium.csv (the word table with surprisal_medium_bits) and words_gpt2_medium.csv for Alice."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").eval()
    for s in songs:
        text, ws = lyric_text(stim, s)
        w = pd.read_csv(words_dir / f"{s}_words.csv")
        w["surprisal_medium_bits"] = surprisal(text, [(a, b) for _, a, b in ws], tok, model)
        w.to_csv(out / f"{s}_words_medium.csv", index=False)
    d = pd.read_csv(alice_csv)
    words = [str(x) for x in d.word_clean]
    text, spans = "", []
    for i, w_ in enumerate(words):
        text += "" if (i == 0 or w_.startswith("'")) else " "
        spans.append((len(text), len(text) + len(w_)))
        text += w_
        if i == len(words) - 1 or d.Sentence.iloc[i + 1] != d.Sentence.iloc[i]:
            text += "."
    d["gpt2_medium_surprisal_bits"] = surprisal_long(text, spans, tok, model)
    d.to_csv(alice_csv.with_name("words_gpt2_medium.csv"), index=False)
