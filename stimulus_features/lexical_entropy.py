"""GPT-2 lexical entropy before each word.

Entropy (bits) of GPT-2's (124 M) predictive distribution at the position that predicts a word's first token, that is,
the uncertainty before the word arrives. Contexts, text and tokenisation are those of the surprisal values:
* CantiSong: the song's lyrics, preceding context only (stimulus_features/lyric_alignment.py);
* Alice: the chapter with sentence punctuation, sliding 1024-token windows with a 512-token stride, each token taken
  from the window that gives it the most context.
Surprisal is recomputed in the same pass and must match the stored values (alignment check).

Requires torch and transformers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stimulus_features.lyric_alignment import words_of

LN2 = np.log(2)


def token_scores(ids: list[int], model, win: int = 1024, stride: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Per token k (k >= 1): surprisal of token k and entropy of the distribution that predicts it (bits)."""
    import torch
    surp, ent = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    start = 0
    while True:
        end = min(start + win, len(ids))
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor([ids[start:end]])).logits[0], dim=-1)
        h = -(logp.exp() * logp).sum(-1) / LN2
        first = 1 if start == 0 else start + (win - stride)
        for k in range(max(first, start + 1), end):
            surp[k] = float(-logp[k - start - 1, ids[k]] / LN2)
            ent[k] = float(h[k - start - 1])
        if end == len(ids):
            break
        start += stride
    return surp, ent


def word_values(text: str, spans: list[tuple[int, int]], tok, model) -> tuple[list[float], list[float]]:
    """Surprisal (summed over a word's tokens) and entropy before the word's first token, per character span."""
    enc = tok(text, return_offsets_mapping=True)
    ids = [tok.eos_token_id] + enc["input_ids"]
    offs = [(0, 0)] + enc["offset_mapping"]
    surp, ent = token_scores(ids, model)
    s_out, e_out = [], []
    for a, b in spans:
        ks = [k for k, (o0, o1) in enumerate(offs) if o1 > a and o0 < b]
        s_out.append(float(np.nansum(surp[ks])) if ks else np.nan)
        e_out.append(float(ent[ks[0]]) if ks else np.nan)
    return s_out, e_out


def cantisong_entropy(stim: Path, words_dir: Path) -> dict:
    """<song>_entropy.csv per song (idx, word, surprisal_bits, surprisal_check, entropy_bits); returns the largest
    surprisal mismatch per song."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    check = {}
    for lf in sorted((stim / "lyrics").glob("*.txt")):
        s = lf.stem
        text = lf.read_text(encoding="utf-8").replace("’", "'")
        ws = words_of(text)
        sv, ev = word_values(text, [(a, b) for _, a, b in ws], tok, model)
        w = pd.read_csv(words_dir / f"{s}_words.csv")
        assert len(w) == len(ws), s
        w["entropy_bits"] = ev
        w["surprisal_check"] = sv
        w[["idx", "word", "surprisal_bits", "surprisal_check", "entropy_bits"]].to_csv(words_dir / f"{s}_entropy.csv", index=False)
        check[s] = float(np.nanmax(np.abs(w.surprisal_bits - w.surprisal_check)))
    return check


def alice_entropy(words_csv: Path, out_csv: Path) -> float:
    """Entropy for the Alice word table (columns word_clean, Sentence, gpt2_surprisal_bits); returns the largest
    surprisal mismatch."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    d = pd.read_csv(words_csv)
    words = [str(x) for x in d.word_clean]
    text, spans = "", []
    for i, w_ in enumerate(words):
        text += "" if (i == 0 or w_.startswith("'")) else " "
        spans.append((len(text), len(text) + len(w_)))
        text += w_
        if i == len(words) - 1 or d.Sentence.iloc[i + 1] != d.Sentence.iloc[i]:
            text += "."
    sv, ev = word_values(text, spans, tok, model)
    d["gpt2_entropy_bits"] = ev
    d["surprisal_check"] = sv
    d.to_csv(out_csv, index=False)
    return float(np.nanmax(np.abs(d.gpt2_surprisal_bits - d.surprisal_check)))
