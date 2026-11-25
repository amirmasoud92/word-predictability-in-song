"""Word features of the MEG attention stimuli: language-model surprisal, lexical controls and a timing audit.

* Surprisal (bits) of each word = sum of -log2 P(token | preceding text) over its tokens. Context = the preceding text of
  the same presented chapter (attended stream) or excerpt (ignored stream) only; the text starts with the end-of-text
  token. Words are joined by single spaces with their punctuation. Main model benjamin/gerpt2 (GPT-2 small architecture);
  robustness model benjamin/gerpt2-large; both pinned by revision. Texts longer than 1024 tokens use sliding windows of
  1024 tokens with a stride of 512.
* Controls: Zipf frequency (SUBTLEX-DE), function or content word (Snowball German stop-word list), word duration (from
  the aligner) and peak broadband envelope within the word (3.5 kHz low-pass, 8 kHz, |x|^0.6, 1 kHz, then 64 Hz).
* Timing audit per chapter: the word-onset train (impulses at the aligned onsets, 100 Hz, 50 ms Hann) cross-correlated
  with the envelope onsets (half-wave rectified first difference of the envelope at 100 Hz) at lags of -5 to +5 s;
  positive lag = envelope later than the word onsets; parabolic interpolation around the peak.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
import torch
from scipy.io import wavfile

from common.settings import load_analysis
from stimulus_features import german_text as T, speech_envelope as tk, transcription as A

DER, OUT = A.DER, A.OUT
LMS = {"gerpt2": ("benjamin/gerpt2", "0093b45ef0d957cc545d872c208e66fc13861a0e"),
       "gerpt2_large": ("benjamin/gerpt2-large", "2fa57adea207b02fb56972f07544fa0c65e41ee7")}
FS_ENV = 64
MAX_LAG_S = 5.0


def surprisal_long(text: str, spans: list[tuple[int, int]], tok, model, win: int = 1024, stride: int = 512):
    enc = tok(text, return_offsets_mapping=True)
    ids = [tok.eos_token_id] + enc["input_ids"]
    offs = [(0, 0)] + enc["offset_mapping"]
    tok_s = np.full(len(ids), np.nan)
    start = 0
    while True:
        end = min(start + win, len(ids))
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor([ids[start:end]], device="cuda")).logits[0].float(), dim=-1)
        first = 1 if start == 0 else start + (win - stride)
        for k in range(max(first, start + 1), end):
            tok_s[k] = float(-logp[k - start - 1, ids[k]] / np.log(2))
        if end == len(ids):
            break
        start += stride
    s_out, n_out = [], []
    for a, b in spans:
        sel = [v for (o0, o1), v in zip(offs, tok_s) if o1 > a and o0 < b]
        s_out.append(float(np.nansum(sel)) if sel else np.nan)
        n_out.append(len(sel))
    return s_out, n_out, len(ids)


def envelope_1k(path: Path) -> np.ndarray:
    sr, y = wavfile.read(path)
    y = np.asarray(y, float)
    if y.dtype.kind != "f" or np.abs(y).max() > 1.5:
        y = y / 32768.0
    y = y.mean(1) if y.ndim > 1 else y
    return tk.broadband_envelope(y, sr, load_analysis().section("envelope"))


def audit(on_s: np.ndarray, env1k: np.ndarray) -> dict:
    e100 = ss.resample_poly(env1k, 1, 10)
    eon = np.maximum(np.diff(e100, prepend=e100[:1]), 0)
    train = np.zeros(len(e100))
    idx = np.round(on_s[np.isfinite(on_s)] * 100).astype(int)
    np.add.at(train, idx[(idx >= 0) & (idx < len(train))], 1.0)
    train = np.convolve(train, np.hanning(7)[1:-1], mode="same")
    a, r = train - train.mean(), eon - eon.mean()
    n, L = len(a), int(MAX_LAG_S * 100)
    cc = np.array([np.corrcoef(a[: n - k], r[k:])[0, 1] if k >= 0 else np.corrcoef(a[-k:], r[: n + k])[0, 1]
                   for k in range(-L, L + 1)])
    k = int(np.argmax(cc))
    frac = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return {"lag_s": (k - L + frac) / 100, "r_best": float(cc[k]), "r_zero": float(cc[L]),
            "second_peak_r": float(np.sort(cc[np.abs(np.arange(len(cc)) - k) > 20])[-1])}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def word_features():
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    wt = pd.read_csv(DER / "word_times.csv")
    chs = {c["stem"]: c for c in A.chapters()}
    rows, aud, pins = [], [], {}
    texts = {}
    for stem, g in wt.groupby("stem", sort=False):
        tr = json.loads((DER / "transcripts" / f"{stem}.json").read_text(encoding="utf-8"))
        raw = tr["words"]
        keep = sorted(set(g.raw_index))
        text, spans = "", {}
        for i in keep:
            if text:
                text += " "
            spans[i] = (len(text), len(text) + len(raw[i]))
            text += raw[i]
        texts[stem] = (text, spans)
    surp = {}
    for name, (rid, rev) in LMS.items():
        path = Path(snapshot_download(rid, revision=rev, allow_patterns=["*.json", "model.safetensors", "*.txt"]))
        pins[name] = {"repo": rid, "revision": rev, "model.safetensors_sha256": sha256(path / "model.safetensors")}
        tok = AutoTokenizer.from_pretrained(rid, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(rid, revision=rev, dtype=torch.float32).to("cuda").eval()
        for stem, (text, spans) in texts.items():
            idx = sorted(spans)
            s, n, ntok = surprisal_long(text, [spans[i] for i in idx], tok, model)
            surp[(name, stem)] = (dict(zip(idx, s)), dict(zip(idx, n)), ntok)
        del model
        torch.cuda.empty_cache()
    for stem, g in wt.groupby("stem", sort=False):
        ch = chs[stem]
        tr = json.loads((DER / "transcripts" / f"{stem}.json").read_text(encoding="utf-8"))
        raw = tr["words"]
        env1k = envelope_1k(ch["path"])
        env64 = ss.resample_poly(env1k, FS_ENV, 1000)
        for r in g.itertuples():
            w = raw[r.raw_index]
            z, found = T.zipf(w)
            if np.isfinite(r.onset):
                a = int(round(r.onset * FS_ENV))
                b = max(int(round(r.offset * FS_ENV)), a + 1)
                peak = float(env64[a:b].max()) if a < env64.size else np.nan
            else:
                peak = np.nan
            rows.append({"stem": stem, "stream": ch["stream"], "speaker": ch["speaker"], "book": ch["book"], "chapter": ch["chapter"],
                         "word_index": r.word_index, "raw_index": r.raw_index, "word": w, "norm": T.normalise(w),
                         "onset": r.onset, "offset": r.offset, "duration": r.offset - r.onset, "aligned": r.aligned,
                         "surprisal_gerpt2": surp[("gerpt2", stem)][0][r.raw_index],
                         "n_tokens_gerpt2": surp[("gerpt2", stem)][1][r.raw_index],
                         "surprisal_gerpt2_large": surp[("gerpt2_large", stem)][0][r.raw_index],
                         "zipf": z, "zipf_found": found, "function": T.is_function(w), "peak_env": peak,
                         "source": tr["source"][r.raw_index]})
        aud.append({"stem": stem, "stream": ch["stream"], "speaker": ch["speaker"], **audit(g.onset.to_numpy(), env1k)})
    d = pd.DataFrame(rows)
    d.to_csv(DER / "words.csv", index=False)
    d.drop(columns=["word", "norm", "raw_index"]).to_csv(OUT / "word_features.csv", index=False)
    pd.DataFrame(aud).to_csv(OUT / "timing_audit.csv", index=False)
    (OUT / "language_model_pins.json").write_text(json.dumps({"models": pins, "lexicon": T.lexicon_constants(), "lexicon_pins": T.PINS,
                                                     "n_tokens_per_text": {f"{k[0]}|{k[1]}": v[2] for k, v in surp.items()}},
                                                    indent=2), encoding="utf-8")
