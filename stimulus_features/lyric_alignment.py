"""Word timing and GPT-2 surprisal of the CantiSong lyrics.

Words are aligned to the released phone annotations of the spoken and sung versions by Needleman-Wunsch alignment of
CMU-dictionary pronunciations (stress removed) to the annotated ARPAbet sequence (sil and sp removed), with three rounds
in which each word takes the pronunciation variant closest to its aligned segment. A word's onset is the start of the
first annotated phone aligned to it. Because the annotation is phonetic (sung realisations: vocalised /l/, reduced
vowels), the audit criterion is the anchoring of the word-initial phone: the word's first dictionary phone is aligned to
an identical annotated phone, or vowel to vowel. Words that are not anchored keep the onset of their first aligned phone.

Surprisal: GPT-2 (124 M), -log2 P(word | all preceding lyrics of the song), summed over the word's tokens, with
<|endoftext|> as the first context token.

Requires pronouncing (CMU Pronouncing Dictionary), torch and transformers.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

VOWELS = set("aa ae ah ao aw ay eh er ey ih iy ow oy uh uw".split())
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’]*(?:-[A-Za-z0-9'’]+)*")


def words_of(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD.finditer(text)]


def prons(word: str) -> list[list[str]] | None:
    """All CMU pronunciation variants (stress removed); hyphenated words combine the parts' first variants."""
    import pronouncing
    w = word.replace("’", "'").lower()
    if "-" not in w:
        cands = pronouncing.phones_for_word(w) or pronouncing.phones_for_word(w.strip("'"))
        if cands:
            return [re.sub(r"\d", "", c).lower().split() for c in cands]
    one = pron(word)
    return [one] if one else None


def lev(a: list[str], b: list[str]) -> int:
    D = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, D[0] = D.copy(), i
        for j in range(1, len(b) + 1):
            D[j] = min(prev[j] + 1, D[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return int(D[-1])


def pron(word: str) -> list[str] | None:
    """First pronunciation of each hyphen-separated part; g-dropping (darlin') and spelled capitals (ABC's) resolved."""
    import pronouncing
    parts = []
    for part in word.replace("’", "'").lower().split("-"):
        cands = pronouncing.phones_for_word(part) or pronouncing.phones_for_word(part.strip("'"))
        if not cands and part.endswith("in'"):
            cands = pronouncing.phones_for_word(part[:-1] + "g")
        if not cands and part.isalpha() and part.upper() == word.split("-")[0].replace("’", "'").split("'")[0]:
            cands = None
        if not cands:
            letters = [pronouncing.phones_for_word(ch) for ch in part if ch.isalpha()]
            if part.rstrip("'s").isalpha() and all(letters) and word[:1].isupper() and word.upper() == word.replace("’", "'").upper():
                cands = [" ".join(l[0] for l in letters)]
        if not cands:
            return None
        parts.append(re.sub(r"\d", "", cands[0]).lower())
    return " ".join(parts).split()


def align(ref: list[str], hyp: list[str]) -> list[int]:
    """Needleman-Wunsch (match 0, mismatch 1, gap 1); for each ref position the aligned hyp index or -1."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        sub = D[i - 1, :-1] + (np.array(hyp) != ref[i - 1])
        dele = D[i - 1, 1:] + 1
        row = np.minimum(sub, dele)
        cur = np.empty(m + 1)
        cur[0] = D[i, 0]
        for j in range(1, m + 1):
            cur[j] = min(row[j - 1], cur[j - 1] + 1)
        D[i] = cur
    i, j, out = n, m, [-1] * n
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def load_phones(path: Path) -> list[tuple[float, float, str]]:
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    return [(float(a), float(b), p) for a, b, p in rows if p not in ("sil", "sp")]


def aligned_pronunciations(ws, variants, hyp: list[str]) -> tuple[list[str], list[int], list[int]]:
    """Variant refinement (at most three rounds) and the final alignment: dictionary phones (ref), the word owning each
    of them, and the annotated phone aligned to each (-1 if none). Words without a dictionary entry get placeholder
    phones that align to nothing."""
    chosen = [v[0] if v else ["?"] * max(2, len(ws[k][0]) // 2) for k, v in enumerate(variants)]
    for _ in range(3):
        ref, owner = [], []
        for k, p in enumerate(chosen):
            ref += p
            owner += [k] * len(p)
        amap = align(ref, hyp)
        new = []
        for k, v in enumerate(variants):
            hit = [amap[i] for i, o in enumerate(owner) if o == k and amap[i] >= 0]
            seg = hyp[hit[0]: hit[-1] + 1] if hit else []
            new.append(min(v, key=lambda c: lev(c, seg)) if v and seg else chosen[k])
        if new == chosen:
            break
        chosen = new
    return ref, owner, amap


def surprisal(text: str, spans: list[tuple[int, int]], tok, model) -> list[float]:
    """GPT-2 surprisal (bits) of each character span, summed over its tokens, in one pass over the text."""
    import torch
    enc = tok(text, return_offsets_mapping=True)
    ids = [tok.eos_token_id] + enc["input_ids"]
    offs = [(0, 0)] + enc["offset_mapping"]
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([ids])).logits[0], dim=-1)
    tok_s = [np.nan] + [float(-logp[k - 1, ids[k]] / np.log(2)) for k in range(1, len(ids))]
    out = []
    for a, b in spans:
        sel = [s for (o0, o1), s in zip(offs, tok_s) if o1 > a and o0 < b]
        out.append(float(np.sum(sel)) if sel else np.nan)
    return out


def word_tables(stim: Path, out: Path):
    """For every song (Stimuli/lyrics/<song>.txt): <song>_words.csv with word, surprisal, dictionary coverage and, per
    version (sung, spoken), onset, offset, number of mismatched phones and the anchoring flag; qc.json with the share of
    anchored words."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    out.mkdir(parents=True, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    qc = {}
    for lf in sorted((stim / "lyrics").glob("*.txt")):
        s = lf.stem
        text = lf.read_text(encoding="utf-8").replace("’", "'")
        ws = words_of(text)
        variants = [prons(w) for w, _, _ in ws]
        surp = surprisal(text, [(a, b) for _, a, b in ws], tok, model)
        rows = [{"idx": k, "word": w, "surprisal_bits": surp[k], "in_dict": variants[k] is not None} for k, (w, _, _) in enumerate(ws)]
        for version in ("song", "speech"):
            ph = load_phones(stim / "ph_annotations" / f"{s}_{version}.txt")
            ref, owner, amap = aligned_pronunciations(ws, variants, [p for _, _, p in ph])
            for k in range(len(ws)):
                pos = [i for i, o in enumerate(owner) if o == k]
                hit = [amap[i] for i in pos if amap[i] >= 0]
                mism = sum(1 for i in pos if amap[i] < 0 or ph[amap[i]][2] != ref[i])
                j0 = amap[pos[0]] if pos else -1
                rows[k][f"anchored_{version}"] = bool(j0 >= 0 and (ph[j0][2] == ref[pos[0]] or (ref[pos[0]] in VOWELS and ph[j0][2] in VOWELS)))
                rows[k][f"onset_{version}"] = ph[hit[0]][0] if hit else np.nan
                rows[k][f"offset_{version}"] = ph[hit[-1]][1] if hit else np.nan
                rows[k][f"mismatch_{version}"] = int(mism)
        with open(out / f"{s}_words.csv", "w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows[0]))
            wri.writeheader()
            wri.writerows(rows)
        qc[s] = {"n_words": len(ws), "in_dict": int(sum(r["in_dict"] for r in rows)),
                 **{f"le1_mismatch_{v}": float(np.mean([r[f"mismatch_{v}"] <= 1 for r in rows])) for v in ("song", "speech")},
                 **{f"onset_anchored_{v}": float(np.mean([r[f"anchored_{v}"] for r in rows])) for v in ("song", "speech")},
                 **{f"monotonic_{v}": bool(np.all(np.diff([r[f"onset_{v}"] for r in rows if not np.isnan(r[f"onset_{v}"])]) >= 0))
                    for v in ("song", "speech")}}
    allw = sum(v["n_words"] for v in qc.values())
    qc["overall"] = {"n_words": allw, **{f"{m}_{v}": float(sum(q[f"{m}_{v}"] * q["n_words"] for k, q in qc.items() if k != "overall") / allw)
                                         for v in ("song", "speech") for m in ("le1_mismatch", "onset_anchored")}}
    (out / "qc.json").write_text(json.dumps(qc, indent=2))
