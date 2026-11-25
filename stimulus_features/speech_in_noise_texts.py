"""Transcripts of the speech-in-noise passages and the alignment of their released content words.

The 15 passages of the speech-in-noise release are excerpts of weekly addresses of President Obama. The transcripts were
taken from the archived White House website (obamawhitehouse.archives.gov; US government works) and saved as plain text,
one file per address (`<name>.txt`, names as in SOURCES). Each passage's released content words (Stimuli/Run<k>.mat,
`wordVec`) are aligned in order with every transcript: the start is located where the passage's first content words match
best (greedy, within 12 tokens each), and the words are then aligned by longest common subsequence. A passage is usable
when at least 95 % of its content words are matched. The tokeniser writes digits as number words (years as two pairs),
joins line-break hyphenation, splits other hyphens and spells out notation that a speaker reads aloud.

Outputs (in ``out_dir``): word_map.json (per passage: the source transcript and the map from content word to transcript
token), clean_<name>.txt and spans_<name>.json (the normalised transcript and the character span of each token; inputs of
the surprisal step in encoding_models/speech_in_noise_word_level.py) and passage_match.json (match per passage).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

SOURCES = {
    "c1": "https://obamawhitehouse.archives.gov/the-press-office/weekly-address-president-obama-praises-benefits-and-successes-health-reform-already",
    "c2": "https://obamawhitehouse.archives.gov/realitycheck/the-press-office/weekly-address-president-obama-calls-hails-progress-health-insurance-reform-despite",
    "c3": "https://obamawhitehouse.archives.gov/photos-and-video/video/weekly-address-giving-government-back-american-people",
    "c4": "https://obamawhitehouse.archives.gov/realitycheck/the-press-office/weekly-address-president-obama-says-small-business-must-be-forefront-recovery",
    "c5": "https://obamawhitehouse.archives.gov/the-press-office/weekly-address-president-obama-says-it-time-move-forward-health-care-reform",
    "c6": "https://obamawhitehouse.archives.gov/photos-and-video/video/weekly-address-taking-insurance-companies-down-stretch/index__q_page=1.html",
    "c7": "https://obamawhitehouse.archives.gov/photos-and-video/video/weekly-address-premiums-profits-and-need-health-reform?page=51",
}
MIN_MATCH = 0.95
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
DEG, PRIME, RSQ, LSQ = "°", "′", "’", "‘"


def num_words(n: int, year: bool = False) -> str:
    if year and 1100 <= n <= 1999 and n % 100:
        return f"{num_words(n // 100)} {num_words(n % 100)}"
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if n < 1_000_000:
        return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return str(n)


def ordinal(n: int) -> str:
    w = num_words(n).split()
    last = w[-1]
    w[-1] = ORD.get(last, (last[:-1] + "ieth") if last.endswith("y") else last + "th")
    return " ".join(w)


def normalise(t: str) -> str:
    """Spoken forms of notation (coordinates, thousands separators, ordinals)."""
    t = re.sub(r"\b([NS])\.\s*lat\.?", lambda m: {"N": " north latitude ", "S": " south latitude "}[m.group(1)], t)
    t = re.sub(r"\b([EW])\.\s*long\.?", lambda m: {"E": " east longitude ", "W": " west longitude "}[m.group(1)], t)
    t = re.sub(r"\blat\.", " latitude ", t)
    t = re.sub(r"\blong\.", " longitude ", t)
    t = t.replace(DEG, " degrees ")
    t = re.sub(r"(\d)\s*['" + PRIME + "]", r"\1 minutes ", t)
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t)
    return t


def tokens(raw: str, version: int = 2):
    """Upper-case word tokens, their character spans in the cleaned text, and the cleaned text."""
    t = raw.replace("\r", "").replace(RSQ, "'").replace(LSQ, "'")
    t = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", t)
    if version >= 2:
        t = normalise(t)
    out, spans = [], []
    for m in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", t):
        w = m.group(0)
        if w.isdigit():
            for x in num_words(int(w), year=len(w) == 4).split():
                out.append(x.upper())
                spans.append(m.span())
        else:
            out.append(w.upper())
            spans.append(m.span())
    return out, spans, t


def lcs_align(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Longest common subsequence of a and b as matched index pairs."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), np.int32)
    for i in range(n - 1, -1, -1):
        ai, Di, Di1 = a[i], D[i], D[i + 1]
        for j in range(m - 1, -1, -1):
            Di[j] = Di1[j + 1] + 1 if ai == b[j] else max(Di1[j], Di[j + 1])
    i = j = 0
    pairs = []
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j)); i += 1; j += 1
        elif D[i + 1][j] >= D[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def head_score(run: list[str], toks: list[str], s: int, n_head: int = 15) -> int:
    k, p = 1, s + 1
    for w in run[1:n_head]:
        q = next((x for x in range(p, min(p + 12, len(toks))) if toks[x] == w), None)
        if q is not None:
            k, p = k + 1, q + 1
    return k


def locate(run: list[str], toks: list[str], lo: int, hi: int, version: int) -> int:
    """Start index where the run's first content words align best (greedy, within 12 tokens each)."""
    def best_in(a, b, n_head):
        best, best_i = -1, a
        for s in range(a, max(a + 1, b)):
            if toks[s] == run[0]:
                k = head_score(run, toks, s, n_head)
                if k > best:
                    best, best_i = k, s
        return best, best_i
    n_head = 15 if version >= 2 else 10
    best, i = best_in(lo, hi, n_head)
    if version >= 2 and best < min(n_head, len(run)) / 2:
        best2, i2 = best_in(0, len(toks), n_head)
        if best2 > best:
            i = i2
    return i


def match_passages(transcript_dir: Path, stim_dir: Path, out_dir: Path) -> list[dict]:
    """Align the content words of the 15 passages with the transcripts and write the alignment files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {}
    for name in SOURCES:
        raw = (transcript_dir / f"{name}.txt").read_text(encoding="utf-8")
        toks, spans, clean = tokens(raw, 2)
        texts[name] = (toks, spans, clean, hashlib.sha256(raw.encode("utf-8")).hexdigest())
    rows, align = [], {}
    for k in range(1, 16):
        m = sio.loadmat(stim_dir / f"Run{k}.mat", squeeze_me=True)
        run = [str(w).upper().replace("’", "'") for w in np.atleast_1d(m["wordVec"])]
        best = None
        for name, (toks, _, _, _) in texts.items():
            s = locate(run, toks, 0, len(toks), 2)
            pairs = lcs_align(run, toks[s: s + 6 * len(run) + 300])
            if best is None or len(pairs) > best[1]:
                best = (name, len(pairs), {i: s + j for i, j in pairs})
        name, n_match, mp = best
        rows.append({"passage": k, "source": name, "n": len(run), "matched": n_match, "match": n_match / len(run),
                     "usable": n_match / len(run) >= MIN_MATCH})
        align[k] = {"source": name, "map": mp}
    (out_dir / "passage_match.json").write_text(json.dumps({"passages": rows, "sources": {n: {"url": u, "sha256": texts[n][3]}
                                                                                          for n, u in SOURCES.items()}}, indent=2))
    (out_dir / "word_map.json").write_text(json.dumps(align))
    for n, (toks, spans, clean, _) in texts.items():
        (out_dir / f"clean_{n}.txt").write_text(clean, encoding="utf-8")
        (out_dir / f"spans_{n}.json").write_text(json.dumps(spans))
    return rows
