"""German text normalisation, word frequency and word class for the MEG attention dataset.

Normalised form (comparison of transcripts, aligner input), applied to each raw word separately: Unicode NFC, lower case
(ß kept); typographic apostrophes become straight, then apostrophes are removed; a token of digits becomes German words
(num2words, lang "de"; 1100-1999 read as years, otherwise cardinals); hyphens inside a word are removed; every other
character that is not a letter is removed (an empty result means the raw word was punctuation only).

Frequency: SUBTLEX-DE (Brysbaert et al., 2011; counts summed over capitalisations) on the Zipf scale of van Heuven et al.
(2014): Zipf = log10((count + 1) / (corpus millions + type millions)) + 3; out-of-vocabulary words get the value of a
count of 0. Lookup: the normalised form; for a raw word with an apostrophe, the part before it if the full form is
absent; for a hyphenated raw word, the minimum Zipf over its parts. Word class: function word if the normalised form is
in the Snowball German stop-word list (231 entries; its entry "daß" also matches "dass"). The lexicon files are checked
against their SHA-256 values.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import unicodedata
import zipfile
from functools import lru_cache
from pathlib import Path

LEX = Path(__file__).parents[1] / "data" / "lexicon"
PINS = {"SUBTLEX-DE_txt_cleaned_with_Google00.zip": "0d00058488620f0bbe6b77dc7ef348e0d5ce8420c530eafb6b392d3048a66c1c",
        "snowball_german_stop.txt": "22a7785b53860256280fba74b93d8f09a0f3a8f52edaa3cdc3750e8c13a3874c"}
REFORM = {"daß": "dass"}


def check_pins():
    for f, h in PINS.items():
        got = hashlib.sha256((LEX / f).read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"{f}: checksum {got} differs from the pinned {h}")
APOS = {"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"}


def _num_de(m: re.Match) -> str:
    from num2words import num2words
    n = int(m.group(0))
    if 1100 <= n <= 1999:
        return num2words(n, lang="de", to="year").replace(" ", "")
    return num2words(n, lang="de").replace(" ", "")


def normalise(raw: str) -> str:
    w = unicodedata.normalize("NFC", str(raw)).lower()
    for a, b in APOS.items():
        w = w.replace(a, b)
    w = w.replace("'", "")
    w = re.sub(r"\d+", _num_de, w)
    w = w.replace("-", "").replace("‐", "").replace("‑", "")
    w = "".join(ch for ch in w if ch.isalpha())
    return unicodedata.normalize("NFC", w)


def normalise_words(raws: list[str]) -> tuple[list[str], list[int]]:
    """Normalised tokens and, for each, the index of its raw word (raw words that are punctuation only drop out)."""
    toks, idx = [], []
    for i, r in enumerate(raws):
        n = normalise(r)
        if n:
            toks.append(n)
            idx.append(i)
    return toks, idx


@lru_cache(maxsize=1)
def _tables():
    import pandas as pd
    check_pins()
    z = zipfile.ZipFile(LEX / "SUBTLEX-DE_txt_cleaned_with_Google00.zip")
    d = pd.read_csv(io.BytesIO(z.read("SUBTLEX-DE_cleaned_with_Google00.txt")), sep="\t", encoding="latin-1", decimal=",",
                    keep_default_na=False, na_values=[], quoting=3)
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    d["w"] = [unicodedata.normalize("NFC", str(x)).lower() for x in d.Word]
    counts = d.groupby("w")["CUMfreqcount"].max().to_dict()
    hf = d[d.CUMfreqcount >= 1000]
    corpus_m = float((hf.CUMfreqcount / hf.SUBTLEX).median())
    types_m = len(d) / 1e6
    stop = set()
    for ln in (LEX / "snowball_german_stop.txt").read_text(encoding="utf-8").splitlines():
        w = unicodedata.normalize("NFC", ln.split("|")[0].strip())
        if w:
            stop.add(w)
            if w in REFORM:
                stop.add(REFORM[w])
    return counts, corpus_m, types_m, stop


def lexicon_constants() -> dict:
    counts, corpus_m, types_m, stop = _tables()
    return {"corpus_millions": corpus_m, "types_millions": types_m, "n_types": len(counts), "n_stop": len(stop),
            "zipf_oov": math.log10(1 / (corpus_m + types_m)) + 3}


def _zipf_count(c: float) -> float:
    _, corpus_m, types_m, _ = _tables()
    return math.log10((c + 1) / (corpus_m + types_m)) + 3


def zipf(raw: str) -> tuple[float, bool]:
    """Zipf frequency of a raw word and whether it was found."""
    counts, *_ = _tables()
    r = unicodedata.normalize("NFC", str(raw))
    for a, b in APOS.items():
        r = r.replace(a, b)
    parts = [p for p in re.split(r"[-‐‑]", r) if normalise(p)]
    if len(parts) > 1:
        vals = [zipf(p) for p in parts]
        return min(v for v, _ in vals), all(f for _, f in vals)
    n = normalise(r)
    if n in counts:
        return _zipf_count(counts[n]), True
    if "'" in r:
        host = normalise(r.split("'")[0])
        if host in counts:
            return _zipf_count(counts[host]), True
    return _zipf_count(0), False


def is_function(raw: str) -> bool:
    *_, stop = _tables()
    return normalise(raw) in stop
