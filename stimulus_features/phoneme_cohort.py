"""Phoneme surprisal and cohort entropy at the released phone onsets of the CantiSong lyrics (Fig. S5).

Phone tables: the word alignment of stimulus_features/lyric_alignment.py is re-run to keep, for every annotated phone
(sil and sp removed), the word it is aligned to and the position of the dictionary phone it is aligned to (1 =
word-initial). The reproduced word onsets must equal those of the word tables.

Lexicon: CMU Pronouncing Dictionary entries (stress removed) of the words that are in SUBTLEX-US under the frequency
lookup of stimulus_features/english_lexicon.py, each word weighted by 10^Zipf and the weight split equally over its
distinct pronunciations; every lyric word is added with its chosen pronunciation and its Zipf weight (out-of-vocabulary
words: the smoothed Zipf of a zero count), so that its own cohort is never empty.

Cohort measures for phone k of a word with pronunciation p1..pn: cohort C_k = lexicon entries whose pronunciation begins
with p1..pk (C_0 = the whole lexicon); phoneme surprisal = -log2 W(C_k) / W(C_{k-1}); cohort entropy = entropy (bits) of
the word distribution in C_k (weights summed over a word's pronunciations). A value is placed at the onset of the
annotated phone aligned to dictionary phone k; dictionary phones without an aligned annotated phone carry no value,
annotated phones aligned to no dictionary phone carry none (they keep their phone onset), and words without a dictionary
pronunciation carry no phone values.

Phone tables require pronouncing; the cohort measures and regressors run without it.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from stimulus_features import english_lexicon as LX, song_regressors as g
from stimulus_features.lyric_alignment import aligned_pronunciations, load_phones, prons, words_of


# ---------------------------------------------------------------- phone tables
def phone_table(stim: Path, s: str, version: str, ws, variants) -> tuple[list[dict], list[list[str]], list[float]]:
    """Per annotated phone: onset, offset, label, aligned word (-1 if none), position in the word's pronunciation (0 if
    none), dictionary phone; the pronunciation of each word used by the alignment; the reproduced word onsets."""
    ph = load_phones(stim / "ph_annotations" / f"{s}_{version}.txt")
    ref, owner, amap = aligned_pronunciations(ws, variants, [p for _, _, p in ph])
    word_pron = [[] for _ in ws]
    for i, o in enumerate(owner):
        word_pron[o].append(ref[i])
    pos_in_word, seen = [], {}
    for o in owner:
        seen[o] = seen.get(o, 0) + 1
        pos_in_word.append(seen[o])
    rows = [{"phone": j, "onset": a, "offset": b, "label": p, "word": -1, "pos": 0, "dict_phone": ""} for j, (a, b, p) in enumerate(ph)]
    for i, j in enumerate(amap):
        if j >= 0:
            rows[j].update(word=owner[i], pos=pos_in_word[i], dict_phone=ref[i])
    onsets = []
    for k in range(len(ws)):
        hit = [amap[i] for i, o in enumerate(owner) if o == k and amap[i] >= 0]
        onsets.append(ph[hit[0]][0] if hit else np.nan)
    return rows, word_pron, onsets


def phone_tables(stim: Path, words_dir: Path, out: Path):
    """<song>_<version>_phones.csv, <song>_words_pron.csv and cmudict_nostress.json; asserts that the word onsets of the
    word tables are reproduced."""
    import pronouncing
    out.mkdir(parents=True, exist_ok=True)
    pronouncing.init_cmu()
    cmu = {}
    for w, p in pronouncing.pronunciations:
        q = " ".join(re.sub(r"\d", "", p).lower().split())
        cmu.setdefault(w.lower(), [])
        if q not in cmu[w.lower()]:
            cmu[w.lower()].append(q)
    (out / "cmudict_nostress.json").write_text(json.dumps(cmu))
    for lf in sorted((stim / "lyrics").glob("*.txt")):
        s = lf.stem
        text = lf.read_text(encoding="utf-8").replace("’", "'")
        ws = words_of(text)
        variants = [prons(w) for w, _, _ in ws]
        stored = list(csv.DictReader(open(words_dir / f"{s}_words.csv", encoding="utf-8")))
        assert len(stored) == len(ws)
        prs = {}
        for version in ("song", "speech"):
            rows, word_pron, onsets = phone_table(stim, s, version, ws, variants)
            ref_on = [float(r[f"onset_{version}"]) if r[f"onset_{version}"] not in ("", "nan") else np.nan for r in stored]
            assert all((np.isnan(a) and np.isnan(b)) or a == b for a, b in zip(onsets, ref_on)), (s, version)
            with open(out / f"{s}_{version}_phones.csv", "w", newline="", encoding="utf-8") as f:
                wri = csv.DictWriter(f, fieldnames=list(rows[0]))
                wri.writeheader()
                wri.writerows(rows)
            prs[version] = word_pron
        with open(out / f"{s}_words_pron.csv", "w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=["idx", "word", "in_dict", "pron_song", "pron_speech"])
            wri.writeheader()
            for k, (w, _, _) in enumerate(ws):
                wri.writerow({"idx": k, "word": w, "in_dict": variants[k] is not None,
                              "pron_song": " ".join(prs["song"][k]), "pron_speech": " ".join(prs["speech"][k])})


# ---------------------------------------------------------------- cohort measures
def cohort_values(phones_dir: Path, songs: list[str]) -> tuple[dict, dict]:
    """{(song, timing version, word idx): (surprisal per position, entropy per position, pronunciation)} and counts."""
    LX.check_pins()
    cmu = json.loads((phones_dir / "cmudict_nostress.json").read_text())
    entries, in_vocab = {}, 0
    for w, prs in cmu.items():
        z, inv = LX.zipf(w)
        if not inv:
            continue
        in_vocab += 1
        for p in prs:
            entries[(w, tuple(p.split()))] = entries.get((w, tuple(p.split())), 0.0) + 10 ** z / len(prs)
    lex_words = {k for k, _ in entries}
    qc = {"cmudict_words": len(cmu), "cmudict_words_in_subtlex": in_vocab, "lexicon_entries_from_cmudict": len(entries),
          "lyric_entries_added_word_not_in_lexicon": 0, "lyric_entries_added_pron_not_in_lexicon": 0,
          "lyric_tokens_without_pronunciation": {}, "lyric_entries_added_oov_subtlex": 0}
    lyr = {s: pd.read_csv(phones_dir / f"{s}_words_pron.csv", keep_default_na=False) for s in songs}
    targets, added = {}, set()
    for s, d in lyr.items():
        for r in d.itertuples():
            for tv in ("song", "speech"):
                p = tuple(str(getattr(r, f"pron_{tv}")).split())
                if not bool(r.in_dict) or not p or "?" in p:
                    qc["lyric_tokens_without_pronunciation"][tv] = qc["lyric_tokens_without_pronunciation"].get(tv, 0) + 1
                    continue
                key = LX.form(r.word)
                targets[(s, tv, int(r.idx))] = (key, p)
                if (key, p) not in entries and (key, p) not in added:
                    z, inv = LX.zipf(r.word)
                    entries[(key, p)] = 10 ** z
                    added.add((key, p))
                    qc["lyric_entries_added_word_not_in_lexicon" if key not in lex_words else "lyric_entries_added_pron_not_in_lexicon"] += 1
                    qc["lyric_entries_added_oov_subtlex"] += int(not inv)
    qc["lexicon_entries"] = len(entries)
    need = {p[:k] for _, p in targets.values() for k in range(len(p) + 1)}
    W, words = {}, {}
    for (key, p), w in entries.items():
        for k in range(len(p) + 1):
            pre = p[:k]
            if pre in need:
                W[pre] = W.get(pre, 0.0) + w
                dw = words.setdefault(pre, {})
                dw[key] = dw.get(key, 0.0) + w
    H = {}
    for pre, dw in words.items():
        q = np.fromiter(dw.values(), float)
        q = q / q.sum()
        H[pre] = float(-(q * np.log2(q)).sum())
    out = {}
    for t, (key, p) in targets.items():
        sur = [float(-np.log2(W[p[:k]] / W[p[:k - 1]])) for k in range(1, len(p) + 1)]
        ent = [H[p[:k]] for k in range(1, len(p) + 1)]
        out[t] = (sur, ent, p)
    qc["lyric_tokens_with_pronunciation"] = {tv: sum(1 for t in targets if t[1] == tv) for tv in ("song", "speech")}
    return out, qc


def phone_features(phones_dir: Path, s: str, version: str, n: int, vals: dict) -> dict:
    """Phone-onset impulses (every annotated phone) and the valued phones (sample index, surprisal, entropy, position
    class 0 = word-initial, 1 = later) of one song and version (hummed versions take the sung timing)."""
    tv = g.TIMING[version]
    off = g.annotation_offset(s, tv)
    ph = pd.read_csv(phones_dir / f"{s}_{tv}_phones.csv", keep_default_na=False)
    idx = ((ph.onset.to_numpy(float) + off) * g.FS).round().astype(int)
    keep = idx < n
    on = np.zeros(n)
    np.add.at(on, idx[keep], 1.0)
    cnt = {"phones": int(len(ph)), "beyond_audio": int((~keep).sum()), "not_aligned_to_a_word": 0, "word_without_pronunciation": 0, "valued": 0}
    rows = []
    for r, i, kp in zip(ph.itertuples(), idx, keep):
        if not kp:
            continue
        if int(r.word) < 0:
            cnt["not_aligned_to_a_word"] += 1
            continue
        v = vals.get((s, tv, int(r.word)))
        if v is None:
            cnt["word_without_pronunciation"] += 1
            continue
        sur, ent, p = v
        assert p[int(r.pos) - 1] == r.dict_phone, (s, tv, r.phone)
        rows.append((int(i), sur[int(r.pos) - 1], ent[int(r.pos) - 1], 0 if int(r.pos) == 1 else 1))
    cnt["valued"] = len(rows)
    cnt["valued_word_initial"] = int(sum(1 for x in rows if x[3] == 0))
    aligned_pos = {(int(r.word), int(r.pos)) for r in ph.itertuples() if int(r.word) >= 0}
    cnt["dictionary_phones_without_annotated_phone"] = int(sum(1 for (ss, t, w), (sur, _, _) in vals.items() if ss == s and t == tv
                                                               for k in range(1, len(sur) + 1) if (w, k) not in aligned_pos))
    arr = np.array(rows, float).reshape(-1, 4)
    return {"phone_on": on, "pidx": arr[:, 0].astype(int), "psur": arr[:, 1], "pent": arr[:, 2], "pcls": arr[:, 3].astype(int), "counts": cnt}
