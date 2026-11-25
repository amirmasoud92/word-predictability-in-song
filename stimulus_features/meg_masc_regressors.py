"""Stimulus regressors per MEG-MASC sound file at 64 Hz.

Per sound file (30):
* acoustic (34 rows, as in the MEG attention dataset): broadband envelope and its onsets, 16-band spectrogram and onsets;
* phonetic (19 rows, the features of the CantiSong models), boxes over each phoneme (ARPAbet; word-position suffix
  stripped);
* word onsets: sentence words and list words (two rows); pseudo-words are excluded from every word regressor;
* controls (4 rows): impulses at every sentence and list word scaled by Zipf frequency (SUBTLEX-US), function word
  (Snowball), duration and peak envelope (maximum of the 1 kHz broadband envelope over the word), each z-scored over the
  sound file's words;
* per word: onset sample, condition, type id (per story), surprisal (GPT-2 narrative; GPT-2 medium; GPT-2 as heard) and
  the type mean (mean narrative surprisal of the type's sentence occurrences in the same story), from which the tested
  regressors and their permutations are built.
Length = the sound file at 64 Hz (ceiling of its 1000 Hz length x 8/125); regressors are band-passed when they are used.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from common.settings import load_analysis
from readers import meg_masc as C
from stimulus_features import english_lexicon as LX, speech_envelope as tk
from stimulus_features.meg_attention_regressors import acoustic, impulses, zs
from stimulus_features.song_regressors import PHON, PHON_NAMES

OUT = C.DER / "features"


def envelope_1k(path: Path) -> np.ndarray:
    fs, x = wavfile.read(path)
    x = np.asarray(x, float)
    if x.dtype.kind != "f" or np.abs(x).max() > 1.5:
        x = x / 32768.0
    x = x.mean(1) if x.ndim > 1 else x
    return tk.broadband_envelope(x, fs, load_analysis().section("envelope"))


def sound_file_regressors():
    OUT.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(C.DER / "words" / "words_surprisal.csv")
    w = w[w.condition.isin(["sentence", "word_list"])].copy()
    w["type_id"] = -1
    for story, g in w.groupby("story"):
        types = {t: i for i, t in enumerate(sorted(g.w.unique()))}
        w.loc[g.index, "type_id"] = g.w.map(types)
        sent = g[g.condition == "sentence"]
        for col in ("surprisal_gpt2", "surprisal_gpt2_medium"):
            tm = sent.groupby("w")[col].mean()
            w.loc[g.index, f"typemean_{col.split('_', 1)[1]}"] = g.w.map(tm)
        list_types = set(g[g.condition == "word_list"].w)
        w.loc[g.index, "list_type"] = g.w.isin(list_types)
    w["function"] = w.word.map(LX.is_function)
    w["zipf"] = [LX.zipf(x)[0] for x in w.word]
    _, _, phones = C.stimulus_tables()
    rows = []
    for key in C.sound_keys():
        path = C.audio_path(key)
        fs, x = wavfile.read(path)
        n1000 = int(math.ceil(len(x) * 1000 / fs))
        n = int(math.ceil(n1000 * 8 / 125))
        e1k = envelope_1k(path)
        g = w[w.key == key].sort_values("order").copy()
        g["peak_env"] = [float(e1k[int(a * 1000): max(int((a + d) * 1000), int(a * 1000) + 1)].max()) if int(a * 1000) < e1k.size else np.nan
                         for a, d in zip(g.start, g.duration)]
        w.loc[g.index, "peak_env"] = g.peak_env
        idx = np.round(g.start.to_numpy() * C.FS).astype(int)
        is_list = (g.condition == "word_list").to_numpy()
        P = np.zeros((len(PHON_NAMES), n))
        ph = phones[phones.key == key]
        for a, d, lab in zip(ph.t, ph.duration, ph.phoneme):
            p = str(lab).split("_")[0].lower()
            if p in PHON:
                for name in PHON[p].split():
                    P[PHON_NAMES.index(name), int(a * C.FS): max(int((a + d) * C.FS), int(a * C.FS) + 1)] = 1.0
        arrays = {"n": np.array(n), "n1000": np.array(n1000),
                  "acoustic_raw": acoustic(path, n), "phonetic_raw": P,
                  "word_sent_raw": impulses(idx[~is_list], [np.ones((~is_list).sum())], n),
                  "word_list_raw": impulses(idx[is_list], [np.ones(is_list.sum())], n),
                  "controls_raw": impulses(idx, [zs(g.zipf), zs(g.function.astype(float)), zs(g.duration), zs(g.peak_env)], n),
                  "onset_idx": idx, "is_list": is_list, "type_id": g.type_id.to_numpy().astype(int), "order": g.order.to_numpy().astype(int),
                  "list_type": g.list_type.to_numpy().astype(bool),
                  "s_gpt2": g.surprisal_gpt2.to_numpy(float), "s_gpt2_medium": g.surprisal_gpt2_medium.to_numpy(float),
                  "s_heard": g.surprisal_heard_gpt2.to_numpy(float),
                  "tm_gpt2": g.typemean_gpt2.to_numpy(float), "tm_gpt2_medium": g.typemean_gpt2_medium.to_numpy(float)}
        np.savez(OUT / f"{key}.npz", **arrays)
        rows.append({"key": key, "n64": n, "sentence_words": int((~is_list).sum()), "list_words": int(is_list.sum()),
                     "sentence_with_surprisal": int(np.isfinite(arrays["s_gpt2"][~is_list]).sum()),
                     "list_with_typemean": int(np.isfinite(arrays["tm_gpt2"][is_list]).sum()),
                     "sentence_of_list_types": int(arrays["list_type"][~is_list].sum())})
    c = pd.DataFrame(rows)
    C.TAB.mkdir(parents=True, exist_ok=True)
    c.to_csv(C.TAB / "feature_counts.csv", index=False)
    cols = ["order", "story", "key", "condition", "start", "duration", "type_id", "list_type", "function", "zipf", "peak_env", "voice",
            "speech_rate", "surprisal_gpt2", "surprisal_gpt2_medium", "surprisal_heard_gpt2", "typemean_gpt2", "typemean_gpt2_medium"]
    w[cols].sort_values("order").to_csv(C.TAB / "word_features.csv", index=False)
