"""GPT-2 surprisal of every MEG-MASC word, in two contexts.

* Narrative context: the story text (stimuli/text/<story>.txt), without the inserted word lists and pseudo-words. This
  is the surprisal of the sentence words; a list word carries the type mean (stimulus_features/meg_masc_regressors.py).
* Context as heard: the produced texts (stimuli/text_with_wordlists/<story>_produced_<k>.txt, k = sound file),
  concatenated in sound-file order, with the lists and pseudo-words. The context is the whole preceding story as heard,
  including earlier lists; it is not reset at list or sound-file boundaries, only at the start of each story.
* The texts are tokenised by spaces. Clitics ('s, n't, 're, 'll, 've, 'd, 'm) join the preceding word, and punctuation
  joins its neighbour without a space (opening quotes the next word, other marks the previous). Hyphen tokens join the
  words around them without spaces.
* Surprisal of a word: -log2 P(tokens | preceding text), summed over its GPT-2 tokens, with 1024-token windows and a
  512-token stride (stimulus_features/surprisal.py); GPT-2 (124 M) and, in the narrative context, GPT-2 medium.
* Event words (sentence words for the narrative text; every word for the as-heard text) are aligned to the text words
  by sequence matching of lower-cased forms; unaligned words get no value.
"""
from __future__ import annotations

import ast
import difflib
import re

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from common.settings import dataset_root, load_paths
from stimulus_features.surprisal import surprisal_long

DS = dataset_root("meg_masc")
OUT = load_paths().derived_root / "meg_masc" / "words"
STORY = {0: "lw1", 1: "cable_spool_fort", 2: "easy_money", 3: "The_Black_Willow"}
CLITIC = re.compile(r"^(?:'s|n't|'re|'ll|'ve|'d|'m|'S|N'T)$")
CLOSE = set(",.!?;:)]}%") | {"...", "''"}
OPEN = set("([{$") | {"``"}


def norm(w) -> str:
    return re.sub(r"[^a-z0-9']", "", str(w).lower().replace("\u2019", "'"))


def detok(tokens: list[str]) -> tuple[str, list[tuple[int, int]], list[str]]:
    """Text, character spans of the words (clitics merged) and their forms."""
    text, spans, forms, quote_open, glue = "", [], [], False, False
    for t in tokens:
        is_word = bool(re.search(r"[A-Za-z0-9]", t))
        if CLITIC.match(t) and spans:
            a, _ = spans[-1]
            text += t
            spans[-1] = (a, len(text))
            forms[-1] += t
            continue
        if t == '"':
            if quote_open:
                text += t
            else:
                text += ("" if not text else " ") + t
                glue = True
            quote_open = not quote_open
            continue
        if t in CLOSE:
            text += t
            continue
        if t in ("-", "--"):
            text += t
            glue = True
            continue
        sep = "" if (not text or glue or text.endswith(tuple(OPEN))) else " "
        text += sep
        glue = False
        if is_word:
            spans.append((len(text), len(text) + len(t)))
            forms.append(t)
        text += t
    return text, spans, forms


def event_words() -> pd.DataFrame:
    rows = []
    for task, story in STORY.items():
        d = pd.read_csv(DS / "sub-01" / "ses-0" / "meg" / f"sub-01_ses-0_task-{task}_events.tsv", sep="\t")
        tt = pd.DataFrame(list(d.trial_type.map(ast.literal_eval)))
        tt["duration"] = d.duration.to_numpy()
        w = tt[tt.kind == "word"].copy()
        w["story"], w["task"] = story, task
        w["key"] = [f"{story}_{int(float(s))}" for s in w.sound_id]
        rows.append(w)
    w = pd.concat(rows, ignore_index=True)
    w["order"] = np.arange(len(w))
    w["w"] = w.word.map(norm)
    return w


def align(ev_forms: list[str], tx_forms: list[str]) -> np.ndarray:
    """Index into the text words for each event word (-1 = unaligned)."""
    out = np.full(len(ev_forms), -1)
    sm = difflib.SequenceMatcher(None, ev_forms, tx_forms, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        out[a: a + n] = np.arange(b, b + n)
    return out


def word_surprisal():
    OUT.mkdir(parents=True, exist_ok=True)
    w = event_words()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {m: (GPT2TokenizerFast.from_pretrained(m), GPT2LMHeadModel.from_pretrained(m).eval().to(dev)) for m in ("gpt2", "gpt2-medium")}

    def scores(text, spans, name):
        tok, mdl = models[name]
        orig = mdl.forward

        def fwd(input_ids, **kw):                                  # surprisal_long passes CPU tensors
            return orig(input_ids.to(dev), **kw)
        mdl.forward = fwd
        try:
            s = surprisal_long(text, spans, tok, mdl)
        finally:
            mdl.forward = orig
        return np.array(s)

    for col in ("surprisal_gpt2", "surprisal_gpt2_medium", "surprisal_heard_gpt2"):
        w[col] = np.nan
    w["aligned_narrative"] = False
    w["aligned_heard"] = False
    for task, story in STORY.items():
        sel = w.story == story
        # narrative context: sentence words only
        tokens = (DS / "stimuli" / "text" / f"{story.lower()}.txt").read_text(encoding="utf-8").split()
        text, spans, forms = detok(tokens)
        sw = w[sel & (w.condition == "sentence")]
        ix = align(sw.w.tolist(), [norm(f) for f in forms])
        ok = ix >= 0
        for col, m in (("surprisal_gpt2", "gpt2"), ("surprisal_gpt2_medium", "gpt2-medium")):
            s = scores(text, spans, m)
            w.loc[sw.index[ok], col] = s[ix[ok]]
        w.loc[sw.index[ok], "aligned_narrative"] = True
        # context as heard: every word, in sound-file order
        n_files = len({k for k in w[sel].key})
        tokens = []
        for k in range(n_files):
            tokens += (DS / "stimuli" / "text_with_wordlists" / f"{story.lower()}_produced_{k}.txt").read_text(encoding="utf-8").split()
        text, spans, forms = detok(tokens)
        aw = w[sel]
        ix = align(aw.w.tolist(), [norm(f) for f in forms])
        ok = ix >= 0
        s = scores(text, spans, "gpt2")
        w.loc[aw.index[ok], "surprisal_heard_gpt2"] = s[ix[ok]]
        w.loc[aw.index[ok], "aligned_heard"] = True
    keep = ["order", "story", "task", "key", "condition", "word", "w", "start", "duration", "sequence_id", "word_index", "voice",
            "speech_rate", "aligned_narrative", "aligned_heard", "surprisal_gpt2", "surprisal_gpt2_medium", "surprisal_heard_gpt2"]
    w[keep].to_csv(OUT / "words_surprisal.csv", index=False)
    return w[keep]
