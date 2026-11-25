# Neural sensitivity to word predictability survives singing and runs alongside melody

Analysis code and figure-panel data for the article *Neural sensitivity to word predictability survives singing and runs
alongside melody* (Ahmadi & Ghorbani Lohesara).

## Overview

The article measures how strongly the brain tracks the predictability of words, using a word-level gain. An encoding
model predicts held-out EEG or MEG from each word's GPT-2 surprisal. The gain is how much more accurate the prediction
is when each word keeps its own surprisal value than when the values are shuffled among the words of the same song or
passage. The measure is applied to:

- spoken, sung and hummed versions of the same songs;
- narrative speech;
- speech in noise;
- an attended and an ignored talker;
- the same words in sentences and in word lists.

In parallel, the predictability of each note of the melodies is modelled.

This repository contains:

- the implementation of the methods reported in the article;
- for each figure panel that plots data, the plotted values and the code that draws the panel.

The repository is organised by method. It is not a single pipeline to be run from start to finish, and the recordings
are not included (see [Data availability](#data-availability)).

## Key features

- **Word-level gain.** True against permuted surprisal, with leave-one-song-out (or leave-one-passage-out) lagged ridge
  encoding models (lags 0–688 ms) and accuracy measured within 0–800 ms after each word. Two estimates are provided: the
  original one (10 permutations) and a stabilised one (up to 500 permutations, chosen by a Monte Carlo rule).
- **Exact GPU implementation.** The encoding models are computed in double precision on a GPU and give the same result
  as the CPU implementation.
- **EEG and MEG preprocessing.** Robust detrending, bad-channel detection, spherical-spline interpolation, ocular ICA
  and FASTER epoch rejection for EEG; field interpolation, FastICA and reference regression for MEG.
- **Stimulus features.**
  - GPT-2 surprisal and lexical entropy (English and German);
  - word frequency and word-class controls;
  - acoustic and phonetic regressors;
  - phoneme-cohort measures;
  - melodic surprisal (PPM) from notes aligned to the hummed f0;
  - language- and audio-model representations (GPT-2, HuBERT, AST).
- **Statistics.**
  - sign-flip tests and noncentral-t intervals for d<sub>z</sub>;
  - equivalence tests (TOST) and JZS Bayes factors;
  - max-t correction;
  - Holm and Benjamini–Hochberg correction over the 13 primary tests;
  - crossed mixed models (R) and min F′;
  - split-half reliability of the sensor maps and of the listener estimates;
  - listener ICC.
- **Source analysis.** LCMV beamformer on a template head model, with the attended − ignored contrast per
  Desikan–Killiany label.
- **Figure panels.** 37 self-contained panel folders, each with its data and a plotting script. They need no recordings.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt          Analysis environment (Python 3.11)
├── config/
│   ├── analysis_parameters.yaml   Analysis parameters and the base random seed (to be set)
│   └── paths.yaml                 Locations of the datasets and of derived data and results
├── common/                   Configuration loading and reproducible random streams
├── readers/                  Readers of the five public releases
├── preprocessing/            EEG cleaning operators and MEG preprocessing
├── stimulus_features/        Word timing, surprisal, lexical controls, acoustics, melody, model features,
│                             transcription and forced alignment
├── encoding_models/          Ridge encoding models (CPU and GPU), the word-level gain and the analyses per dataset
├── source_analysis/          LCMV beamformer, label time courses, label-level statistics
├── statistical_tests/        Inference, multiplicity corrections, mixed models (with R scripts), reliability
└── figures/
    ├── README.md             List of panels
    ├── style.py              Shared colours, type and marks
    └── figure_<n>/panel_<letter>/   Panel data (CSV/JSON) and plot.py
```

### Methods and their implementation

| Method | Module |
| --- | --- |
| EEG cleaning: robust detrending, bad channels, spherical-spline interpolation, ocular extended-Infomax ICA, FASTER epoch rejection, 1–8 Hz at 64 Hz | `preprocessing/`, `readers/cantisong.py`, `readers/alice.py`, `readers/speech_in_noise.py` |
| MEG preprocessing (field interpolation, FastICA; reference regression for MEG-MASC) | `preprocessing/meg_attention.py`, `preprocessing/meg_masc.py` |
| Alignment of the lyrics to the released phone annotations (Needleman–Wunsch) and the annotation audit | `stimulus_features/lyric_alignment.py`, `stimulus_features/song_annotation.py` |
| GPT-2 surprisal and lexical entropy; word frequency and word class | `stimulus_features/surprisal.py`, `stimulus_features/lexical_entropy.py`, `stimulus_features/english_lexicon.py`, `stimulus_features/lexical_controls.py` |
| Acoustic (envelope, spectrogram and their onsets) and phonetic regressors | `stimulus_features/song_regressors.py`, `stimulus_features/speech_envelope.py` |
| Notes (MIDI aligned to the hummed f0 by dynamic time warping) and melodic surprisal (PPM, short- and long-term models) | `stimulus_features/melody.py` |
| Language- and audio-model representations (GPT-2, HuBERT, AST) | `stimulus_features/model_features.py`, `encoding_models/model_representations.py` |
| Leave-one-out lagged ridge encoding models; exact GPU implementation | `encoding_models/ridge_encoding.py`, `encoding_models/ridge_folds.py`, `encoding_models/gpu_encoding.py`, `encoding_models/sufficient_statistics.py` |
| Word-level gain: original (10 permutations) and stabilised (up to 500, Monte Carlo rule) estimates | `encoding_models/song_word_level.py`, `encoding_models/word_level_trials.py`, `encoding_models/word_level_gain.py`, `encoding_models/alice_word_level.py` |
| Robustness, lexical and acoustic controls, repetition control, response kernels | `encoding_models/robustness.py`, `encoding_models/lexical_controls.py`, `encoding_models/repetition_control.py`, `encoding_models/response_kernels.py` |
| Songs as the sample, leave-one-song-out, lag windows and per-channel maps | `encoding_models/song_level.py` |
| Melodic surprisal and the interference tests | `encoding_models/melodic_surprisal.py` |
| Phoneme and word surprisal beyond each other; lexical × melodic surprisal | `stimulus_features/phoneme_cohort.py`, `encoding_models/phoneme_and_interaction.py` |
| Transfer of the surprisal response between spoken and sung versions (residual method, stretched kernels) | `encoding_models/speech_song_transfer.py` |
| Speech in noise: transcripts of the passages and their alignment; audiovisual against audio-only | `stimulus_features/speech_in_noise_texts.py`, `encoding_models/speech_in_noise_word_level.py` |
| MEG attention: transcription, alignment, German surprisal, the attention test and its checks | `stimulus_features/transcription.py`, `stimulus_features/word_alignment.py`, `stimulus_features/german_text.py`, `stimulus_features/german_surprisal.py`, `stimulus_features/meg_attention_regressors.py`, `encoding_models/meg_attention_checks.py`, `encoding_models/meg_attention_word_level.py`, `statistical_tests/meg_attention_tests.py` |
| MEG-MASC: surprisal, regressors, sentences against lists, the two sessions | `stimulus_features/meg_masc_surprisal.py`, `stimulus_features/meg_masc_regressors.py`, `encoding_models/meg_masc_checks.py`, `encoding_models/meg_masc_word_level.py`, `statistical_tests/meg_masc_tests.py` |
| Source analysis (LCMV, 68 labels, BH-FDR, lateralisation) | `source_analysis/` |
| Sign-flip tests, d<sub>z</sub> intervals, TOST, JZS Bayes factors, max-t; Holm and Benjamini–Hochberg over the 13 primary tests, Holm over the seven additional tests | `statistical_tests/inference.py`, `statistical_tests/max_t.py`, `statistical_tests/primary_tests.py` |
| Sung − spoken equivalence; crossed mixed models and min F′ | `statistical_tests/sung_spoken_equivalence.py`, `statistical_tests/crossed_mixed_models.py` and the R scripts beside it |
| Sensor maps and their split-half reliability | `statistical_tests/sensor_maps.py` |
| Stability of the word-level gain for the listener (split-half reliability, listener ICC) | `statistical_tests/listener_stability.py`, `statistical_tests/listener_stability.R` |

## Requirements

- **Analysis environment.** Python 3.11 with the packages pinned in `requirements.txt`:
  - numpy, scipy, pandas, scikit-learn, statsmodels, joblib;
  - MNE-Python, nibabel, h5py, PyYAML, matplotlib;
  - PyTorch.

  A CUDA GPU is optional. The encoding models run in double precision on the device named by `ENCODING_DEVICE`, for
  example `cuda:0` (the default) or `cpu`.
- **Language and audio models, lyric alignment and melody** (a separate environment): transformers 4.46.3,
  torch 2.14.0, pronouncing 0.2.0 (CMU Pronouncing Dictionary), librosa 0.10.2.post1 (pYIN), mido 1.3.3 and
  scikit-learn 1.9.1. Model weights are downloaded from the Hugging Face Hub.
- **Transcription of the MEG audio** (a separate environment):
  - transformers 5.18.0, num2words 0.5.14, jiwer 4.0.0, soundfile 0.14.0 and pdfplumber;
  - Montreal Forced Aligner 3.4.2 with its pretrained German models.

  `MFA_ENV` and `MFA_WORK` point to the aligner's environment and to its working folder, which must be a path without
  spaces.
- **R 4.4.2** with lme4 1.1-37 and lmerTest 3.1-3. R is used for the crossed mixed models and the listener ICC.
  `RSCRIPT` overrides the `Rscript` executable.
- **The fsaverage template of MNE-Python**, for the source analysis and the cortical panel.
- **Word-frequency and stop-word lists**, placed in `data/lexicon/` (not included; see
  [Data availability](#data-availability)).

The figure panels need only numpy, pandas and matplotlib (plus MNE-Python and nibabel for panel 3d).

## Installation

```bash
git clone https://github.com/amirmasoud92/word-predictability-in-song.git
cd word-predictability-in-song

# Analysis environment
conda create -n word-predictability python=3.11
conda activate word-predictability
pip install -r requirements.txt
```

To install a PyTorch build for a specific CUDA version, follow the instructions at pytorch.org before running
`pip install -r requirements.txt`. With `python -m venv`, the same `pip install` applies.

The two model environments are installed in the same way, with the packages and versions listed under
[Requirements](#requirements). For example:

```bash
conda create -n word-predictability-models python=3.11
conda activate word-predictability-models
pip install -r requirements.txt transformers==4.46.3 pronouncing==0.2.0 librosa==0.10.2.post1 mido==1.3.3
```

## Usage

All modules are imported from the repository root, so run Python there or put the root on `PYTHONPATH`.

### Configuration

Before you run any analysis:

1. Download the datasets and set their locations in `config/paths.yaml`. Paths are relative to the repository root;
   absolute paths also work.
2. Set an integer base seed in `config/analysis_parameters.yaml`:

   ```yaml
   project: {seed: 12345}
   ```

`common.settings.load_analysis()` stops with an error while the seed is unset.

### Draw a figure panel

Each panel folder is self-contained:

```bash
cd figures/figure_2/panel_a
python plot.py            # writes panel.pdf and panel.png in this folder
```

To draw all panels from the repository root:

```bash
for d in figures/figure_*/panel*/; do (cd "$d" && python plot.py); done
```

### Use the statistics on your own estimates

The example below simulates 20 listeners, applies the Monte Carlo rule to the permuted accuracies, summarises the
resulting gains and corrects three p values:

```python
import numpy as np

from encoding_models.word_level_gain import mc_rule
from statistical_tests.inference import summarise
from statistical_tests.primary_tests import holm

rng = np.random.default_rng(1)
true_r = rng.normal(0.050, 0.010, 20)                    # accuracy with the true surprisal, per listener
effect = rng.normal(0.004, 0.004, 20)                    # per-listener gain
null_r = (true_r - effect)[:, None] + rng.normal(0, 0.002, (20, 500))  # accuracy with permuted surprisal

rule = mc_rule(true_r, null_r)                           # number of permutations needed for a stable null
delta_r = true_r - null_r[:, :rule["n_used"]].mean(1)    # word-level gain per listener
print(rule["n_used"], rule["met"])
print(summarise(delta_r, "greater"))                     # n, mean, d_z with 95% CI, sign-flip p, ...
print(holm({"G1": 0.001, "G2": 0.02, "TH1": 0.04}))      # Holm-adjusted p values
```

### Run an analysis on the public data

The analyses need the downloaded recordings and the derived stimulus features. Two examples:

- the transcript alignment of the speech-in-noise passages;
- the Alice tests (the comprehension test needs the release's quiz scores).

```python
from pathlib import Path

from common.settings import dataset_root, load_analysis, load_paths
from stimulus_features.speech_in_noise_texts import match_passages

paths, cfg = load_paths(), load_analysis()

# Speech in noise: align the released content words with the transcripts (c1.txt ... c7.txt, see SOURCES)
rows = match_passages(Path("data/transcripts/speech_in_noise"),
                      dataset_root("speech_in_noise") / "Stimuli",
                      paths.derived_root / "speech_in_noise_texts")

# Alice: word-level gain per listener (TH1) and its association with comprehension (TH2)
from encoding_models.alice_word_level import alice_tests
results, per_listener = alice_tests(paths, seed=cfg.project.seed)
```

Each module's docstring describes its method, inputs and outputs.

## Data availability

All recordings are publicly available from their original sources. They are not redistributed here.

| Dataset | Recording | Listeners | Used for | Source |
| --- | --- | --- | --- | --- |
| CantiSong (Cantisani et al., 2023, 2024) | EEG, 64 channels | 20 | Spoken, sung and hummed versions of 18 songs; melody | [10.5281/zenodo.14590637](https://doi.org/10.5281/zenodo.14590637) |
| Alice (Brennan & Hale, 2019) | EEG, 61 channels | 48 | Narrative speech; comprehension across listeners | [10.7302/746w-g237](https://doi.org/10.7302/746w-g237) |
| Speech in noise (Crosse et al., 2016) | EEG, 128 channels | 21 | Audio-only and audiovisual passages at −9 dB SNR | [10.5061/dryad.070jc](https://doi.org/10.5061/dryad.070jc) |
| Two audiobooks, one attended (Riegel et al., 2024; Schüller et al., 2023, 2025) | MEG, 248 magnetometers | 58 | Attended and ignored talker | [10.5281/zenodo.12793944](https://doi.org/10.5281/zenodo.12793944), [12794769](https://doi.org/10.5281/zenodo.12794769), [12794808](https://doi.org/10.5281/zenodo.12794808) |
| MEG-MASC (Gwilliams et al., 2023) | MEG, 208 gradiometers | 27 (22 in two sessions) | Sentences and word lists; the same stories heard twice | [osf.io/rguwj](https://osf.io/rguwj/) |

The following inputs are also not included and must be obtained from their providers:

- **Speech-in-noise transcripts.** The passages are excerpts of seven weekly addresses. Their transcripts come from the
  archived White House website; the pages are listed in `stimulus_features/speech_in_noise_texts.py` (`SOURCES`).
- **Lexical resources.** The SUBTLEX-US and SUBTLEX-DE word-frequency lists, the Snowball stop-word lists and a German
  stop-word list, placed in `data/lexicon/`.
- **Pretrained models.** The language and audio models (from the Hugging Face Hub) and the forced aligner's German
  models (from the Montreal Forced Aligner).

The figure-panel data in `figures/` are derived values (per-listener, per-song, per-sensor or per-label estimates) and
are included.

## Outputs

- **Derived data.** Preprocessed recordings, stimulus features, transcript alignments and per-listener permutation
  results are written under `derived_root` (default `data/derived/`).
- **Results.** Test results (JSON) and per-listener tables (CSV) are written under `results_root` (default
  `results/`).
- **Figure panels.** `plot.py` writes `panel.pdf` (vector) and `panel.png` next to itself.

`data/`, `results/` and the rendered panels are excluded from version control (`.gitignore`).

## Reproducibility

- **Random seeds.** The base seed in `config/analysis_parameters.yaml` is intentionally empty and must be set before
  running. Every random draw comes from an independent stream that `common.settings.rng(seed, *keys)` derives from the
  base seed and a key path. The key strings in the code are labels of these streams and have no other meaning. The few
  functions with their own draws take a `seed` argument; without one they are unseeded.
- **Published values.** The values in the article were computed with seeds that are not included. The following are
  deterministic and reproduce exactly:
  - exact sign-flip tests (all sign patterns, used up to 22 listeners);
  - the ridge fits;
  - the figure panels.

  Estimates that rest on random permutations, random sign flips or bootstrap resamples will differ from the published
  values by Monte Carlo error. For the stabilised estimates, the Monte Carlo rule limits that error to at most 10% of
  the between-listener SD of the gain wherever the rule is met.
- **Numerical precision.** The GPU encoding models run in float64 and give the same results as the CPU implementation.
  Set `ENCODING_DEVICE=cpu` to run without a GPU.
- **CantiSong annotation.** `CANTISONG_ANNOTATION` selects the released annotation (`released`) or the annotation with
  the corrected timing of song 09 (`corrected`, the default). Each analysis sets the version it reports.
- **Software versions.** The versions are pinned in `requirements.txt` and listed under [Requirements](#requirements).
  The German language models and the transcription models are loaded at pinned revisions. The English GPT-2 models are
  loaded by name (`gpt2`, `gpt2-medium`).

## Citation

If you use this code, please cite the article:

> Ahmadi, A., & Ghorbani Lohesara, F. Neural sensitivity to word predictability survives singing and runs alongside
> melody. *Journal and DOI to be added on publication.*

```bibtex
@article{ahmadi_word_predictability_song,
  author  = {Ahmadi, Amirmasoud and Ghorbani Lohesara, Fatemeh},
  title   = {Neural sensitivity to word predictability survives singing and runs alongside melody},
  journal = {To be added on publication},
  year    = {To be added on publication}
}
```

Please also cite the datasets you use (see [Data availability](#data-availability)).

## Authors

- Amirmasoud Ahmadi, Max Planck Institute for Biological Intelligence, Seewiesen, Germany. Corresponding author:
  Amirmasoud.Ahmadi@bi.mpg.de.
- Fatemeh Ghorbani Lohesara, Communication Systems Group, Technical University Berlin, Germany.

## License

This project is released under the MIT License (see `LICENSE`).
