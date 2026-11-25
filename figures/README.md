# Figure panels

Each data-driven panel of the paper's figures has its own folder, `figure_<n>/panel_<letter>/` (Supplementary Figures:
`figure_S<n>/`), with the values the panel plots and a `plot.py` that draws that panel alone. Run `python plot.py` inside
a panel folder; it writes `panel.pdf` and `panel.png` next to it. `style.py` holds the colours, type and marks shared by
all panels. No script combines panels into a complete figure.

| Figure | Panels with data and code | Content |
| --- | --- | --- |
| 1 | a | The three versions of one song (envelopes, words, notes) |
| 2 | a–d | Word-predictability gain in spoken, sung and hummed lyrics and the paired differences; Alice with its scalp map; per-song gains; robustness and controls |
| 3 | a–d | MEG attention excerpt and sensor layout; sensor maps and listener gains for the attended and ignored talker; controls; cortical labels |
| 4 | a–e | MEG-MASC story excerpt and word lists; sentences against lists; their effect sizes; session maps and the two sessions; session effects |
| 5 | a–e | Words and notes of one song; melodic surprisal; lexical gain on expected and unexpected notes; melodic gain under expected and unexpected words; interference estimates |
| S1 | one panel | The primary tests and their secondary estimates |
| S2 | a–d | Response kernels; repetition control; annotation offsets; corrected timing |
| S3 | a–d | Lag windows; per-channel maps of spoken (b) and sung (c) lyrics, each with the shared colour scale; lexical entropy |
| S4 | a–f | Note-source audit; word-note coincidence; per-song melodic gains; GPT-2, HuBERT and AST layers |
| S5 | a–c | Word surprisal beyond phoneme surprisal; phoneme surprisal beyond word surprisal; lexical × melodic surprisal |

Figure 1b (the word-level measure illustrated on a recording) has no folder, because it plots an excerpt of one
listener's EEG. Figure 1c (the datasets, their sensor layouts and the questions each answers) is a schematic and has no
folder.

Panel data are CSV files (one row per listener, song, sensor, label or estimate) and, for a few summary statistics, small
JSON files. Listener-level values are the per-listener prediction gains (Δr) and effect sizes reported in the paper.
The cortical panel (3d) uses the fsaverage template of MNE-Python (`SUBJECTS_DIR`, or `mne.datasets.fetch_fsaverage()`).
