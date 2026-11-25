"""Listener-level statistics of the MEG attention dataset.

* QC: bad channels and removed ICA components per recording, rejected epochs per chapter, dropped chapters.
* Pipeline checks: positive control (attended > ignored envelope tracking), timing gate (true > next chapter for the
  ignored stream; attended stream as a check), per-listener flags (true ignored r not above the mean of the four
  mismatched chapters).
* Test A: Monte Carlo rule per stream and variant (the first n in 100, 200, ... at which the largest listener's Monte
  Carlo SE of the null mean is at most 10 % of the between-listener SD of the gain); the gain per listener; the primary
  attended - ignored contrast (one-sided sign-flip test, 10^5 flips) and its sensitivity analyses (text-matched words;
  each recording; the same talker attended and ignored; without flagged listeners; gerpt2-large). d_z with noncentral-t
  95 % CIs, JZS BF01 (two-sided).
* The word-level gain of the attended stream against comprehension (Spearman, one-sided permutation test of ranks,
  10^5; Fisher CI).
Random streams are keyed by test label.
"""
from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd
from scipy import stats

from readers import meg_attention as C
from statistical_tests.inference import dz_ci, jzs_bf10, sign_flip_test

N_FLIP, N_PERM = 100_000, 100_000
VARIANTS = {"primary": 500, "matched": 500, "large": 100}
SUB = {"all": 0, "LP": 1, "HP": 2}


def seed(key: str) -> int:
    return zlib.crc32(f"7|{key}".encode())


def one_sample(x, alternative: str, key: str) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    p, _ = sign_flip_test(x, alternative, n_perm=N_FLIP, rng=C.rng(f"p7|signflip|{key}"))
    lo, hi = dz_ci(x)
    out = {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "dz": float(x.mean() / x.std(ddof=1)),
           "dz_ci95": [float(lo), float(hi)], "p": float(p), "alternative": alternative, "n_positive": int((x > 0).sum())}
    out["bf01_two_sided"] = float(1.0 / jzs_bf10(x, alternative="two-sided"))
    return out


def spearman_perm(x, y, key: str, two_sided: bool = False) -> dict:
    """Spearman rho tested by permutation of ranks (10^5): one-sided (rho > 0) or two-sided."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.spearmanr(x, y).correlation
    g = np.random.default_rng(seed(key))
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    null = np.array([np.corrcoef(rx, g.permutation(ry))[0, 1] for _ in range(N_PERM)])
    p = (1 + (np.abs(null) >= abs(r)).sum()) / (1 + N_PERM) if two_sided else (1 + (null >= r).sum()) / (1 + N_PERM)
    return {"rho": float(r), "p": float(p), "n": int(x.size), "alternative": "two-sided" if two_sided else "greater"}


def fisher_ci(r: float, n: int) -> list[float]:
    z, se = np.arctanh(r), np.sqrt(1.06 / (n - 3))
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def qc_tables() -> tuple[pd.DataFrame, dict]:
    files, ica, chap = [], [], []
    for p in range(1, 59):
        q = C.qc(p)
        for tag, v in q["files"].items():
            files.append({"listener": p, "file": tag, "n_bad_channels": len(v["bad_channels"]),
                          "bad_channels": ";".join(v["bad_channels"]), "n_ica_flagged": len(v["ica"]["flagged"]),
                          "n_ica_removed": len(v["ica"]["removed"]), "ecg_error": v["ica"]["ecg_error"],
                          "eog_error": v["ica"]["eog_error"], "length_minus_audio": v["length_minus_audio"]})
            for c in v["ica"]["removed"]:
                ica.append({"listener": p, "file": tag, **c})
        for c in q["chapters"]:
            chap.append({"listener": p, **c})
    F, I, H = pd.DataFrame(files), pd.DataFrame(ica), pd.DataFrame(chap)
    F.to_csv(C.TAB / "qc_files.csv", index=False)
    I.to_csv(C.TAB / "qc_ica_removed.csv", index=False)
    H.to_csv(C.TAB / "qc_chapters.csv", index=False)
    lst = pd.DataFrame([{"listener": p, "included": C.qc(p)["included"], "reject_fraction": C.qc(p)["reject_fraction"],
                         "dropped": ";".join(C.qc(p)["dropped_chapters"])} for p in range(1, 59)])
    summ = {"files_with_bad_channels": int((F.n_bad_channels > 0).sum()), "bad_channels_total": int(F.n_bad_channels.sum()),
            "bad_channels_max_per_file": int(F.n_bad_channels.max()),
            "ica_removed_per_file": F.n_ica_removed.value_counts().sort_index().to_dict(),
            "ica_removed_by_type": I.type.value_counts().to_dict() if len(I) else {},
            "ecg_errors": int(F.ecg_error.notna().sum()), "eog_errors": int(F.eog_error.notna().sum()),
            "rejected_fraction_mean": float(H.rejected_fraction.mean()), "rejected_fraction_max": float(H.rejected_fraction.max()),
            "chapters_dropped": int((~H.kept).sum()), "listeners_included": int(lst.included.sum()),
            "listeners_excluded": lst.loc[~lst.included, "listener"].tolist()}
    return lst, summ


def load_A(listeners: list[int]) -> dict:
    out = {}
    for p in listeners:
        with np.load(C.DER / "A" / f"P{p:02d}.npz") as z:
            if "excluded" in z.files:
                continue
            out[p] = {k: z[k] for k in z.files}
    return out


def mc_rule(A: dict, variant: str, stream: str) -> tuple[int, list[dict]]:
    trail = []
    n_max = VARIANTS[variant]
    for n in range(100, n_max + 1, 100):
        dr, se = [], []
        for d in A.values():
            null = d[f"{variant}_{stream}_rperm"][:n, 0, :].mean(1)
            dr.append(d[f"{variant}_{stream}_rtrue"][0].mean() - null.mean())
            se.append(null.std(ddof=1) / np.sqrt(n))
        ratio = float(np.max(se) / np.std(dr, ddof=1))
        trail.append({"n": n, "max_ratio": ratio})
        if ratio <= 0.1:
            return n, trail
    return n_max, trail


def delta_r(A: dict, variant: str, stream: str, n: int, sub: str) -> pd.Series:
    i = SUB[sub]
    return pd.Series({p: float((d[f"{variant}_{stream}_rtrue"][i] - d[f"{variant}_{stream}_rperm"][:n, i].mean(0)).mean())
                      for p, d in A.items()})


def listener_tests() -> dict:
    """Pipeline checks, Test A with its sensitivity analyses, and the word-level gain of the attended stream against
    comprehension (one-sided)."""
    res = {}
    lst, res["qc"] = qc_tables()
    inc = lst.loc[lst.included, "listener"].tolist()

    ck = pd.read_csv(C.TAB / "checks_listeners.csv")
    ck = ck[ck.included.astype(bool)].set_index("listener")
    chk = {"positive_control": one_sample(ck.r_attended_true - ck.r_ignored_true, "greater", "pc"),
           "ignored_gate": one_sample(ck.r_ignored_true - ck.r_ignored_mis1, "greater", "gate_ign"),
           "attended_check": one_sample(ck.r_attended_true - ck.r_attended_mis1, "greater", "gate_att"),
           "r_attended_mean": float(ck.r_attended_true.mean()), "r_ignored_mean": float(ck.r_ignored_true.mean())}
    mis_mean = ck[[f"r_ignored_mis{s}" for s in range(1, 5)]].mean(1)
    flagged = ck.index[ck.r_ignored_true <= mis_mean].tolist()
    chk["flagged_listeners"] = [int(x) for x in flagged]
    chk["positive_control_pass"] = chk["positive_control"]["p"] < 0.05
    chk["ignored_gate_pass"] = chk["ignored_gate"]["p"] < 0.05
    res["checks"] = chk

    A = load_A(inc)
    n_used, trails, drs = {}, {}, {}
    for v in VARIANTS:
        for s in C.STREAMS:
            n, trail = mc_rule(A, v, s)
            n_used[f"{v}_{s}"], trails[f"{v}_{s}"] = n, trail
            for sub in SUB:
                drs[(v, s, sub)] = delta_r(A, v, s, n, sub)
    tab = pd.DataFrame({f"dr_{v}_{s}_{sub}": x for (v, s, sub), x in drs.items()})
    tab.index.name = "listener"
    tab.to_csv(C.TAB / "word_level_listeners.csv")
    prim = drs[("primary", "attended", "all")] - drs[("primary", "ignored", "all")]
    res["A"] = {"n_permutations_used": n_used, "monte_carlo_trail": trails,
                "primary": one_sample(prim, "greater", "A_primary"),
                "attended": one_sample(drs[("primary", "attended", "all")], "greater", "A_att"),
                "ignored": one_sample(drs[("primary", "ignored", "all")], "two-sided", "A_ign"),
                "matched": one_sample(drs[("matched", "attended", "all")] - drs[("matched", "ignored", "all")], "greater", "A_matched"),
                "matched_attended": one_sample(drs[("matched", "attended", "all")], "greater", "A_matched_att"),
                "matched_ignored": one_sample(drs[("matched", "ignored", "all")], "two-sided", "A_matched_ign"),
                "within_file_LP": one_sample(drs[("primary", "attended", "LP")] - drs[("primary", "ignored", "LP")], "greater", "A_fileLP"),
                "within_file_HP": one_sample(drs[("primary", "attended", "HP")] - drs[("primary", "ignored", "HP")], "greater", "A_fileHP"),
                "speaker_LP_book1_vs_book2": one_sample(drs[("primary", "attended", "LP")] - drs[("primary", "ignored", "HP")], "greater", "A_spkLP"),
                "speaker_HP_book3_vs_book4": one_sample(drs[("primary", "attended", "HP")] - drs[("primary", "ignored", "LP")], "greater", "A_spkHP"),
                "without_flagged": one_sample(prim.drop([p for p in flagged if p in prim.index]), "greater", "A_noflag"),
                "large": one_sample(drs[("large", "attended", "all")] - drs[("large", "ignored", "all")], "greater", "A_large"),
                "large_attended": one_sample(drs[("large", "attended", "all")], "greater", "A_large_att")}

    part = C.participants().set_index("participant")
    d = pd.DataFrame({"comp": part.comprehension, "dr": drs[("primary", "attended", "all")]})
    d = d.loc[[p for p in inc if p in d.index]]
    b_df = d.dropna(subset=["comp", "dr"])
    rho_d = spearman_perm(b_df.dr, b_df.comp, "B_rho_dr")
    res["comprehension"] = {**rho_d, "ci95_fisher": fisher_ci(rho_d["rho"], len(b_df))}
    (C.TAB / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res
