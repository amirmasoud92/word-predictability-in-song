"""Statistics of MEG-MASC.

* Pipeline checks: C1 (envelope tracking > 0 in each session) and C2 (true > mismatched word timing), one-sided sign-flip
  tests; the word-level tests are computed only when both pass.
* Word-level Delta r = mean over channels of r(true) - r(mean over permutations). The Monte Carlo rule is applied per
  test: the first permutation count at which every listener's SE of the null mean (for Test 2, the mean over subsamples)
  is at most 10 % of the between-listener SD, in blocks of 100 up to 500 (Test 2: blocks of 20 up to 60); counts are
  fixed in advance for some sensitivity and exploratory estimates.
* Test 2: per listener, Delta r(M) - Delta r(L) averaged over subsamples and sessions, one-sided; each kind of word
  against zero (list words two-sided, with TOST against +/-0.5); sensitivity analyses (all sentence occurrences of the
  list types with their own surprisal; without the non-native listener sub-21; session 1 only; GPT-2 medium);
  exploratory: list words with the surprisal as heard (two-sided).
* Standard measure in each session (GPT-2; GPT-2 medium; context as heard), and its Spearman correlation across
  sessions (Fisher 95 % CI) in the listeners with two sessions.
* Test 3: the standard measure, session 2 - session 1 (two-sided, JZS BF01, TOST against +/-0.5), with the change in
  envelope tracking; adjusted for the envelope-tracking change and the head displacement between sessions.
Sign-flip tests use 10^5 random flips; d_z with noncentral-t 95 % CIs; random streams are keyed by test label.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

from readers import meg_masc as C
from statistical_tests.inference import dz_ci, jzs_bf10, sign_flip_test, tost_one_sample

N_FLIP = 100_000
WL = C.DER / "word_level"


def one_sample(x, alternative: str, key: str, tost: bool = False) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    p, _ = sign_flip_test(x, alternative, n_perm=N_FLIP, rng=C.rng(f"9|signflip|{key}"))
    lo, hi = dz_ci(x)
    out = {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "dz": float(x.mean() / x.std(ddof=1)),
           "dz_ci95": [float(lo), float(hi)], "p": float(p), "alternative": alternative, "n_positive": int((x > 0).sum()),
           "bf01_two_sided": float(1.0 / jzs_bf10(x, alternative="two-sided"))}
    if tost:
        out["tost_0.5_p"] = float(tost_one_sample(x, 0.5).p)
    return out


def fisher_ci(r: float, n: int) -> list[float]:
    z, se = np.arctanh(r), np.sqrt(1.06 / (n - 3))
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def fisher_ci(r: float, n: int) -> list[float]:
    z, se = np.arctanh(r), np.sqrt(1.06 / (n - 3))
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def load_wl(sub: str, ses: int) -> dict:
    z = dict(np.load(WL / f"sub-{sub}_ses-{ses}.npz"))
    for f in sorted(glob.glob(str(WL / f"sub-{sub}_ses-{ses}_ext*.npz"))):          # Monte Carlo extensions, in order
        e = np.load(f)
        for k in e.files:
            if k.endswith("_rperm"):
                z[k] = np.concatenate([z[k], e[k]], axis=-2)
    return z


def dr_single(z: dict, name: str, n: int | None = None) -> tuple[float, float]:
    """Delta r and the Monte Carlo SE of its null mean, from the first n permutations."""
    rp = z[f"{name}_rperm"]
    rp = rp if n is None else rp[:n]
    nm = rp.mean(1)                                                          # channel means per permutation
    return float(z[f"{name}_rtrue"].mean() - nm.mean()), float(nm.std(ddof=1) / np.sqrt(len(nm)))


def dr_draws(z: dict, name: str, part: str, n: int | None = None) -> tuple[float, float]:
    """Mean over subsamples of Delta r, and the Monte Carlo SE of the subsample-averaged null mean (first n
    permutations per subsample)."""
    rt, rp = z[f"{name}_{part}_rtrue"], z[f"{name}_{part}_rperm"]          # D x ch, D x K x ch
    rp = rp if n is None else rp[:, :n]
    nm = rp.mean(2)                                                          # D x K
    D, K = nm.shape
    return float((rt.mean(1) - nm.mean(1)).mean()), float(np.sqrt((nm.var(1, ddof=1) / K).sum()) / D)


def mc_ok(se: np.ndarray, values: np.ndarray) -> dict:
    sd = float(np.nanstd(values, ddof=1))
    worst = float(np.nanmax(se))
    return {"worst_se": worst, "between_listener_sd": sd, "ratio": worst / sd if sd > 0 else float("inf"), "met": bool(worst <= 0.10 * sd)}


SINGLE = ("std", "std_medium", "std_heard", "listctx", "unm_S", "unm_L")     # 100 permutations, blocks of 100 up to 500
DRAWN = ("match", "match_medium")                                            # 20 per subsample, blocks of 20 up to 60
# Permutation counts fixed in advance for these sensitivity and exploratory estimates: no extension, ratio reported.
FIXED = {"unm_S": 100, "unm_L": 100, "listctx": 100, "match_medium": 20}


def projected(last: int, ratio: float, block: int, cap: int) -> int:
    """Total permutations at which the worst ratio is projected to reach 0.10 (SE ~ 1/sqrt(n)), in whole blocks, at least
    one more block, at most the cap. Computing ahead changes nothing: the first n meeting the rule is still selected."""
    n = int(np.ceil(last * (ratio / 0.10) ** 2 / block) * block)
    return int(min(cap, max(last + block, n)))


def complete(W: pd.DataFrame, col: str) -> bool:
    return col in W and bool(W[col].notna().all())


def monte_carlo(W: pd.DataFrame) -> tuple[dict, dict, dict]:
    """First-n Monte Carlo rule. Single tests: within each session, every listener's
    SE of the null mean at most 10 % of the between-listener SD. Subsampled tests: the listener's subsample- and
    session-averaged null mean, against the SD of the listener values, for M and L. Returns (report, n used, extensions
    needed)."""
    rep, used, need = {}, {}, {}
    for name in SINGLE:
        used[name] = None
        if name in FIXED:
            n = FIXED[name]
            rep[f"{name}_{n}"] = {f"session{s + 1}": mc_ok(W[W.session == s][f"{name}_{n}_se"].to_numpy(), W[W.session == s][f"{name}_{n}"].to_numpy())
                                  for s in (0, 1)}
            used[name] = n
            rep[f"{name}_fixed"] = True
            continue
        for n in range(100, 501, 100):
            if not complete(W, f"{name}_{n}"):
                break
            rule = {f"session{s + 1}": mc_ok(W[W.session == s][f"{name}_{n}_se"].to_numpy(), W[W.session == s][f"{name}_{n}"].to_numpy())
                    for s in (0, 1)}
            rep[f"{name}_{n}"] = rule
            if all(v["met"] for v in rule.values()):
                used[name] = n
                break
        if used[name] is None:
            last = max([n for n in range(100, 501, 100) if complete(W, f"{name}_{n}")], default=0)
            if last < 500:
                need[name] = projected(last, max(v["ratio"] for v in rep[f"{name}_{last}"].values()), 100, 500)
            else:
                used[name] = 500
                rep[f"{name}_cap_reached"] = True
    g = W.groupby("listener")
    for name in DRAWN:
        used[name] = None
        if name in FIXED:
            n = FIXED[name]
            rule = {}
            for part in ("M", "L"):
                se = np.sqrt(g[f"{name}_{part}_{n}_se"].apply(lambda s: float((s ** 2).sum()))) / g.size()
                rule[part] = mc_ok(se.to_numpy(), g[f"{name}_{part}_{n}"].mean().to_numpy())
            rep[f"{name}_{n}"], used[name], rep[f"{name}_fixed"] = rule, n, True
            continue
        for n in (20, 40, 60):
            if not complete(W, f"{name}_M_{n}"):
                break
            rule = {}
            for part in ("M", "L"):
                se = np.sqrt(g[f"{name}_{part}_{n}_se"].apply(lambda s: float((s ** 2).sum()))) / g.size()
                rule[part] = mc_ok(se.to_numpy(), g[f"{name}_{part}_{n}"].mean().to_numpy())
            rep[f"{name}_{n}"] = rule
            if all(v["met"] for v in rule.values()):
                used[name] = n
                break
        if used[name] is None:
            last = max([n for n in (20, 40, 60) if complete(W, f"{name}_M_{n}")], default=0)
            if last < 60:
                need[name] = projected(last, max(v["ratio"] for v in rep[f"{name}_{last}"].values()), 20, 60)
            else:
                used[name] = 60
                rep[f"{name}_cap_reached"] = True
    return rep, used, need


def masc_tests() -> dict:
    """Pipeline checks C1 and C2, the Monte Carlo rule, Test 2 and its sensitivity analyses, the standard measure per
    session, its Spearman correlation across sessions, and Test 3 (session 2 - session 1)."""
    inc = C.inclusion()
    res = {"n_listener_sessions_usable": int(inc.usable.sum()), "two_session_listeners": C.two_session_listeners(),
           "one_session_listeners": C.one_session_listeners()}
    ck = pd.read_csv(C.TAB / "checks_listeners.csv", dtype={"listener": str})
    ck = ck.merge(inc[inc.usable][["listener", "session"]], on=["listener", "session"])
    missing = len(inc[inc.usable]) - len(ck)
    if missing:
        raise SystemExit(f"checks missing for {missing} usable sessions")
    chk = {f"C1_session{s + 1}": one_sample(ck[ck.session == s].c1_r, "greater", f"C1|{s}") for s in (0, 1)}
    c2 = ck.groupby("listener").c2_diff.mean()
    chk["C2"] = one_sample(c2, "greater", "C2")
    chk["pass"] = bool(all(chk[k]["p"] < 0.05 for k in ("C1_session1", "C1_session2", "C2")))
    res["checks"] = chk
    if not chk["pass"]:
        (C.TAB / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return res
    rows = []
    for _, r in inc[inc.usable].iterrows():
        z = load_wl(r.listener, int(r.session))
        row = {"listener": r.listener, "session": int(r.session)}
        for name in SINGLE:
            for n in range(100, z[f"{name}_rperm"].shape[0] + 1, 100):
                row[f"{name}_{n}"], row[f"{name}_{n}_se"] = dr_single(z, name, n)
        for name in DRAWN:
            for n in range(20, z[f"{name}_M_rperm"].shape[1] + 1, 20):
                for part in ("M", "L"):
                    row[f"{name}_{part}_{n}"], row[f"{name}_{part}_{n}_se"] = dr_draws(z, name, part, n)
            row[f"{name}_n_with_replacement"] = int(np.max(z[f"{name}_n_with_replacement"]))
            row[f"{name}_words_L"] = int(np.mean(z[f"{name}_L_nwords"]))
        rows.append(row)
    W = pd.DataFrame(rows)
    mc, used, need = monte_carlo(W)
    res["monte_carlo"] = {"rule": mc, "n_used": used, "extensions_needed": need}
    if need:
        # further permutations with encoding_models.meg_masc_word_level.session_word_level(..., ext=need), then rerun
        (C.TAB / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        W.to_csv(C.TAB / "word_level_listeners.csv", index=False)
        return res
    for name in SINGLE:
        W[name] = W[f"{name}_{used[name]}"]
    for name in DRAWN:
        for part in ("M", "L"):
            W[f"{name}_{part}"] = W[f"{name}_{part}_{used[name]}"]
    W["match_diff"] = W.match_M - W.match_L
    W["unm_diff"] = W.unm_S - W.unm_L
    W["match_medium_diff"] = W.match_medium_M - W.match_medium_L
    per = W.groupby("listener")[["match_diff", "match_M", "match_L", "unm_diff", "match_medium_diff", "listctx"]].mean()
    res["test2"] = {"primary": one_sample(per.match_diff, "greater", "T2"),
                    "lists_two_sided": one_sample(per.match_L, "two-sided", "T2_L", tost=True),
                    "sentences": one_sample(per.match_M, "greater", "T2_M"),
                    "sens_unmatched": one_sample(per.unm_diff, "greater", "T2_unm"),
                    "sens_without_sub21": one_sample(per.drop(index="21", errors="ignore").match_diff, "greater", "T2_no21"),
                    "sens_session1_only": one_sample(W[W.session == 0].set_index("listener").match_diff, "greater", "T2_s1"),
                    "sens_gpt2_medium": one_sample(per.match_medium_diff, "greater", "T2_medium"),
                    "exploratory_list_context": one_sample(per.listctx, "two-sided", "T2_listctx"),
                    "n_with_replacement_max": int(W.match_n_with_replacement.max()), "list_words_mean": float(W.match_words_L.mean())}
    L1 = C.two_session_listeners()
    disp = inc[inc.two_session_set].groupby("listener").head_displacement_mm.first().loc[L1]
    c1 = ck.pivot(index="listener", columns="session", values="c1_r").loc[L1]
    std = W[W.listener.isin(L1)].pivot(index="listener", columns="session", values="std").loc[L1]
    rho = float(stats.spearmanr(std[0], std[1]).correlation)
    res["word_level_across_sessions"] = {"rho": rho, "n": len(L1), "ci95_fisher": fisher_ci(rho, len(L1))}
    d = std[1] - std[0]
    t3 = {"difference": one_sample(d, "two-sided", "T3", tost=True), "session2": one_sample(std[1], "greater", "T3_s2"),
          "session1": one_sample(std[0], "greater", "T3_s1"), "envelope_change": one_sample(c1[1] - c1[0], "two-sided", "T3_env")}
    X = np.column_stack([np.ones(len(d)), (c1[1] - c1[0]) - (c1[1] - c1[0]).mean(), disp - disp.mean()])
    beta, *_ = np.linalg.lstsq(X, d.to_numpy(), rcond=None)
    resid = d.to_numpy() - X @ beta
    sd_r = float(np.sqrt((resid ** 2).sum() / (len(d) - X.shape[1])))
    se_b0 = sd_r / np.sqrt(len(d))
    t3["adjusted"] = {"intercept": float(beta[0]), "dz": float(beta[0] / sd_r), "t": float(beta[0] / se_b0),
                      "p_two_sided": float(2 * stats.t.sf(abs(beta[0] / se_b0), len(d) - X.shape[1]))}
    res["test3"] = t3
    res["standard_measure"] = {f"session{s + 1}": {"gpt2": one_sample(W[W.session == s]["std"], "greater", f"std|{s}"),
                                                   "gpt2_medium": one_sample(W[W.session == s].std_medium, "greater", f"stdm|{s}"),
                                                   "context_as_heard": one_sample(W[W.session == s].std_heard, "greater", f"stdh|{s}")}
                               for s in (0, 1)}
    W.to_csv(C.TAB / "word_level_listeners.csv", index=False)
    (C.TAB / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res
