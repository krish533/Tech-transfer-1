"""Independent audit of the numbers reported in the manuscript.

Each statistic is recomputed from the canonical data with code written
separately from the analysis scripts: regressions are solved with NumPy least
squares and within-university transformations rather than statsmodels
formulas, Shapley shares are computed by enumerating all orders of entry, and
the shift-share is computed university by university with its own loop.
(An earlier version of this audit reused the analysis script's aggregate
shift-share formula and so could not detect that it failed to reconcile.) Only the published
coding rules (keyword patterns for topics and rule forms) are shared with the
analysis code, because they define the quantities being audited.

Each recomputed value is compared with the value the manuscript reports.
Writes paper_outputs/tables/revision_analyses/audit_report.csv and prints a
summary; exits with status 1 if any check fails.

    python paper_outputs/code/audit_manuscript_numbers.py
"""
from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_revision_analyses import DEONTIC, SANCTION, TOPICS  # coding rules only  # noqa: E402

REGIONS = {
    **dict.fromkeys(["ME", "NH", "VT", "MA", "RI", "CT"], "Northeast"),
    **dict.fromkeys(["NY", "NJ", "PA", "DE", "MD", "DC"], "Mid-Atlantic"),
    **dict.fromkeys(["OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"], "Midwest"),
    **dict.fromkeys(["VA", "WV", "KY", "TN", "NC", "SC", "GA", "FL", "AL", "MS", "AR", "LA", "OK", "TX"], "South"),
    **dict.fromkeys(["MT", "ID", "WY", "CO", "NM", "AZ", "UT", "NV"], "Southwest"),
    **dict.fromkeys(["WA", "OR", "CA"], "West Coast"),
}

checks: list[dict] = []


def check(name: str, recomputed: float, reported: float, tol: float) -> None:
    ok = abs(recomputed - reported) <= tol
    checks.append({"Check": name, "Reported": reported, "Recomputed": round(float(recomputed), 6),
                   "Tolerance": tol, "Pass": ok})


def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def r2(y: np.ndarray, X: np.ndarray) -> float:
    b = ols(y, X)
    e = y - X @ b
    return 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))


def demean(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    s = pd.Series(v)
    return (s - s.groupby(g).transform("mean")).to_numpy()


def fe_trend(df: pd.DataFrame, y: str) -> float:
    """Coefficient on year with university fixed effects (within transformation)."""
    g = df["Institution"].to_numpy()
    yy = demean(df[y].to_numpy(float), g)
    xx = demean(df["Year"].to_numpy(float), g)
    return float((xx @ yy) / (xx @ xx))


def main() -> None:
    panel = pd.read_csv(PKG / "data/derived/policy_level_indices_primary_1944_2025.csv", low_memory=False)
    sent = pd.read_csv(PKG / "data/derived/sentence_scores_canonical.csv", low_memory=False)
    sent = sent[sent["Year"].between(1944, 2025)].copy()
    obs = panel[panel["Is_Carried_Forward"] == 0].copy()
    x = panel["Mean_Tone_Score"]

    # --- Sample and distribution --------------------------------------------
    check("Policy-in-force observations", len(panel), 4277, 0)
    check("Source-text university-years", len(obs), 480, 0)
    check("Sentences", len(sent), 87159, 0)
    check("Source documents", sent.groupby(["Institution", "Year", "File"]).ngroups, 518, 0)
    check("Mean PCSI", x.mean(), 0.437, 0.0005)
    check("Median PCSI", x.median(), 0.427, 0.0005)
    check("SD PCSI", x.std(), 0.066, 0.0005)
    check("Share below 0.50 (%)", (x < 0.5).mean() * 100, 86.0, 0.05)
    check("Mean PCSI, source-text years", obs["Mean_Tone_Score"].mean(), 0.429, 0.0005)
    check("Share below 0.50, source-text years (%)", (obs["Mean_Tone_Score"] < 0.5).mean() * 100, 89.6, 0.05)

    # --- Between-university variance shares ---------------------------------
    for frame, rep, lab in [(panel, 0.610, "panel"), (obs, 0.487, "source-text years")]:
        v = frame["Mean_Tone_Score"]
        between = ((frame.groupby("Institution")["Mean_Tone_Score"].transform("mean") - v.mean()) ** 2).sum()
        check(f"Between-university share, {lab}", between / ((v - v.mean()) ** 2).sum(), rep, 0.0005)

    # --- Shapley shares (all 3! orders) ---------------------------------------
    names = ["Tone_Index", "Clarity_Index", "Legal_Load_Index"]
    for frame, rep, lab in [(panel, (0.89, 0.02, 0.09), "panel"), (obs, (0.78, None, None), "source-text years")]:
        d = frame[["Mean_Tone_Score"] + names].dropna()
        y = d["Mean_Tone_Score"].to_numpy(float)

        def fit(sub):
            X = np.column_stack([np.ones(len(d))] + [d[n].to_numpy(float) for n in sub])
            return r2(y, X) if sub else 0.0

        contrib = dict.fromkeys(names, 0.0)
        orders = list(itertools.permutations(names))
        for order in orders:
            for i, n in enumerate(order):
                contrib[n] += (fit(order[: i + 1]) - fit(order[:i])) / len(orders)
        total = fit(tuple(names))
        if lab == "panel":
            check("R2, Tone alone", fit(("Tone_Index",)), 0.216, 0.0005)
            check("R2, all three diagnostics", total, 0.236, 0.0005)
        for n, r in zip(names, rep):
            if r is not None:
                check(f"Shapley share {n}, {lab}", contrib[n] / total, r, 0.005)

    # --- Rule form ------------------------------------------------------------
    text = sent["Sentence_Cleaned"].fillna("").str.lower()
    form = pd.Series("None", index=sent.index)
    for lab in ["Permission", "Obligation", "Prohibition"]:
        form[text.str.contains(DEONTIC[lab], regex=True)] = lab
    sanc = text.str.contains(SANCTION, regex=True)
    cell = form + np.where(sanc, "+S", "")
    s = sent["ToneScore_0_1"]
    cm = s.groupby(cell).transform("mean")
    check("Rule form: sentence-level R2 (%)", (1 - ((s - cm) ** 2).sum() / ((s - s.mean()) ** 2).sum()) * 100, 4.8, 0.05)
    check("Obligation mean sentence score", s[form == "Obligation"].mean(), 0.329, 0.0005)
    check("No-operator mean sentence score", s[form == "None"].mean(), 0.467, 0.0005)
    check("Obligation share of sentences", (form == "Obligation").mean(), 0.314, 0.0005)
    sent["framing"] = s - cm + s.mean()
    fr = sent.groupby(["Institution", "Year"])["framing"].mean().rename("framing").reset_index()
    o2 = obs.merge(fr, on=["Institution", "Year"])
    check("Correlation, PCSI and framing net of rule form", o2["Mean_Tone_Score"].corr(o2["framing"]), 0.943, 0.0005)

    # --- Topic trends -----------------------------------------------------------
    s80 = sent[sent["Year"] >= 1980].copy()
    t80 = s80["Sentence_Cleaned"].fillna("").str.lower()
    flags = {k: t80.str.contains(re.sub(r"\((?!\?)", "(?:", p), regex=True) for k, p in TOPICS.items()}
    reported = {"Ownership & assignment": -0.00031, "Disclosure": -0.00223, "Revenue sharing": -0.00169,
                "Administration & disputes": -0.00134}
    for topic, rep in reported.items():
        g = s80[flags[topic]].groupby(["Institution", "Year"], as_index=False)["ToneScore_0_1"].mean()
        g = g[g.groupby("Institution")["Year"].transform("nunique") > 1]
        check(f"Within-topic FE trend: {topic}", fe_trend(g, "ToneScore_0_1"), rep, 0.000005)
    matched = pd.DataFrame(flags).any(axis=1)
    check("Share of 1980+ sentences matching a topic (%)", matched.mean() * 100, 59.1, 0.05)

    # --- Shift-share (university by university) ------------------------------
    primary = pd.Series("Other", index=s80.index)
    for topic in reversed(list(TOPICS)):  # first listed topic wins
        primary[flags[topic]] = topic
    s80["cat"] = primary
    s80["form_cell"] = cell.loc[s80.index]
    early = s80[s80["Year"].between(1980, 1999)]
    late = s80[s80["Year"].between(2010, 2025)]
    unis = sorted(set(early["Institution"]) & set(late["Institution"]))
    check("Shift-share comparison universities", len(unis), 63, 0)
    # Exact decomposition: split each university's change, then average.
    check("Observed change, balanced universities", np.mean([
        late.loc[late["Institution"] == u, "ToneScore_0_1"].mean() - early.loc[early["Institution"] == u, "ToneScore_0_1"].mean()
        for u in unis]), -0.028816, 0.0000005)
    for col, rep_w, rep_c in [("cat", -0.022229, -0.006587), ("form_cell", -0.027077, -0.001739)]:
        W = C = 0.0
        for u in unis:
            a, z = early[early["Institution"] == u], late[late["Institution"] == u]
            wa, wz = a[col].value_counts(normalize=True), z[col].value_counts(normalize=True)
            ma, mz = a.groupby(col)["ToneScore_0_1"].mean(), z.groupby(col)["ToneScore_0_1"].mean()
            w_u = c_u = 0.0
            for k in set(wa.index) | set(wz.index):
                m0, m1 = ma.get(k, mz.get(k)), mz.get(k, ma.get(k))
                w_u += 0.5 * (wa.get(k, 0) + wz.get(k, 0)) * (m1 - m0)
                c_u += 0.5 * (m0 + m1) * (wz.get(k, 0) - wa.get(k, 0))
            d_u = z["ToneScore_0_1"].mean() - a["ToneScore_0_1"].mean()
            assert abs(w_u + c_u - d_u) < 1e-11
            W += w_u / len(unis)
            C += c_u / len(unis)
        lab = "topics" if col == "cat" else "rule forms"
        check(f"Shift-share within, {lab}", W, rep_w, 0.0000005)
        check(f"Shift-share composition, {lab}", C, rep_c, 0.0000005)

    # --- Institutional comparison, source-text years with year FE ------------
    d = obs.copy()
    d["Region"] = d["STATE"].map(REGIONS)
    d = d.dropna(subset=["Urbanicity (cat)", "Region"])
    X = pd.get_dummies(d[["Type", "Urbanicity (cat)", "Region"]].astype(str), drop_first=False)
    X = X.drop(columns=["Type_Public R1", "Urbanicity (cat)_Rural", "Region_Midwest"])
    X["LG"] = d["Land-Grant Institution"]
    X["Med"] = d["MEDSCHOOL"]
    yrs = pd.get_dummies(d["Year"].astype(int).astype(str), drop_first=True)
    M = np.column_stack([np.ones(len(d)), X.to_numpy(float), yrs.to_numpy(float)])
    b = ols(d["Mean_Tone_Score"].to_numpy(float), M)
    cols = ["const"] + list(X.columns) + list(yrs.columns)
    check("Private R1, source-text years with year FE", b[cols.index("Type_Private R1")], 0.040, 0.0005)
    check("Land-grant, source-text years with year FE", b[cols.index("LG")], -0.005, 0.0005)
    check("Observations, source-text years with year FE", len(d), 452, 0)

    # --- Alternative constructions ----------------------------------------------
    hard = sent["Pred_Label"].map({"restrictive": 0.0, "neutral": 0.5, "supportive": 1.0})
    sent["hard"] = hard
    hl = sent.groupby(["Institution", "Year"])["hard"].mean().rename("hard").reset_index()
    o3 = obs.merge(hl, on=["Institution", "Year"])
    check("Correlation, PCSI and hard-label PCSI", o3["Mean_Tone_Score"].corr(o3["hard"]), 0.980, 0.0005)
    mx = sent[["P_restrictive_calib", "P_neutral_calib", "P_supportive_calib"]].max(axis=1)
    check("Share of sentences with max probability >= 0.9 (%)", (mx >= 0.9).mean() * 100, 65.6, 0.05)

    out = pd.DataFrame(checks)
    dest = PKG / "paper_outputs/tables/revision_analyses/audit_report.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print(f"\n{int(out['Pass'].sum())} of {len(out)} checks pass. Written to {dest}")
    sys.exit(0 if out["Pass"].all() else 1)


if __name__ == "__main__":
    main()
