from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from adni_analysis.config import load_yaml
from adni_analysis.utils import ensure_dir


Direction = Literal["higher_is_more_positive", "lower_is_more_positive"]


@dataclass(frozen=True)
class PaperBArtifacts:
    dataset_one_per_rid_all: Path
    pattern_posteriors: Path
    representativeness: Path
    timing_observed: Path
    timing_ppc: Path
    sensitivity_key_posteriors: Path


_Z_95 = 1.959963984540054


def _apply_journal_plot_style(plt: object) -> None:
    try:
        rcparams = plt.rcParams  # type: ignore[attr-defined]
    except Exception:
        return
    rcparams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _add_panel_label(ax: object, label: str) -> None:
    try:
        ax.annotate(  # type: ignore[attr-defined]
            label,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    except Exception:
        return


def _format_prob_cell(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "<0.01"
    if p > 0.99:
        return ">0.99"
    return f"{p:.2f}"


def _text_color_for_cmap_value(cmap: object, value: float) -> str:
    try:
        rgba = cmap(float(value))  # type: ignore[call-arg]
        r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    except Exception:
        return "black"
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luma > 0.55 else "white"


def _wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (_Z_95**2) / n
    center = (p + (_Z_95**2) / (2 * n)) / denom
    half = (_Z_95 * np.sqrt((p * (1 - p) + (_Z_95**2) / (4 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (float(lo), float(hi))


def _stable_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:4], "big")


def _as_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _attach_nearest_clinical(triads: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    triads = _as_date(triads, ["pet_date"])
    clinical = _as_date(clinical, ["EXAMDATE"])

    triads = triads.copy()
    clinical = clinical.copy()
    triads["RID"] = pd.to_numeric(triads.get("RID"), errors="coerce")
    clinical["RID"] = pd.to_numeric(clinical.get("RID"), errors="coerce")
    triads = triads.dropna(subset=["RID"]).copy()
    clinical = clinical.dropna(subset=["RID"]).copy()
    triads["RID"] = triads["RID"].astype("int64")
    clinical["RID"] = clinical["RID"].astype("int64")

    clinical = clinical.rename(columns={"EXAMDATE": "clin_examdate"})
    clinical = clinical.dropna(subset=["clin_examdate"]).copy()

    keep_cols = [c for c in clinical.columns if c not in {"row_uid", "source_file", "source_row"}]
    clinical = clinical.loc[:, keep_cols]

    rename_map: dict[str, str] = {}
    for c in clinical.columns:
        if c in {"RID", "clin_examdate"}:
            continue
        if c == "examdate":
            rename_map[c] = "clin_examdate_str"
        else:
            rename_map[c] = f"clin_{c}"
    clinical = clinical.rename(columns=rename_map)

    triads_work = triads.copy()
    triads_work["_triad_row"] = np.arange(len(triads_work), dtype="int64")
    triads_work = triads_work.dropna(subset=["pet_date"]).copy()

    triads_sorted = triads_work.sort_values(["pet_date", "RID"], kind="mergesort")
    clinical_sorted = clinical.sort_values(["clin_examdate", "RID"], kind="mergesort")

    merged = pd.merge_asof(
        triads_sorted,
        clinical_sorted,
        left_on="pet_date",
        right_on="clin_examdate",
        by="RID",
        direction="nearest",
    )
    merged["delta_pet_clin_days"] = (
        (merged["clin_examdate"] - merged["pet_date"]).abs() / np.timedelta64(1, "D")
    ).astype("float64")
    merged = merged.sort_values(["_triad_row"], kind="mergesort").drop(columns=["_triad_row"])
    return merged


def _select_one_triad_per_rid(df: pd.DataFrame, *, csf_metric_col: str) -> pd.DataFrame:
    df = df.copy()
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    df = df.dropna(subset=["RID"]).copy()
    df["RID"] = df["RID"].astype("int64")

    csf_metric = df.get(csf_metric_col)
    df["_has_csf"] = pd.to_numeric(csf_metric, errors="coerce").notna().astype("int64")

    delta_pet_plasma = pd.to_numeric(df.get("delta_pet_plasma_days"), errors="coerce").fillna(np.inf)
    delta_pet_csf = pd.to_numeric(df.get("delta_pet_csf_days"), errors="coerce").fillna(np.inf)
    delta_plasma_csf = pd.to_numeric(df.get("delta_plasma_csf_days"), errors="coerce").fillna(np.inf)

    df["_max_gap"] = np.maximum.reduce(
        [delta_pet_plasma.to_numpy(), delta_pet_csf.to_numpy(), delta_plasma_csf.to_numpy()]
    )
    df["_sum_gap"] = delta_pet_plasma + delta_pet_csf + delta_plasma_csf
    df["_delta_pet_plasma"] = delta_pet_plasma
    df["_delta_pet_csf"] = delta_pet_csf
    df["_delta_plasma_csf"] = delta_plasma_csf
    df["_pet_date"] = pd.to_datetime(df.get("pet_date"), errors="coerce")
    df["_pet_event_id"] = df.get("pet_event_id").astype("string") if "pet_event_id" in df.columns else pd.NA

    df = df.sort_values(
        [
            "RID",
            "_has_csf",
            "_max_gap",
            "_sum_gap",
            "_delta_plasma_csf",
            "_delta_pet_csf",
            "_delta_pet_plasma",
            "_pet_date",
            "_pet_event_id",
        ],
        ascending=[True, False, True, True, True, True, True, True, True],
        kind="mergesort",
    )

    out = df.groupby("RID", as_index=False, sort=False).head(1).copy()
    drop_cols = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop_cols)


def _plasma_score(series: pd.Series, direction: Direction) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if direction == "higher_is_more_positive":
        return x
    if direction == "lower_is_more_positive":
        return -x
    raise ValueError(f"Unknown direction: {direction}")


def _assign_quantile_strata(
    score: pd.Series, *, low_q: float, high_q: float
) -> tuple[pd.Series, dict[str, float]]:
    score = pd.to_numeric(score, errors="coerce")
    q_low = float(score.quantile(low_q))
    q_high = float(score.quantile(high_q))

    def _cat(v: float) -> str:
        if pd.isna(v):
            return ""
        if v <= q_low:
            return "Low"
        if v >= q_high:
            return "High"
        return "Intermediate"

    cat = score.map(_cat).astype("string")
    return cat, {"q_low": q_low, "q_high": q_high}


def _triage_categories_from_paper_a_thresholds(
    df: pd.DataFrame,
    *,
    endpoint: str,
    direction: Direction,
    thresholds: pd.DataFrame,
    operating_point: str,
) -> pd.Series:
    row = thresholds[(thresholds["endpoint"] == endpoint) & (thresholds["operating_point"] == operating_point)].copy()
    if row.empty:
        raise ValueError(f"No Paper A thresholds found for endpoint={endpoint} operating_point={operating_point}")
    r0 = row.iloc[0]
    t_negative = float(r0["t_negative"])
    t_positive = float(r0["t_positive"])

    x = pd.to_numeric(df[endpoint], errors="coerce")
    out = pd.Series(pd.NA, index=df.index, dtype="string")

    if direction == "higher_is_more_positive":
        out = out.mask(x <= t_negative, "Low")
        out = out.mask(x >= t_positive, "High")
        out = out.fillna("Intermediate")
        return out

    if direction == "lower_is_more_positive":
        out = out.mask(x >= t_negative, "Low")
        out = out.mask(x <= t_positive, "High")
        out = out.fillna("Intermediate")
        return out

    raise ValueError(f"Unknown direction: {direction}")


def _rhat(chains: list[np.ndarray]) -> float:
    if len(chains) < 2:
        return float("nan")
    n = min(len(c) for c in chains)
    if n < 2:
        return float("nan")
    m = len(chains)
    x = np.stack([c[:n] for c in chains], axis=0)
    chain_means = x.mean(axis=1)
    grand_mean = chain_means.mean()
    b = (n / (m - 1)) * np.sum((chain_means - grand_mean) ** 2)
    w = float(np.mean(np.var(x, axis=1, ddof=1)))
    var_hat = ((n - 1) / n) * w + (1 / n) * b
    return float(np.sqrt(var_hat / w)) if w > 0 else float("nan")


def _fit_lcm_independence(
    *,
    pet: np.ndarray,
    csf: np.ndarray,
    plasma_cat: np.ndarray,
    priors: dict,
    chains: int,
    burn_in: int,
    draws: int,
    thin: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    n = int(len(pet))
    a_pi, b_pi = (float(priors["prevalence_beta"][0]), float(priors["prevalence_beta"][1]))
    a_s, b_s = (float(priors["sensor_beta"][0]), float(priors["sensor_beta"][1]))
    alpha_plasma = np.asarray(priors["plasma_dirichlet"], dtype=float)

    chain_pi: list[np.ndarray] = []
    chain_pet1: list[np.ndarray] = []
    chain_csf1: list[np.ndarray] = []

    out_all: dict[str, list[np.ndarray]] = {
        "pi": [],
        "p_pet_0": [],
        "p_pet_1": [],
        "p_csf_0": [],
        "p_csf_1": [],
        "plasma_p0": [],
        "plasma_p1": [],
    }

    for ch in range(int(chains)):
        rng = np.random.default_rng(int(seed) + 1009 * ch)
        z = rng.integers(0, 2, size=n, dtype=np.int8)

        pi = float(rng.beta(a_pi, b_pi))
        p_pet_0 = float(rng.beta(a_s, b_s))
        p_pet_1 = float(rng.beta(a_s, b_s))
        p_csf_0 = float(rng.beta(a_s, b_s))
        p_csf_1 = float(rng.beta(a_s, b_s))
        plasma_p0 = rng.dirichlet(alpha_plasma).astype(float)
        plasma_p1 = rng.dirichlet(alpha_plasma).astype(float)

        total_iters = int(burn_in) + int(draws) * int(thin)
        keep_pi: list[float] = []
        keep_pet0: list[float] = []
        keep_pet1: list[float] = []
        keep_csf0: list[float] = []
        keep_csf1: list[float] = []
        keep_pl0: list[np.ndarray] = []
        keep_pl1: list[np.ndarray] = []

        keep_pi_chain: list[float] = []
        keep_pet1_chain: list[float] = []
        keep_csf1_chain: list[float] = []

        saved = 0
        for it in range(total_iters):
            lp1 = (
                np.log(pi + 1e-12)
                + pet * np.log(p_pet_1 + 1e-12)
                + (1 - pet) * np.log(1.0 - p_pet_1 + 1e-12)
                + csf * np.log(p_csf_1 + 1e-12)
                + (1 - csf) * np.log(1.0 - p_csf_1 + 1e-12)
                + np.log(plasma_p1[plasma_cat] + 1e-12)
            )
            lp0 = (
                np.log(1.0 - pi + 1e-12)
                + pet * np.log(p_pet_0 + 1e-12)
                + (1 - pet) * np.log(1.0 - p_pet_0 + 1e-12)
                + csf * np.log(p_csf_0 + 1e-12)
                + (1 - csf) * np.log(1.0 - p_csf_0 + 1e-12)
                + np.log(plasma_p0[plasma_cat] + 1e-12)
            )
            m = np.maximum(lp0, lp1)
            p1 = np.exp(lp1 - m) / (np.exp(lp0 - m) + np.exp(lp1 - m))
            z = (rng.random(n) < p1).astype(np.int8)

            n1 = int(z.sum())
            n0 = n - n1
            pi = float(rng.beta(a_pi + n1, b_pi + n0))

            pet_1 = int(pet[z == 1].sum())
            pet_0 = int(pet[z == 0].sum())
            csf_1 = int(csf[z == 1].sum())
            csf_0 = int(csf[z == 0].sum())

            p_pet_1 = float(rng.beta(a_s + pet_1, b_s + (n1 - pet_1)))
            p_pet_0 = float(rng.beta(a_s + pet_0, b_s + (n0 - pet_0)))
            p_csf_1 = float(rng.beta(a_s + csf_1, b_s + (n1 - csf_1)))
            p_csf_0 = float(rng.beta(a_s + csf_0, b_s + (n0 - csf_0)))

            plasma_counts_1 = np.bincount(plasma_cat[z == 1], minlength=3).astype(float)
            plasma_counts_0 = np.bincount(plasma_cat[z == 0], minlength=3).astype(float)
            plasma_p1 = rng.dirichlet(alpha_plasma + plasma_counts_1).astype(float)
            plasma_p0 = rng.dirichlet(alpha_plasma + plasma_counts_0).astype(float)

            # Identifiability constraint: A*=1 has higher PET positivity rate.
            if p_pet_1 < p_pet_0:
                pi = 1.0 - pi
                p_pet_0, p_pet_1 = p_pet_1, p_pet_0
                p_csf_0, p_csf_1 = p_csf_1, p_csf_0
                plasma_p0, plasma_p1 = plasma_p1, plasma_p0
                z = (1 - z).astype(np.int8)

            if it >= int(burn_in) and ((it - int(burn_in)) % int(thin) == 0):
                keep_pi.append(pi)
                keep_pet0.append(p_pet_0)
                keep_pet1.append(p_pet_1)
                keep_csf0.append(p_csf_0)
                keep_csf1.append(p_csf_1)
                keep_pl0.append(plasma_p0.copy())
                keep_pl1.append(plasma_p1.copy())
                keep_pi_chain.append(pi)
                keep_pet1_chain.append(p_pet_1)
                keep_csf1_chain.append(p_csf_1)
                saved += 1
                if saved >= int(draws):
                    break

        out_all["pi"].append(np.asarray(keep_pi, dtype=float))
        out_all["p_pet_0"].append(np.asarray(keep_pet0, dtype=float))
        out_all["p_pet_1"].append(np.asarray(keep_pet1, dtype=float))
        out_all["p_csf_0"].append(np.asarray(keep_csf0, dtype=float))
        out_all["p_csf_1"].append(np.asarray(keep_csf1, dtype=float))
        out_all["plasma_p0"].append(np.asarray(keep_pl0, dtype=float))
        out_all["plasma_p1"].append(np.asarray(keep_pl1, dtype=float))

        chain_pi.append(np.asarray(keep_pi_chain, dtype=float))
        chain_pet1.append(np.asarray(keep_pet1_chain, dtype=float))
        chain_csf1.append(np.asarray(keep_csf1_chain, dtype=float))

    out = {
        "pi": np.concatenate(out_all["pi"], axis=0),
        "p_pet_0": np.concatenate(out_all["p_pet_0"], axis=0),
        "p_pet_1": np.concatenate(out_all["p_pet_1"], axis=0),
        "p_csf_0": np.concatenate(out_all["p_csf_0"], axis=0),
        "p_csf_1": np.concatenate(out_all["p_csf_1"], axis=0),
        "plasma_p0": np.concatenate(out_all["plasma_p0"], axis=0),
        "plasma_p1": np.concatenate(out_all["plasma_p1"], axis=0),
    }
    diagnostics = {"rhat_pi": _rhat(chain_pi), "rhat_p_pet_1": _rhat(chain_pet1), "rhat_p_csf_1": _rhat(chain_csf1)}
    return out, diagnostics


def _fit_lcm_dependence(
    *,
    pet: np.ndarray,
    csf: np.ndarray,
    plasma_cat: np.ndarray,
    priors: dict,
    chains: int,
    burn_in: int,
    draws: int,
    thin: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    n = int(len(pet))
    a_pi, b_pi = (float(priors["prevalence_beta"][0]), float(priors["prevalence_beta"][1]))
    alpha_pet_csf = np.asarray(priors["pet_csf_dirichlet"], dtype=float)
    alpha_plasma = np.asarray(priors["plasma_dirichlet"], dtype=float)

    cell = (pet.astype(int) * 2 + csf.astype(int)).astype(int)

    chain_pi: list[np.ndarray] = []
    out_all: dict[str, list[np.ndarray]] = {"pi": [], "theta0": [], "theta1": [], "plasma_p0": [], "plasma_p1": []}

    for ch in range(int(chains)):
        rng = np.random.default_rng(int(seed) + 1013 * ch)
        z = rng.integers(0, 2, size=n, dtype=np.int8)

        pi = float(rng.beta(a_pi, b_pi))
        theta0 = rng.dirichlet(alpha_pet_csf).astype(float)
        theta1 = rng.dirichlet(alpha_pet_csf).astype(float)
        plasma_p0 = rng.dirichlet(alpha_plasma).astype(float)
        plasma_p1 = rng.dirichlet(alpha_plasma).astype(float)

        total_iters = int(burn_in) + int(draws) * int(thin)
        keep_pi: list[float] = []
        keep_t0: list[np.ndarray] = []
        keep_t1: list[np.ndarray] = []
        keep_pl0: list[np.ndarray] = []
        keep_pl1: list[np.ndarray] = []
        keep_pi_chain: list[float] = []

        saved = 0
        for it in range(total_iters):
            lp1 = np.log(pi + 1e-12) + np.log(theta1[cell] + 1e-12) + np.log(plasma_p1[plasma_cat] + 1e-12)
            lp0 = np.log(1.0 - pi + 1e-12) + np.log(theta0[cell] + 1e-12) + np.log(plasma_p0[plasma_cat] + 1e-12)
            m = np.maximum(lp0, lp1)
            p1 = np.exp(lp1 - m) / (np.exp(lp0 - m) + np.exp(lp1 - m))
            z = (rng.random(n) < p1).astype(np.int8)

            n1 = int(z.sum())
            n0 = n - n1
            pi = float(rng.beta(a_pi + n1, b_pi + n0))

            cell_counts_1 = np.bincount(cell[z == 1], minlength=4).astype(float)
            cell_counts_0 = np.bincount(cell[z == 0], minlength=4).astype(float)
            theta1 = rng.dirichlet(alpha_pet_csf + cell_counts_1).astype(float)
            theta0 = rng.dirichlet(alpha_pet_csf + cell_counts_0).astype(float)

            plasma_counts_1 = np.bincount(plasma_cat[z == 1], minlength=3).astype(float)
            plasma_counts_0 = np.bincount(plasma_cat[z == 0], minlength=3).astype(float)
            plasma_p1 = rng.dirichlet(alpha_plasma + plasma_counts_1).astype(float)
            plasma_p0 = rng.dirichlet(alpha_plasma + plasma_counts_0).astype(float)

            # Identifiability constraint: A*=1 has higher PET marginal.
            pet_pos_1 = float(theta1[2] + theta1[3])
            pet_pos_0 = float(theta0[2] + theta0[3])
            if pet_pos_1 < pet_pos_0:
                pi = 1.0 - pi
                theta0, theta1 = theta1, theta0
                plasma_p0, plasma_p1 = plasma_p1, plasma_p0
                z = (1 - z).astype(np.int8)

            if it >= int(burn_in) and ((it - int(burn_in)) % int(thin) == 0):
                keep_pi.append(pi)
                keep_t0.append(theta0.copy())
                keep_t1.append(theta1.copy())
                keep_pl0.append(plasma_p0.copy())
                keep_pl1.append(plasma_p1.copy())
                keep_pi_chain.append(pi)
                saved += 1
                if saved >= int(draws):
                    break

        out_all["pi"].append(np.asarray(keep_pi, dtype=float))
        out_all["theta0"].append(np.asarray(keep_t0, dtype=float))
        out_all["theta1"].append(np.asarray(keep_t1, dtype=float))
        out_all["plasma_p0"].append(np.asarray(keep_pl0, dtype=float))
        out_all["plasma_p1"].append(np.asarray(keep_pl1, dtype=float))
        chain_pi.append(np.asarray(keep_pi_chain, dtype=float))

    out = {
        "pi": np.concatenate(out_all["pi"], axis=0),
        "theta0": np.concatenate(out_all["theta0"], axis=0),
        "theta1": np.concatenate(out_all["theta1"], axis=0),
        "plasma_p0": np.concatenate(out_all["plasma_p0"], axis=0),
        "plasma_p1": np.concatenate(out_all["plasma_p1"], axis=0),
    }
    diagnostics = {"rhat_pi": _rhat(chain_pi)}
    return out, diagnostics


def _posterior_p_a1_independence(draws: dict[str, np.ndarray], *, pet: int, csf: int, plasma_cat: int) -> np.ndarray:
    pi = draws["pi"]
    p_pet_0 = draws["p_pet_0"]
    p_pet_1 = draws["p_pet_1"]
    p_csf_0 = draws["p_csf_0"]
    p_csf_1 = draws["p_csf_1"]
    pl0 = draws["plasma_p0"]
    pl1 = draws["plasma_p1"]

    def bern(p: np.ndarray, x: int) -> np.ndarray:
        return p if x == 1 else (1.0 - p)

    lik1 = bern(p_pet_1, pet) * bern(p_csf_1, csf) * pl1[:, plasma_cat]
    lik0 = bern(p_pet_0, pet) * bern(p_csf_0, csf) * pl0[:, plasma_cat]
    num = pi * lik1
    den = num + (1.0 - pi) * lik0
    return num / np.clip(den, 1e-12, np.inf)


def _posterior_p_a1_dependence(draws: dict[str, np.ndarray], *, pet: int, csf: int, plasma_cat: int) -> np.ndarray:
    pi = draws["pi"]
    theta0 = draws["theta0"]
    theta1 = draws["theta1"]
    pl0 = draws["plasma_p0"]
    pl1 = draws["plasma_p1"]

    cell = pet * 2 + csf
    lik1 = theta1[:, cell] * pl1[:, plasma_cat]
    lik0 = theta0[:, cell] * pl0[:, plasma_cat]
    num = pi * lik1
    den = num + (1.0 - pi) * lik0
    return num / np.clip(den, 1e-12, np.inf)


def _expected_rates_independence(draws: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pi = draws["pi"]
    p_pet_0 = draws["p_pet_0"]
    p_pet_1 = draws["p_pet_1"]
    p_csf_0 = draws["p_csf_0"]
    p_csf_1 = draws["p_csf_1"]
    pl0 = draws["plasma_p0"]
    pl1 = draws["plasma_p1"]

    disc1 = p_pet_1 * (1.0 - p_csf_1) + (1.0 - p_pet_1) * p_csf_1
    disc0 = p_pet_0 * (1.0 - p_csf_0) + (1.0 - p_pet_0) * p_csf_0
    disc_pet_csf = pi * disc1 + (1.0 - pi) * disc0

    p_mid = pi * pl1[:, 1] + (1.0 - pi) * pl0[:, 1]
    p_det = pi * (pl1[:, 0] + pl1[:, 2]) + (1.0 - pi) * (pl0[:, 0] + pl0[:, 2])

    # Expected mismatch among determinates must be computed conditional on the latent class.
    pl0_low = pl0[:, 0]
    pl0_high = pl0[:, 2]
    pl1_low = pl1[:, 0]
    pl1_high = pl1[:, 2]

    num_pet = pi * (p_pet_1 * pl1_low + (1.0 - p_pet_1) * pl1_high) + (1.0 - pi) * (
        p_pet_0 * pl0_low + (1.0 - p_pet_0) * pl0_high
    )
    num_csf = pi * (p_csf_1 * pl1_low + (1.0 - p_csf_1) * pl1_high) + (1.0 - pi) * (
        p_csf_0 * pl0_low + (1.0 - p_csf_0) * pl0_high
    )

    mismatch_pet_plasma = num_pet / np.clip(p_det, 1e-12, np.inf)
    mismatch_csf_plasma = num_csf / np.clip(p_det, 1e-12, np.inf)

    return {
        "disc_pet_csf": disc_pet_csf,
        "plasma_intermediate": p_mid,
        "mismatch_pet_plasma_det": mismatch_pet_plasma,
        "mismatch_csf_plasma_det": mismatch_csf_plasma,
    }


def _summarize_draws(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    return (float(np.nanmean(x)), float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975)))


def _median_iqr(x: pd.Series) -> tuple[float, float, float]:
    v = pd.to_numeric(x, errors="coerce")
    if v.notna().sum() == 0:
        return (float("nan"), float("nan"), float("nan"))
    med = float(v.median())
    q1 = float(v.quantile(0.25))
    q3 = float(v.quantile(0.75))
    return (med, q1, q3)


def _time_strata(cfg: dict) -> list[dict]:
    strata = cfg.get("timing", {}).get("strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("definitions.yaml missing timing.strata")
    return strata


def _build_paper_b_pack_one(
    *,
    cfg: dict,
    out_dir: Path,
    triads: pd.DataFrame,
    plasma_wide: pd.DataFrame,
    clinical: pd.DataFrame,
    csf_key: str,
) -> PaperBArtifacts:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    audit_dir = ensure_dir(out_dir / "audit")
    fig_dir = ensure_dir(out_dir / "figures")
    triads = triads.copy()
    plasma_wide = plasma_wide.copy()
    clinical = clinical.copy()

    csf_cfg = cfg.get("benchmarks", {}).get(csf_key)
    if not isinstance(csf_cfg, dict):
        raise ValueError(f"definitions.yaml missing benchmarks.{csf_key}")
    csf_metric_col = str(csf_cfg["metric"])
    csf_cutpoint = float(csf_cfg["cutpoint"])
    csf_band_factors = csf_cfg.get("sensitivity_band_factors", [])
    csf_metric_label = (
        "Aβ42/Aβ40"
        if csf_metric_col == "abeta42_40_ratio"
        else ("Aβ42" if csf_metric_col == "ABETA42" else csf_metric_col)
    )

    plasma_endpoints = cfg.get("plasma", {}).get("endpoints")
    if not isinstance(plasma_endpoints, list) or not plasma_endpoints:
        raise ValueError("definitions.yaml missing plasma.endpoints")
    endpoint_names = [str(e["name"]) for e in plasma_endpoints]

    keep_plasma = ["row_uid"] + endpoint_names
    plasma_small = plasma_wide.loc[:, [c for c in keep_plasma if c in plasma_wide.columns]].copy()
    plasma_small = plasma_small.rename(columns={"row_uid": "plasma_row_uid"})
    triads = triads.merge(plasma_small, on="plasma_row_uid", how="left")
    triads = _attach_nearest_clinical(triads, clinical)

    triads["pet_pos"] = pd.to_numeric(triads.get("AMYLOID_STATUS"), errors="coerce")

    csf_metric = pd.to_numeric(triads.get(csf_metric_col), errors="coerce")
    triads["csf_a_pos"] = np.where(csf_metric.notna(), (csf_metric < csf_cutpoint).astype(int), np.nan)

    # One triad per RID (deterministic).
    triads_one = _select_one_triad_per_rid(triads, csf_metric_col=csf_metric_col)
    triads_one.to_csv(out_dir / "paperB_dataset_one_triad_per_rid_all.csv", index=False)
    triads_one_csf = triads_one[pd.to_numeric(triads_one.get(csf_metric_col), errors="coerce").notna()].copy()

    # Flow counts.
    def _flow(step: str, df: pd.DataFrame, note: str) -> dict:
        return {"step": step, "n_rows": int(len(df)), "n_rids": int(df["RID"].nunique(dropna=True)), "note": note}

    flow_rows: list[dict] = []
    flow_rows.append(_flow("triads_all", triads, "PET-anchored triads (all)"))
    csf_available = pd.to_numeric(triads.get(csf_metric_col), errors="coerce").notna()
    flow_rows.append(_flow(f"triads_all_{csf_key}_available", triads[csf_available].copy(), f"CSF {csf_metric_label} available"))
    flow_rows.append(_flow("one_triad_per_rid_all", triads_one, "Deterministic one-triad-per-RID selection"))
    flow_rows.append(_flow(f"one_triad_per_rid_{csf_key}_available", triads_one_csf, f"One per RID, CSF {csf_metric_label} available"))
    for ep in endpoint_names:
        flow_rows.append(
            _flow(
                f"paperB_primary_{ep}",
                triads_one_csf[triads_one_csf[ep].notna()].copy(),
                f"One per RID, CSF-A available, plasma endpoint present: {ep}",
            )
        )
    pd.DataFrame.from_records(flow_rows).to_csv(audit_dir / "paperB_flow_counts.csv", index=False)

    # Representativeness: included (CSF available) vs excluded.
    inc = triads_one_csf.copy()
    exc = triads_one[~triads_one["RID"].isin(inc["RID"])].copy()

    def _summ(df: pd.DataFrame) -> dict[str, object]:
        out: dict[str, object] = {"n": int(df["RID"].nunique(dropna=True))}
        age = pd.to_numeric(df.get("clin_AGE"), errors="coerce")
        out["age_mean"] = float(age.mean()) if age.notna().any() else float("nan")
        sex = df.get("clin_PTGENDER")
        if sex is not None:
            sex_s = sex.astype("string").str.strip().str.lower()
            female = (sex_s == "female").astype("float64")
            out["pct_female"] = float(female.mean())
        apoe4 = pd.to_numeric(df.get("clin_APOE4"), errors="coerce").fillna(0)
        out["pct_apoe4_pos"] = float((apoe4 >= 1).mean())
        dx = df.get("clin_dx_simplified")
        if dx is not None:
            counts = dx.astype("string").value_counts(dropna=False)
            denom = max(1, int(len(dx)))
            for k in ["CN", "MCI", "AD", "Other"]:
                out[f"pct_dx_{k}"] = float(counts.get(k, 0) / denom)
        return out

    rep = pd.DataFrame.from_records(
        [{"group": f"included_{csf_key}_available", **_summ(inc)}, {"group": f"excluded_no_{csf_key}", **_summ(exc)}]
    )
    rep.to_csv(audit_dir / "representativeness_included_vs_excluded.csv", index=False)

    # Thresholds (Paper A; used only as sensitivity for plasma discretization).
    paper_a_thresholds = pd.read_csv("outputs/paperA/plasma_thresholds_used.csv")

    timing_strata = _time_strata(cfg)
    cat_to_int = {"Low": 0, "Intermediate": 1, "High": 2}

    quantile_rows: list[dict] = []
    posterior_rows: list[dict] = []
    sens_rows: list[dict] = []
    timing_obs_rows: list[dict] = []
    timing_ppc_rows: list[dict] = []
    diag_rows: list[dict] = []
    plaus_rows: list[dict] = []

    key_patterns = cfg.get("reporting", {}).get("key_patterns")
    if not isinstance(key_patterns, list) or not key_patterns:
        raise ValueError("definitions.yaml missing reporting.key_patterns")

    # Model configs.
    m_primary = cfg["models"]["primary"]
    m_dep = cfg["models"]["sensitivity_dependence"]
    m_timing = cfg["models"]["timing_strata_fits"]["mcmc"]

    for ep_cfg in plasma_endpoints:
        ep = str(ep_cfg["name"])
        direction: Direction = str(ep_cfg["direction"])  # type: ignore[assignment]
        ep_label = str(ep_cfg.get("label", ep))

        df_ep = triads_one_csf[triads_one_csf[ep].notna()].copy()
        df_ep = df_ep.dropna(subset=["pet_pos", "csf_a_pos"]).copy()
        df_ep["pet_pos"] = pd.to_numeric(df_ep["pet_pos"], errors="coerce").astype("int64")
        df_ep["csf_a_pos"] = pd.to_numeric(df_ep["csf_a_pos"], errors="coerce").astype("int64")

        score = _plasma_score(df_ep[ep], direction)
        q_cfg = cfg["plasma"]["quantile_strata"]
        cats, cuts = _assign_quantile_strata(score, low_q=float(q_cfg["low_q"]), high_q=float(q_cfg["high_q"]))
        df_ep["plasma_cat"] = cats
        df_ep = df_ep[df_ep["plasma_cat"].isin(list(cat_to_int.keys()))].copy()
        df_ep["plasma_cat_int"] = df_ep["plasma_cat"].map(cat_to_int).astype("int64")

        quantile_rows.append({"endpoint": ep, "endpoint_label": ep_label, "direction": direction, **cuts})
        df_ep.to_csv(out_dir / f"paperB_dataset_one_triad_per_rid_{ep}.csv", index=False)

        # Face-validity summaries by pattern (aggregated only; no participant-level outputs).
        if "CENTILOIDS" in df_ep.columns and csf_metric_col in df_ep.columns:
            for (pet_v, csf_v, cat), g in df_ep.groupby(["pet_pos", "csf_a_pos", "plasma_cat"], dropna=False):
                cent_med, cent_q1, cent_q3 = _median_iqr(g["CENTILOIDS"])
                csf_med, csf_q1, csf_q3 = _median_iqr(g[csf_metric_col])
                plaus_rows.append(
                    {
                        "endpoint": ep,
                        "endpoint_label": ep_label,
                        "csf_metric": csf_metric_col,
                        "csf_metric_label": csf_metric_label,
                        "pet_pos": int(pet_v),
                        "csf_a_pos": int(csf_v),
                        "plasma_cat": str(cat),
                        "n": int(len(g)),
                        "centiloids_median": cent_med,
                        "centiloids_q1": cent_q1,
                        "centiloids_q3": cent_q3,
                        "csf_median": csf_med,
                        "csf_q1": csf_q1,
                        "csf_q3": csf_q3,
                    }
                )

        pet_arr = df_ep["pet_pos"].to_numpy(dtype=np.int8)
        csf_arr = df_ep["csf_a_pos"].to_numpy(dtype=np.int8)
        plasma_arr = df_ep["plasma_cat_int"].to_numpy(dtype=np.int8)

        # Fit models (global).
        draws_ind, diag_ind = _fit_lcm_independence(
            pet=pet_arr,
            csf=csf_arr,
            plasma_cat=plasma_arr,
            priors=m_primary["priors"],
            chains=int(m_primary["mcmc"]["chains"]),
            burn_in=int(m_primary["mcmc"]["burn_in"]),
            draws=int(m_primary["mcmc"]["draws"]),
            thin=int(m_primary["mcmc"]["thin"]),
            seed=int(m_primary["mcmc"]["seed"]),
        )
        diag_rows.append({"endpoint": ep, "model": "independence", **diag_ind, "n": int(len(df_ep))})

        draws_dep, diag_dep = _fit_lcm_dependence(
            pet=pet_arr,
            csf=csf_arr,
            plasma_cat=plasma_arr,
            priors=m_dep["priors"],
            chains=int(m_dep["mcmc"]["chains"]),
            burn_in=int(m_dep["mcmc"]["burn_in"]),
            draws=int(m_dep["mcmc"]["draws"]),
            thin=int(m_dep["mcmc"]["thin"]),
            seed=int(m_dep["mcmc"]["seed"]),
        )
        diag_rows.append({"endpoint": ep, "model": "dependence", **diag_dep, "n": int(len(df_ep))})

        # Pattern posteriors (12 patterns) under both independence and dependence models.
        for model_name, draws_ in [
            ("independence", draws_ind),
            ("dependence", draws_dep),
        ]:
            for pet_v in [0, 1]:
                for csf_v in [0, 1]:
                    for cat_label, cat_i in cat_to_int.items():
                        if model_name == "independence":
                            post = _posterior_p_a1_independence(draws_, pet=pet_v, csf=csf_v, plasma_cat=cat_i)
                        else:
                            post = _posterior_p_a1_dependence(draws_, pet=pet_v, csf=csf_v, plasma_cat=cat_i)
                        mean, lo, hi = _summarize_draws(post)
                        n_pat = int(
                            (
                                (df_ep["pet_pos"] == pet_v)
                                & (df_ep["csf_a_pos"] == csf_v)
                                & (df_ep["plasma_cat_int"] == cat_i)
                            ).sum()
                        )
                        posterior_rows.append(
                            {
                                "endpoint": ep,
                                "endpoint_label": ep_label,
                                "model": model_name,
                                "pet_pos": pet_v,
                                "csf_a_pos": csf_v,
                                "plasma_cat": cat_label,
                                "n": n_pat,
                                "p_a1_mean": mean,
                                "p_a1_lo": lo,
                                "p_a1_hi": hi,
                            }
                        )

        # Sensitivity: independence vs dependence for prespecified patterns.
        for pat in key_patterns:
            pet_v = int(pat["pet_pos"])
            csf_v = int(pat["csf_a_pos"])
            cat_label = str(pat["plasma_cat"])
            cat_i = int(cat_to_int[cat_label])
            for model_name, draws_ in [
                ("independence", draws_ind),
                ("dependence", draws_dep),
            ]:
                if model_name == "independence":
                    p_draws = _posterior_p_a1_independence(draws_, pet=pet_v, csf=csf_v, plasma_cat=cat_i)
                else:
                    p_draws = _posterior_p_a1_dependence(draws_, pet=pet_v, csf=csf_v, plasma_cat=cat_i)
                mean, lo, hi = _summarize_draws(p_draws)
                sens_rows.append(
                    {
                        "endpoint": ep,
                        "endpoint_label": ep_label,
                        "assumption_axis": "conditional_dependence",
                        "assumption_level": model_name,
                        "pattern": str(pat["name"]),
                        "pet_pos": pet_v,
                        "csf_a_pos": csf_v,
                        "plasma_cat": cat_label,
                        "p_a1_mean": mean,
                        "p_a1_lo": lo,
                        "p_a1_hi": hi,
                    }
                )

        # Sensitivity: CSF cutpoint band (refit independence model under each cutpoint).
        factors = csf_band_factors if isinstance(csf_band_factors, list) else []
        for f in factors:
            f = float(f)
            c_alt = csf_cutpoint * f
            df_alt = df_ep.copy()
            df_alt["csf_a_pos"] = (pd.to_numeric(df_alt[csf_metric_col], errors="coerce") < c_alt).astype(int)
            draws_alt, _ = _fit_lcm_independence(
                pet=df_alt["pet_pos"].to_numpy(dtype=np.int8),
                csf=df_alt["csf_a_pos"].to_numpy(dtype=np.int8),
                plasma_cat=df_alt["plasma_cat_int"].to_numpy(dtype=np.int8),
                priors=m_primary["priors"],
                chains=2,
                burn_in=1200,
                draws=2000,
                thin=1,
                seed=int(m_primary["mcmc"]["seed"]) + int(round(1000 * f)),
            )
            for pat in key_patterns:
                pet_v = int(pat["pet_pos"])
                csf_v = int(pat["csf_a_pos"])
                cat_label = str(pat["plasma_cat"])
                cat_i = int(cat_to_int[cat_label])
                p_draws = _posterior_p_a1_independence(draws_alt, pet=pet_v, csf=csf_v, plasma_cat=cat_i)
                mean, lo, hi = _summarize_draws(p_draws)
                sens_rows.append(
                    {
                        "endpoint": ep,
                        "endpoint_label": ep_label,
                        "assumption_axis": "csf_cutpoint_band",
                        "assumption_level": f"factor_{f:.2f}",
                        "pattern": str(pat["name"]),
                        "pet_pos": pet_v,
                        "csf_a_pos": csf_v,
                        "plasma_cat": cat_label,
                        "p_a1_mean": mean,
                        "p_a1_lo": lo,
                        "p_a1_hi": hi,
                    }
                )

        # Sensitivity: plasma discretization using Paper A triage thresholds (op_0.95/0.95).
        triage_cat = _triage_categories_from_paper_a_thresholds(
            df_ep,
            endpoint=ep,
            direction=direction,
            thresholds=paper_a_thresholds,
            operating_point="op_ppa0.95_npa0.95",
        )
        df_tri = df_ep.copy()
        df_tri["plasma_cat"] = triage_cat
        df_tri["plasma_cat_int"] = df_tri["plasma_cat"].map(cat_to_int).astype("int64")
        draws_tri, _ = _fit_lcm_independence(
            pet=df_tri["pet_pos"].to_numpy(dtype=np.int8),
            csf=df_tri["csf_a_pos"].to_numpy(dtype=np.int8),
            plasma_cat=df_tri["plasma_cat_int"].to_numpy(dtype=np.int8),
            priors=m_primary["priors"],
            chains=2,
            burn_in=1500,
            draws=3000,
            thin=1,
            seed=int(m_primary["mcmc"]["seed"]) + 4242,
        )
        for pat in key_patterns:
            pet_v = int(pat["pet_pos"])
            csf_v = int(pat["csf_a_pos"])
            cat_label = str(pat["plasma_cat"])
            cat_i = int(cat_to_int[cat_label])
            p_draws = _posterior_p_a1_independence(draws_tri, pet=pet_v, csf=csf_v, plasma_cat=cat_i)
            mean, lo, hi = _summarize_draws(p_draws)
            sens_rows.append(
                {
                    "endpoint": ep,
                    "endpoint_label": ep_label,
                    "assumption_axis": "plasma_discretization",
                    "assumption_level": "paperA_triage_op_ppa0.95_npa0.95",
                    "pattern": str(pat["name"]),
                    "pet_pos": pet_v,
                    "csf_a_pos": csf_v,
                    "plasma_cat": cat_label,
                    "p_a1_mean": mean,
                    "p_a1_lo": lo,
                    "p_a1_hi": hi,
                }
            )

        # Timing-stratified discordance decomposition: refit independence model within each stratum.
        for pair, delta_field in [
            ("pet_csf", "delta_pet_csf_days"),
            ("pet_plasma", "delta_pet_plasma_days"),
            ("csf_plasma", "delta_plasma_csf_days"),
        ]:
            delta = pd.to_numeric(df_ep.get(delta_field), errors="coerce")
            for s in timing_strata:
                s_name = str(s["name"])
                s_label = str(s.get("label", s_name))
                min_d = float(s["min_days"])
                max_d = float(s["max_days"])
                df_s = df_ep[(delta >= min_d) & (delta <= max_d)].copy()
                n_s = int(len(df_s))
                if n_s < 10:
                    continue

                if pair == "pet_csf":
                    k = int((df_s["pet_pos"] != df_s["csf_a_pos"]).sum())
                    lo, hi = _wilson_ci_95(k, n_s)
                    timing_obs_rows.append(
                        {
                            "endpoint": ep,
                            "endpoint_label": ep_label,
                            "pair": pair,
                            "stratum": s_name,
                            "stratum_label": s_label,
                            "n": n_s,
                            "metric": "discordance",
                            "value": float(k / n_s),
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                else:
                    mid = int((df_s["plasma_cat_int"] == 1).sum())
                    lo_m, hi_m = _wilson_ci_95(mid, n_s)
                    timing_obs_rows.append(
                        {
                            "endpoint": ep,
                            "endpoint_label": ep_label,
                            "pair": pair,
                            "stratum": s_name,
                            "stratum_label": s_label,
                            "n": n_s,
                            "metric": "plasma_intermediate",
                            "value": float(mid / n_s),
                            "lo": lo_m,
                            "hi": hi_m,
                        }
                    )
                    det = df_s[df_s["plasma_cat_int"].isin([0, 2])].copy()
                    n_det = int(len(det))
                    if n_det:
                        plasma_bin = (det["plasma_cat_int"] == 2).astype(int)
                        comp = det["pet_pos"].astype(int) if pair == "pet_plasma" else det["csf_a_pos"].astype(int)
                        mism = int((plasma_bin.to_numpy() != comp.to_numpy()).sum())
                        lo_d, hi_d = _wilson_ci_95(mism, n_det)
                        timing_obs_rows.append(
                            {
                                "endpoint": ep,
                                "endpoint_label": ep_label,
                                "pair": pair,
                                "stratum": s_name,
                                "stratum_label": s_label,
                                "n": n_det,
                                "metric": "mismatch_determinate",
                                "value": float(mism / n_det),
                                "lo": lo_d,
                                "hi": hi_d,
                            }
                        )

                draws_s, _ = _fit_lcm_independence(
                    pet=df_s["pet_pos"].to_numpy(dtype=np.int8),
                    csf=df_s["csf_a_pos"].to_numpy(dtype=np.int8),
                    plasma_cat=df_s["plasma_cat_int"].to_numpy(dtype=np.int8),
                    priors=m_primary["priors"],
                    chains=int(m_timing["chains"]),
                    burn_in=int(m_timing["burn_in"]),
                    draws=int(m_timing["draws"]),
                    thin=int(m_timing["thin"]),
                    seed=int(m_timing["seed"]) + _stable_seed(ep, pair, s_name),
                )
                exp = _expected_rates_independence(draws_s)
                if pair == "pet_csf":
                    mean, lo, hi = _summarize_draws(exp["disc_pet_csf"])
                    timing_ppc_rows.append(
                        {
                            "endpoint": ep,
                            "endpoint_label": ep_label,
                            "pair": pair,
                            "stratum": s_name,
                            "stratum_label": s_label,
                            "n": n_s,
                            "metric": "discordance",
                            "value": mean,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                else:
                    mean, lo, hi = _summarize_draws(exp["plasma_intermediate"])
                    timing_ppc_rows.append(
                        {
                            "endpoint": ep,
                            "endpoint_label": ep_label,
                            "pair": pair,
                            "stratum": s_name,
                            "stratum_label": s_label,
                            "n": n_s,
                            "metric": "plasma_intermediate",
                            "value": mean,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                    key = "mismatch_pet_plasma_det" if pair == "pet_plasma" else "mismatch_csf_plasma_det"
                    mean, lo, hi = _summarize_draws(exp[key])
                    timing_ppc_rows.append(
                        {
                            "endpoint": ep,
                            "endpoint_label": ep_label,
                            "pair": pair,
                            "stratum": s_name,
                            "stratum_label": s_label,
                            "n": n_s,
                            "metric": "mismatch_determinate",
                            "value": mean,
                            "lo": lo,
                            "hi": hi,
                        }
                    )

    # Write outputs.
    pd.DataFrame.from_records(quantile_rows).to_csv(out_dir / "plasma_quantile_cutpoints_used.csv", index=False)
    pd.DataFrame.from_records(posterior_rows).to_csv(out_dir / "pattern_posterior_table.csv", index=False)
    pd.DataFrame.from_records(sens_rows).to_csv(out_dir / "sensitivity_panel_key_posteriors.csv", index=False)
    pd.DataFrame.from_records(timing_obs_rows).to_csv(out_dir / "timing_strata_discordance_observed.csv", index=False)
    pd.DataFrame.from_records(timing_ppc_rows).to_csv(out_dir / "timing_strata_discordance_ppc.csv", index=False)
    pd.DataFrame.from_records(diag_rows).to_csv(audit_dir / "model_diagnostics.csv", index=False)

    # eTable (audit): face-validity summaries by pattern (aggregated only).
    try:
        plaus_df = pd.DataFrame.from_records(plaus_rows).copy()
        post_df = pd.DataFrame.from_records(posterior_rows).copy()
        post_df = post_df[post_df["model"] == "independence"].copy()
        if not plaus_df.empty and not post_df.empty:
            plaus_df = plaus_df.merge(
                post_df[
                    [
                        "endpoint",
                        "endpoint_label",
                        "pet_pos",
                        "csf_a_pos",
                        "plasma_cat",
                        "p_a1_mean",
                        "p_a1_lo",
                        "p_a1_hi",
                    ]
                ],
                on=["endpoint", "endpoint_label", "pet_pos", "csf_a_pos", "plasma_cat"],
                how="left",
            )
            plaus_df = plaus_df.sort_values(
                ["endpoint", "p_a1_mean", "pet_pos", "csf_a_pos", "plasma_cat"],
                ascending=[True, True, True, True, True],
                kind="mergesort",
            )
            plaus_df.to_csv(audit_dir / f"plausibility_by_pattern_{csf_key}.csv", index=False)
    except Exception as exc:
        warnings.warn(f"Failed to write Paper B plausibility table: {exc}")

    # Figures (optional; tables are the primary artifacts).
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore

    if plt is not None:
        _apply_journal_plot_style(plt)

        # eFigure: face-validity / plausibility (aggregated by pattern; no participant-level plots).
        try:
            plaus_path = audit_dir / f"plausibility_by_pattern_{csf_key}.csv"
            if plaus_path.exists():
                d = pd.read_csv(plaus_path)
                d = d.dropna(subset=["p_a1_mean"]).copy()
                if not d.empty:
                    endpoints = d["endpoint"].unique().tolist()
                    fig, axes = plt.subplots(nrows=2, ncols=len(endpoints), figsize=(13, 7), sharex=True)
                    if len(endpoints) == 1:
                        axes = np.array([[axes[0]], [axes[1]]])

                    for j, ep in enumerate(endpoints):
                        dd = d[d["endpoint"] == ep].copy()
                        if dd.empty:
                            continue
                        dd = dd.sort_values("n", kind="mergesort")
                        x = dd["p_a1_mean"].to_numpy(float)
                        xerr = np.vstack(
                            [
                                x - dd["p_a1_lo"].to_numpy(float),
                                dd["p_a1_hi"].to_numpy(float) - x,
                            ]
                        )
                        sizes = 25 + 8 * np.sqrt(dd["n"].to_numpy(float))

                        # A: Centiloids vs posterior.
                        ax = axes[0, j]
                        _add_panel_label(ax, chr(ord("a") + j))
                        y = dd["centiloids_median"].to_numpy(float)
                        yerr = np.vstack(
                            [
                                y - dd["centiloids_q1"].to_numpy(float),
                                dd["centiloids_q3"].to_numpy(float) - y,
                            ]
                        )
                        ax.errorbar(
                            x,
                            y,
                            xerr=xerr,
                            yerr=yerr,
                            fmt="none",
                            ecolor="#1f77b4",
                            alpha=0.6,
                            capsize=2,
                            lw=1.0,
                        )
                        ax.scatter(
                            x,
                            y,
                            s=sizes,
                            color="#1f77b4",
                            alpha=0.55,
                            edgecolors="white",
                            linewidths=0.4,
                            zorder=3,
                        )
                        ax.set_title(str(dd["endpoint_label"].iloc[0]))
                        if j == 0:
                            ax.set_ylabel("Median Centiloids (IQR)")
                        ax.set_xlim(-0.02, 1.02)
                        ax.grid(color="#eeeeee")

                        # B: CSF metric vs posterior.
                        ax = axes[1, j]
                        _add_panel_label(ax, chr(ord("c") + j))
                        y = dd["csf_median"].to_numpy(float)
                        yerr = np.vstack(
                            [
                                y - dd["csf_q1"].to_numpy(float),
                                dd["csf_q3"].to_numpy(float) - y,
                            ]
                        )
                        ax.errorbar(
                            x,
                            y,
                            xerr=xerr,
                            yerr=yerr,
                            fmt="none",
                            ecolor="#2ca02c",
                            alpha=0.6,
                            capsize=2,
                            lw=1.0,
                        )
                        ax.scatter(
                            x,
                            y,
                            s=sizes,
                            color="#2ca02c",
                            alpha=0.55,
                            edgecolors="white",
                            linewidths=0.4,
                            zorder=3,
                        )
                        if j == 0:
                            ax.set_ylabel(f"Median CSF {csf_metric_label} (IQR)")
                        ax.set_xlim(-0.02, 1.02)
                        ax.grid(color="#eeeeee")

                    fig.suptitle(
                        "Face validity by pattern (aggregated): higher posterior aligns with higher Centiloids and more abnormal CSF"
                    )
                    fig.supxlabel("Posterior probability of latent amyloid positivity (95% credible interval)")
                    fig.text(
                        0.01,
                        0.02,
                        "Point size proportional to pattern n",
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        color="#555555",
                    )
                    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
                    fig.savefig(fig_dir / "eFigure_plausibility.png", dpi=300, bbox_inches="tight")
                    fig.savefig(fig_dir / "eFigure_plausibility.svg", bbox_inches="tight")
                    plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Failed to render Paper B eFigure plausibility: {exc}")

        # eFigure: plasma discretization sensitivity (quantiles vs Paper A triage).
        try:
            d = pd.DataFrame.from_records(sens_rows).copy()
            d = d[d["assumption_axis"].isin(["conditional_dependence", "plasma_discretization"])].copy()
            if not d.empty:
                endpoints = [str(e["name"]) for e in plasma_endpoints]
                patterns = [str(p["name"]) for p in key_patterns]
                fig, axes = plt.subplots(nrows=1, ncols=len(endpoints), figsize=(13, 5), sharey=True)
                if len(endpoints) == 1:
                    axes = [axes]

                y = np.arange(len(patterns), dtype=float)
                for ax, ep in zip(axes, endpoints, strict=False):
                    dd = d[d["endpoint"] == ep].copy()
                    if dd.empty:
                        continue
                    base = (
                        dd[(dd["assumption_axis"] == "conditional_dependence") & (dd["assumption_level"] == "independence")]
                        .set_index("pattern")
                        .reindex(patterns)
                        .reset_index()
                    )
                    tri = (
                        dd[(dd["assumption_axis"] == "plasma_discretization") & (dd["assumption_level"] == "paperA_triage_op_ppa0.95_npa0.95")]
                        .set_index("pattern")
                        .reindex(patterns)
                        .reset_index()
                    )
                    ax.set_title(str(base["endpoint_label"].dropna().iloc[0]) if base["endpoint_label"].notna().any() else ep)
                    ax.errorbar(
                        base["p_a1_mean"].to_numpy(float),
                        y - 0.08,
                        xerr=[
                            base["p_a1_mean"].to_numpy(float) - base["p_a1_lo"].to_numpy(float),
                            base["p_a1_hi"].to_numpy(float) - base["p_a1_mean"].to_numpy(float),
                        ],
                        fmt="o",
                        color="#1f77b4",
                        label="Quantile strata" if ax is axes[0] else None,
                        capsize=3,
                    )
                    ax.errorbar(
                        tri["p_a1_mean"].to_numpy(float),
                        y + 0.08,
                        xerr=[
                            tri["p_a1_mean"].to_numpy(float) - tri["p_a1_lo"].to_numpy(float),
                            tri["p_a1_hi"].to_numpy(float) - tri["p_a1_mean"].to_numpy(float),
                        ],
                        fmt="s",
                        color="#d62728",
                        label="Paper A triage" if ax is axes[0] else None,
                        capsize=3,
                    )
                    ax.set_xlim(0, 1)
                    ax.grid(axis="x", color="#eeeeee")
                    _add_panel_label(ax, chr(ord("a") + endpoints.index(ep)))

                axes[0].set_yticks(y)
                axes[0].set_yticklabels(patterns)
                axes[0].invert_yaxis()
                handles, labels = axes[0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02))

                fig.suptitle("Sensitivity to plasma discretization (quantiles vs Paper A triage)")
                fig.supxlabel("Posterior probability of latent amyloid positivity")
                fig.tight_layout(rect=[0, 0.06, 1, 0.93])
                fig.savefig(fig_dir / "eFigure_plasma_discretization_sensitivity.svg", bbox_inches="tight")
                fig.savefig(fig_dir / "eFigure_plasma_discretization_sensitivity.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Failed to render Paper B plasma discretization eFigure: {exc}")

        # Figure 1: heatmap (independence model).
        try:
            post_df = pd.DataFrame.from_records(posterior_rows).copy()
            post_df = post_df[post_df["model"] == "independence"].copy()
            if not post_df.empty:
                from matplotlib.gridspec import GridSpec  # type: ignore

                endpoints = [str(e["name"]) for e in plasma_endpoints]
                n_endpoints = max(1, len(endpoints))
                fig = plt.figure(figsize=(12.5, 4.8))
                gs = GridSpec(1, n_endpoints + 1, figure=fig, width_ratios=[1.0] * n_endpoints + [0.05], wspace=0.25)
                axes = [fig.add_subplot(gs[0, 0])]
                for i in range(1, n_endpoints):
                    axes.append(fig.add_subplot(gs[0, i], sharey=axes[0]))
                cax = fig.add_subplot(gs[0, n_endpoints])

                row_labels = ["PET-/CSF-", "PET-/CSF+", "PET+/CSF-", "PET+/CSF+"]
                col_labels = ["Low", "Intermediate", "High"]

                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad(color="#f0f0f0")

                last_im = None
                for idx, (ax, ep) in enumerate(zip(axes, endpoints, strict=False)):
                    d = post_df[post_df["endpoint"] == ep].copy()
                    if d.empty:
                        continue
                    mat = np.full((4, 3), np.nan)
                    mat_n = np.zeros((4, 3), dtype=int)
                    for i, (pet_v, csf_v) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                        for j, cat in enumerate(col_labels):
                            r = d[(d["pet_pos"] == pet_v) & (d["csf_a_pos"] == csf_v) & (d["plasma_cat"] == cat)]
                            if r.empty:
                                continue
                            rr = r.iloc[0]
                            mat[i, j] = float(rr["p_a1_mean"])
                            mat_n[i, j] = int(rr["n"])

                    # Gray-out unobserved cells (n=0) to avoid over-interpreting model-implied probabilities.
                    mat = np.where(mat_n > 0, mat, np.nan)

                    last_im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap=cmap)
                    ax.set_aspect("auto")
                    ax.set_xticks(np.arange(3))
                    ax.set_xticklabels(col_labels)
                    ax.set_yticks(np.arange(4))
                    ax.set_yticklabels(row_labels)
                    title = str(d["endpoint_label"].iloc[0])
                    ax.set_title(title)
                    _add_panel_label(ax, chr(ord("a") + idx))
                    if idx > 0:
                        ax.tick_params(labelleft=False)
                    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
                    ax.grid(which="minor", color=(1, 1, 1, 0.65), linestyle="-", linewidth=1.2)
                    ax.tick_params(which="minor", bottom=False, left=False)
                    for i in range(4):
                        for j in range(3):
                            if mat_n[i, j] == 0:
                                ax.text(j, i, "n=0", ha="center", va="center", fontsize=8, color="#666666")
                                continue
                            txt_color = _text_color_for_cmap_value(cmap, float(mat[i, j]))
                            ax.text(
                                j,
                                i,
                                f"{_format_prob_cell(mat[i, j])}\n(n={mat_n[i, j]})",
                                ha="center",
                                va="center",
                                fontsize=10,
                                color=txt_color,
                            )

                if last_im is not None:
                    cb = fig.colorbar(last_im, cax=cax)
                    cb.set_label("Posterior probability of latent amyloid positivity")
                fig.suptitle(f"Posterior probability of latent amyloid positivity by pattern (CSF={csf_metric_label})")
                fig.text(
                    0.01,
                    0.02,
                    "Cells shaded light gray indicate unobserved patterns (n=0). Values are posterior means.",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color="#555555",
                )
                fig.tight_layout(rect=[0, 0.05, 1, 0.92])
                fig.savefig(fig_dir / "figure1_pattern_posterior_heatmap.svg", bbox_inches="tight")
                fig.savefig(fig_dir / "figure1_pattern_posterior_heatmap.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass

        # Figure S2: heatmap (dependence model; supplement).
        try:
            post_df = pd.DataFrame.from_records(posterior_rows).copy()
            post_df = post_df[post_df["model"] == "dependence"].copy()
            if not post_df.empty:
                from matplotlib.gridspec import GridSpec  # type: ignore

                endpoints = [str(e["name"]) for e in plasma_endpoints]
                n_endpoints = max(1, len(endpoints))
                fig = plt.figure(figsize=(12.5, 4.8))
                gs = GridSpec(1, n_endpoints + 1, figure=fig, width_ratios=[1.0] * n_endpoints + [0.05], wspace=0.25)
                axes = [fig.add_subplot(gs[0, 0])]
                for i in range(1, n_endpoints):
                    axes.append(fig.add_subplot(gs[0, i], sharey=axes[0]))
                cax = fig.add_subplot(gs[0, n_endpoints])

                row_labels = ["PET-/CSF-", "PET-/CSF+", "PET+/CSF-", "PET+/CSF+"]
                col_labels = ["Low", "Intermediate", "High"]

                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad(color="#f0f0f0")

                last_im = None
                for idx, (ax, ep) in enumerate(zip(axes, endpoints, strict=False)):
                    d = post_df[post_df["endpoint"] == ep].copy()
                    if d.empty:
                        continue
                    mat = np.full((4, 3), np.nan)
                    mat_n = np.zeros((4, 3), dtype=int)
                    for i, (pet_v, csf_v) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                        for j, cat in enumerate(col_labels):
                            r = d[(d["pet_pos"] == pet_v) & (d["csf_a_pos"] == csf_v) & (d["plasma_cat"] == cat)]
                            if r.empty:
                                continue
                            rr = r.iloc[0]
                            mat[i, j] = float(rr["p_a1_mean"])
                            mat_n[i, j] = int(rr["n"])

                    mat = np.where(mat_n > 0, mat, np.nan)
                    last_im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap=cmap)
                    ax.set_aspect("auto")
                    ax.set_xticks(np.arange(3))
                    ax.set_xticklabels(col_labels)
                    ax.set_yticks(np.arange(4))
                    ax.set_yticklabels(row_labels)
                    title = str(d["endpoint_label"].iloc[0])
                    ax.set_title(title)
                    _add_panel_label(ax, chr(ord("a") + idx))
                    if idx > 0:
                        ax.tick_params(labelleft=False)
                    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
                    ax.grid(which="minor", color=(1, 1, 1, 0.65), linestyle="-", linewidth=1.2)
                    ax.tick_params(which="minor", bottom=False, left=False)
                    for i in range(4):
                        for j in range(3):
                            if mat_n[i, j] == 0:
                                ax.text(j, i, "n=0", ha="center", va="center", fontsize=8, color="#666666")
                                continue
                            txt_color = _text_color_for_cmap_value(cmap, float(mat[i, j]))
                            ax.text(
                                j,
                                i,
                                f"{_format_prob_cell(mat[i, j])}\n(n={mat_n[i, j]})",
                                ha="center",
                                va="center",
                                fontsize=10,
                                color=txt_color,
                            )

                if last_im is not None:
                    cb = fig.colorbar(last_im, cax=cax)
                    cb.set_label("Posterior probability of latent amyloid positivity")
                fig.suptitle(f"Posterior probability of latent amyloid positivity by pattern (dependence model; CSF={csf_metric_label})")
                fig.text(
                    0.01,
                    0.02,
                    "Cells shaded light gray indicate unobserved patterns (n=0). Values are posterior means.",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color="#555555",
                )
                fig.tight_layout(rect=[0, 0.05, 1, 0.92])
                fig.savefig(fig_dir / "figureS2_pattern_posterior_heatmap_dependence.svg", bbox_inches="tight")
                fig.savefig(fig_dir / "figureS2_pattern_posterior_heatmap_dependence.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Failed to render Paper B Figure S2 (dependence heatmap): {exc}")

        # Figure 2: timing-stratified discordance (per endpoint).
        try:
            obs_all = pd.DataFrame.from_records(timing_obs_rows).copy()
            ppc_all = pd.DataFrame.from_records(timing_ppc_rows).copy()
            if not obs_all.empty and not ppc_all.empty:
                strata_order = [str(s["name"]) for s in timing_strata]
                x = np.arange(len(strata_order), dtype=float)
                xlabels = [str(s["label"]) for s in timing_strata]
                x_map = {name: idx for idx, name in enumerate(strata_order)}
                x_min = float(np.min(x)) - 0.25
                x_max = float(np.max(x)) + 0.25

                for ep in obs_all["endpoint"].unique().tolist():
                    obs = obs_all[obs_all["endpoint"] == ep].copy()
                    ppc = ppc_all[ppc_all["endpoint"] == ep].copy()
                    if obs.empty or ppc.empty:
                        continue

                    csf_title = "CSF amyloid"

                    # Keep a strict ≤4 panels and equal sizing (2×2).
                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11.5, 7.0), sharex=True, sharey=True)
                    (ax_pet_csf, ax_pet_plasma), (ax_csf_plasma, ax_mid) = axes

                    fig.suptitle(
                        f"Timing-stratified discordance (CSF={csf_metric_label}) — {obs['endpoint_label'].iloc[0]}"
                    )

                    def _plot_metric(
                        *,
                        ax: "plt.Axes",
                        pair_key: str,
                        metric: str,
                        pair_offset: float = 0.0,
                        markerfacecolor: str | None = None,
                        markeredgewidth: float | None = None,
                        alpha: float = 1.0,
                        add_legend: bool = False,
                    ) -> None:
                        for df_src, color, marker, lab, xoff in [
                            (obs, "#1f77b4", "o", "Observed", -0.06),
                            (ppc, "#d62728", "s", "Model", 0.06),
                        ]:
                            d = df_src[(df_src["pair"] == pair_key) & (df_src["metric"] == metric)].copy()
                            if d.empty:
                                continue
                            d["stratum"] = pd.Categorical(d["stratum"], categories=strata_order, ordered=True)
                            d = d.sort_values(["stratum"], kind="mergesort")
                            x_use = d["stratum"].map(x_map).to_numpy(dtype=float) + pair_offset + xoff
                            yv = d["value"].to_numpy(dtype=float)
                            lo = d["lo"].to_numpy(dtype=float)
                            hi = d["hi"].to_numpy(dtype=float)
                            yerr_lower = np.maximum(0.0, yv - lo)
                            yerr_upper = np.maximum(0.0, hi - yv)
                            err_kws: dict[str, object] = {
                                "x": x_use,
                                "y": yv,
                                "yerr": [yerr_lower, yerr_upper],
                                "fmt": marker,
                                "color": color,
                                "alpha": alpha,
                                "label": lab if add_legend else None,
                                "capsize": 3,
                                "markersize": 6,
                                "elinewidth": 1.2,
                            }
                            if markerfacecolor is not None:
                                err_kws["markerfacecolor"] = markerfacecolor
                                err_kws["markeredgecolor"] = color
                            if markeredgewidth is not None:
                                err_kws["markeredgewidth"] = markeredgewidth
                            ax.errorbar(**err_kws)

                    ax_pet_csf.set_title(f"PET vs {csf_title}\nDiscordance")
                    _plot_metric(ax=ax_pet_csf, pair_key="pet_csf", metric="discordance", add_legend=True)

                    ax_pet_plasma.set_title("PET vs plasma\nMismatch (determinates)")
                    _plot_metric(ax=ax_pet_plasma, pair_key="pet_plasma", metric="mismatch_determinate", alpha=0.95)

                    ax_csf_plasma.set_title(f"{csf_title} vs plasma\nMismatch (determinates)")
                    _plot_metric(ax=ax_csf_plasma, pair_key="csf_plasma", metric="mismatch_determinate", alpha=0.95)

                    ax_mid.set_title("Plasma intermediate fraction")
                    _plot_metric(ax=ax_mid, pair_key="pet_plasma", metric="plasma_intermediate", pair_offset=-0.12)
                    _plot_metric(
                        ax=ax_mid,
                        pair_key="csf_plasma",
                        metric="plasma_intermediate",
                        pair_offset=0.12,
                        markerfacecolor="white",
                        markeredgewidth=1.4,
                    )

                    for ax in [ax_pet_csf, ax_pet_plasma, ax_csf_plasma, ax_mid]:
                        ax.set_ylim(0, 1)
                        ax.set_xlim(x_min, x_max)
                        ax.set_xticks(x)
                        ax.grid(axis="y", color="#eeeeee")

                    for ax in [ax_pet_csf, ax_pet_plasma]:
                        ax.tick_params(labelbottom=False)
                    for ax in [ax_csf_plasma, ax_mid]:
                        ax.set_xticklabels(xlabels)

                    ax_pet_csf.set_ylabel("Proportion")
                    ax_csf_plasma.set_ylabel("Proportion")
                    ax_pet_plasma.tick_params(labelleft=False)
                    ax_mid.tick_params(labelleft=False)

                    _add_panel_label(ax_pet_csf, "a")
                    _add_panel_label(ax_pet_plasma, "b")
                    _add_panel_label(ax_csf_plasma, "c")
                    _add_panel_label(ax_mid, "d")

                    handles, labels = ax_pet_csf.get_legend_handles_labels()
                    if handles:
                        fig.legend(handles, labels, frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.06))
                    fig.text(
                        0.01,
                        0.02,
                        "Error bars indicate 95% uncertainty intervals (Observed: Wilson CI; Model: posterior predictive credible interval). "
                        "Mismatch is computed among determinate plasma strata (Low/High only). "
                        "Timing strata with n < 10 are not plotted. "
                        "In panel d, filled markers indicate PET–plasma and open markers indicate CSF amyloid–plasma; points are offset slightly to avoid overlap. "
                        "Panel d uses pair-specific denominators (pair-specific missingness).",
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        color="#555555",
                        wrap=True,
                    )

                    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
                    fig.savefig(fig_dir / f"figure2_timing_stratified_discordance_{ep}.svg", bbox_inches="tight")
                    fig.savefig(fig_dir / f"figure2_timing_stratified_discordance_{ep}.pdf", bbox_inches="tight")
                    fig.savefig(fig_dir / f"figure2_timing_stratified_discordance_{ep}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Failed to render Paper B Figure 2: {exc}")

        # Figure S1: sensitivity (key patterns).
        try:
            sens_df = pd.DataFrame.from_records(sens_rows).copy()
            if not sens_df.empty:
                for ep in sens_df["endpoint"].unique().tolist():
                    d = sens_df[sens_df["endpoint"] == ep].copy()
                    patterns = [str(p["name"]) for p in key_patterns]
                    patterns = [p for p in patterns if p in set(d["pattern"].unique())]
                    y = np.arange(len(patterns), dtype=float)

                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.5, 5), sharey=True)
                    fig.suptitle(f"Sensitivity analyses (key patterns; CSF={csf_metric_label}) — {d['endpoint_label'].iloc[0]}")

                    # A: independence vs dependence
                    ax = axes[0]
                    ax.set_title("Conditional dependence")
                    _add_panel_label(ax, "a")
                    dd = d[d["assumption_axis"] == "conditional_dependence"].copy()
                    for model_name, color, marker, off in [
                        ("independence", "#1f77b4", "o", -0.08),
                        ("dependence", "#d62728", "s", 0.08),
                    ]:
                        m = dd[dd["assumption_level"] == model_name].set_index("pattern").loc[patterns].reset_index()
                        ax.errorbar(
                            m["p_a1_mean"].to_numpy(float),
                            y + off,
                            xerr=[
                                m["p_a1_mean"].to_numpy(float) - m["p_a1_lo"].to_numpy(float),
                                m["p_a1_hi"].to_numpy(float) - m["p_a1_mean"].to_numpy(float),
                            ],
                            fmt=marker,
                            color=color,
                            label="Independence" if model_name == "independence" else "PET–CSF dependence",
                            capsize=3,
                        )
                    ax.legend(frameon=False)
                    ax.set_xlim(0, 1)
                    ax.grid(axis="x", color="#eeeeee")

                    # B: CSF cutpoint band (range)
                    ax = axes[1]
                    ax.set_title("CSF cutpoint band")
                    _add_panel_label(ax, "b")
                    dd = d[d["assumption_axis"] == "csf_cutpoint_band"].copy()
                    if not dd.empty:
                        ax.plot([], [], color="#555555", lw=2, label="Sensitivity range")
                        for i, pat in enumerate(patterns):
                            vals = dd[dd["pattern"] == pat]["p_a1_mean"].to_numpy(float)
                            if len(vals):
                                ax.hlines(i, float(np.min(vals)), float(np.max(vals)), color="#555555", lw=2)
                        base = dd[dd["assumption_level"] == "factor_1.00"].set_index("pattern").loc[patterns].reset_index()
                        ax.plot(base["p_a1_mean"].to_numpy(float), y, "o", color="#1f77b4", label="Base cutpoint")
                        ax.legend(frameon=False)
                    ax.set_xlim(0, 1)
                    ax.grid(axis="x", color="#eeeeee")

                    axes[0].set_yticks(y)
                    axes[0].set_yticklabels(patterns)
                    axes[0].invert_yaxis()
                    fig.supxlabel("Posterior probability of latent amyloid positivity")
                    fig.tight_layout(rect=[0, 0, 1, 0.92])
                    fig.savefig(fig_dir / f"figureS1_sensitivity_assumptions_{ep}.svg")
                    fig.savefig(fig_dir / f"figureS1_sensitivity_assumptions_{ep}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Failed to render Paper B Figure S1: {exc}")

    return PaperBArtifacts(
        dataset_one_per_rid_all=out_dir / "paperB_dataset_one_triad_per_rid_all.csv",
        pattern_posteriors=out_dir / "pattern_posterior_table.csv",
        representativeness=audit_dir / "representativeness_included_vs_excluded.csv",
        timing_observed=out_dir / "timing_strata_discordance_observed.csv",
        timing_ppc=out_dir / "timing_strata_discordance_ppc.csv",
        sensitivity_key_posteriors=out_dir / "sensitivity_panel_key_posteriors.csv",
    )


def build_paper_b_pack(*, definitions_path: Path, out_dir: Path) -> PaperBArtifacts:
    cfg = load_yaml(definitions_path)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    triads = pd.read_csv(Path(cfg["evidence_core"]["tables"]["triads"]))
    plasma_wide = pd.read_csv(Path(cfg["evidence_core"]["tables"]["plasma_wide"]))
    clinical = pd.read_csv(Path(cfg["evidence_core"]["tables"]["clinical"]), low_memory=False)

    primary = _build_paper_b_pack_one(
        cfg=cfg,
        out_dir=out_dir,
        triads=triads,
        plasma_wide=plasma_wide,
        clinical=clinical,
        csf_key="csf_a",
    )

    # Coverage/transportability analysis: CSF Aβ42 alone.
    coverage_dir: Path | None = None
    if isinstance(cfg.get("benchmarks", {}).get("csf_abeta42"), dict):
        _build_paper_b_pack_one(
            cfg=cfg,
            out_dir=out_dir / "coverage_csf_abeta42",
            triads=triads,
            plasma_wide=plasma_wide,
            clinical=clinical,
            csf_key="csf_abeta42",
        )
        coverage_dir = out_dir / "coverage_csf_abeta42"

    # eFigure: STROBE/STARD-style cohort accounting flow diagram (aggregated counts only).
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore

    if plt is not None:
        try:
            _apply_journal_plot_style(plt)
            flow_primary = pd.read_csv(out_dir / "audit" / "paperB_flow_counts.csv")
            flow_cov = pd.read_csv(coverage_dir / "audit" / "paperB_flow_counts.csv") if coverage_dir is not None else None

            def _step(df: pd.DataFrame, step: str) -> tuple[int, int]:
                r = df[df["step"] == step]
                if r.empty:
                    return (0, 0)
                rr = r.iloc[0]
                return (int(rr["n_rows"]), int(rr["n_rids"]))

            triads_all_rows, triads_all_rids = _step(flow_primary, "triads_all")
            one_rows, one_rids = _step(flow_primary, "one_triad_per_rid_all")
            pri_rows, pri_rids = _step(flow_primary, "one_triad_per_rid_csf_a_available")

            cov_rows, cov_rids = (0, 0)
            if flow_cov is not None:
                cov_rows, cov_rids = _step(flow_cov, "one_triad_per_rid_csf_abeta42_available")

            # Endpoint-level counts (one per RID + CSF available + plasma present).
            endpoints = cfg.get("plasma", {}).get("endpoints", [])
            ep_labels: list[tuple[str, str]] = []
            if isinstance(endpoints, list):
                for ep_cfg in endpoints:
                    if isinstance(ep_cfg, dict) and "name" in ep_cfg:
                        ep_labels.append((str(ep_cfg["name"]), str(ep_cfg.get("label", ep_cfg["name"]))))

            def _endpoint_counts(df: pd.DataFrame) -> dict[str, int]:
                out: dict[str, int] = {}
                for ep_name, _ in ep_labels:
                    step_name = f"paperB_primary_{ep_name}"
                    r = df[df["step"] == step_name]
                    if not r.empty:
                        out[ep_name] = int(r.iloc[0]["n_rids"])
                return out

            pri_ep = _endpoint_counts(flow_primary)
            cov_ep: dict[str, int] = _endpoint_counts(flow_cov) if flow_cov is not None else {}

            def _fmt_counts(*, triads_n: int, rids_n: int) -> str:
                return f"Triads: {triads_n}; Participants: {rids_n}"

            fig_dir = ensure_dir(out_dir / "figures")
            fig, ax = plt.subplots(figsize=(10.0, 6.5))
            ax.axis("off")

            def _box(x: float, y: float, text: str, *, fontsize: int = 10) -> None:
                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F4F6F8", "edgecolor": "#4C566A"},
                )

            def _arrow(x0: float, y0: float, x1: float, y1: float) -> None:
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#2E3440"},
                )

            top_x, top_y = 0.5, 0.84
            mid_x, mid_y = 0.5, 0.62
            left_x, left_y = 0.30, 0.42
            right_x, right_y = 0.70, 0.42

            _box(
                top_x,
                top_y,
                "PET-anchored triads\n" + _fmt_counts(triads_n=triads_all_rows, rids_n=triads_all_rids),
                fontsize=12,
            )
            _box(
                mid_x,
                mid_y,
                "Deterministic selection\nOne triad per participant\n" + _fmt_counts(triads_n=one_rows, rids_n=one_rids),
                fontsize=11,
            )
            _arrow(top_x, top_y - 0.07, mid_x, mid_y + 0.07)

            pri_lines = [
                "Primary CSF amyloid definition\nElecsys Aβ42/40 available\n"
                + _fmt_counts(triads_n=pri_rows, rids_n=pri_rids)
            ]
            for ep_name, ep_label in ep_labels:
                if ep_name in pri_ep:
                    pri_lines.append(f"{ep_label}: {pri_ep[ep_name]} participants")
            _box(left_x, left_y, "\n".join(pri_lines), fontsize=9)
            _arrow(mid_x - 0.06, mid_y - 0.07, left_x + 0.06, left_y + 0.10)

            if flow_cov is not None:
                cov_lines = [
                    "Coverage analysis\nElecsys Aβ42 available\n" + _fmt_counts(triads_n=cov_rows, rids_n=cov_rids)
                ]
                for ep_name, ep_label in ep_labels:
                    if ep_name in cov_ep:
                        cov_lines.append(f"{ep_label}: {cov_ep[ep_name]} participants")
                _box(right_x, right_y, "\n".join(cov_lines), fontsize=9)
                _arrow(mid_x + 0.06, mid_y - 0.07, right_x - 0.06, right_y + 0.10)

            fig.text(
                0.01,
                0.02,
                "Participants are unique RIDs; counts are triads / participants. Primary analysis uses CSF Aβ42/40; "
                "coverage analysis uses CSF Aβ42.",
                ha="left",
                va="bottom",
                fontsize=9,
                color="#444444",
            )

            fig.tight_layout()
            fig.savefig(fig_dir / "eFigure_flow.svg", bbox_inches="tight")
            fig.savefig(fig_dir / "eFigure_flow.png", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Failed to render Paper B flow diagram: {exc}")

    return primary
