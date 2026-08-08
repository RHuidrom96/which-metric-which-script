"""
stats_utils.py — Statistical Utilities for QRA Meta-Evaluation (QRA++ Protocol)
─────────────────────────────────────────────────────────────────────────────
Implements precision (CV*), concordance (W), pairwise agreement (P), 
dependent correlation testing (Williams), permutation testing, and bootstrap CIs.
"""

import numpy as np
import scipy.stats as stats
import scipy.special as sp
from typing import Tuple, List, Dict, Callable, Optional


# ─── Type I Measures: Small-Sample Unbiased CV* ─────────────────────────────

def c4(n: int) -> float:
    """Exact c4(n) bias correction factor for sample standard deviation."""
    if n < 2:
        return np.nan
    return float(np.sqrt(2.0 / (n - 1)) * np.exp(sp.gammaln(n / 2.0) - sp.gammaln((n - 1) / 2.0)))


def cv_star(values: np.ndarray) -> float:
    """
    Unbiased coefficient of variation with small-sample correction (CV*).
    CV* = (1 + 1/(4n)) * (s* / |m|), where s* = s / c4(n).
    Scaled by 100 for reporting (percentage).
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    
    if n < 2:
        return np.nan
    
    m = np.mean(vals)
    if np.abs(m) < 1e-12:
        return np.nan
        
    s = np.std(vals, ddof=1)
    s_star = s / c4(n)
    correction = 1.0 + (1.0 / (4.0 * n))
    
    return float(correction * (s_star / np.abs(m)) * 100.0)


def observed_range(values: np.ndarray) -> float:
    """Observed spread (max - min)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return np.nan
    return float(np.max(vals) - np.min(vals))


def effective_num_distinct_values(values: np.ndarray) -> float:
    """Effective number of distinct values using Simpson's Diversity Index."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return np.nan
    
    _, counts = np.unique(vals, return_counts=True)
    probs = counts / len(vals)
    return float(1.0 / np.sum(probs ** 2))


# ─── Type II Measures: Correlation & Concordance ──────────────────────────

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 3: return np.nan
    r, _ = stats.pearsonr(x[mask], y[mask])
    return float(r)

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 3: return np.nan
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return float(rho)

def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 3: return np.nan
    tau, _ = stats.kendalltau(x[mask], y[mask])
    return float(tau)

def kendalls_w(rankings_matrix: np.ndarray) -> float:
    mat = np.asarray(rankings_matrix, dtype=float)
    if mat.ndim != 2: return np.nan
    m, n = mat.shape
    if m < 2 or n < 2: return np.nan
        
    ranked = np.zeros_like(mat)
    for i in range(m):
        if np.isnan(mat[i]).any(): return np.nan
        ranked[i] = stats.rankdata(mat[i])
        
    R = np.sum(ranked, axis=0)
    R_mean = np.mean(R)
    S = np.sum((R - R_mean) ** 2)
    W = (12.0 * S) / ((m ** 2) * (n ** 3 - n))
    return float(np.clip(W, 0.0, 1.0))


# ─── Type IV Measures: Pairwise Agreement P & Tie Rate ─────────────────────

def pairwise_P(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Proportion of pairwise system comparisons with the same sign (P).
    Tie Rule: tie-vs-tie counts as agreement; tie-vs-difference as disagreement.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    
    if n < 2:
        return {"P": np.nan, "tie_rate": np.nan, "n_pairs": 0}
        
    n_pairs = n * (n - 1) // 2
    agreed, ties = 0, 0
    
    for i in range(n):
        for j in range(i + 1, n):
            sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if sx == 0 or sy == 0: ties += 1
            if sx == sy: agreed += 1
                
    return {
        "P": float(agreed / n_pairs),
        "tie_rate": float(ties / n_pairs),
        "n_pairs": n_pairs,
        "agreement_count": agreed
    }

def agreement_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Binary pairwise-agreement vector (length n*(n-1)/2), 1.0 where the
    sign of the pairwise difference agrees between x and y, else 0.0.
    Shared utility — previously duplicated verbatim in experiment2 and
    experiment5 as a private `_get_agreement_vector`."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    agreements = []
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            agreements.append(1.0 if sx == sy else 0.0)
    return np.array(agreements)


def leave_one_out_centrality(scores_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    instruments = list(scores_dict.keys())
    result = {}
    
    for target in instruments:
        taus, ps, ties = [], [], []
        for other in instruments:
            if target == other: continue
            x, y = scores_dict[target], scores_dict[other]
            t = kendall_tau_b(x, y)
            p_info = pairwise_P(x, y)
            
            if not np.isnan(t): taus.append(t)
            if not np.isnan(p_info["P"]):
                ps.append(p_info["P"])
                ties.append(p_info["tie_rate"])
                
        result[target] = {
            "mean_tau": float(np.mean(taus)) if taus else np.nan,
            "mean_P": float(np.mean(ps)) if ps else np.nan,
            "mean_tie_rate": float(np.mean(ties)) if ties else np.nan,
        }
    return result


# ─── Statistical Inference: Bootstrap CIs, Hypothesis Tests & Permutations ──────────────

def bootstrap_ci(values: np.ndarray, func: Callable = np.mean, n_resamples: int = 1000, ci: float = 0.95, seed: int = 16) -> Tuple[float, float, float]:
    """BCa (Bias-Corrected Accelerated) Bootstrap Confidence Interval."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    
    if n < 3:
        return float(func(vals)) if n > 0 else np.nan, np.nan, np.nan
        
    point = float(func(vals))
    rng = np.random.RandomState(seed)
    
    boot_stats = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_stats[b] = func(vals[idx])
        
    prop_less = np.clip(np.mean(boot_stats < point), 1e-6, 1.0 - 1e-6)
    z0 = stats.norm.ppf(prop_less)
    
    jk_stats = np.empty(n)
    for i in range(n):
        jk_stats[i] = func(np.delete(vals, i))
    
    jk_mean = np.mean(jk_stats)
    denom = 6.0 * (np.sum((jk_mean - jk_stats) ** 2) ** 1.5)
    accel = np.sum((jk_mean - jk_stats) ** 3) / (denom + 1e-12) if denom != 0 else 0.0
    
    alpha = 1.0 - ci
    z_alpha_low, z_alpha_high = stats.norm.ppf(alpha / 2.0), stats.norm.ppf(1.0 - alpha / 2.0)
    
    p_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1.0 - accel * (z0 + z_alpha_low) + 1e-12))
    p_high = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1.0 - accel * (z0 + z_alpha_high) + 1e-12))
    
    ci_low = float(np.percentile(boot_stats, np.clip(p_low * 100.0, 0.0, 100.0)))
    ci_high = float(np.percentile(boot_stats, np.clip(p_high * 100.0, 0.0, 100.0)))
    
    return point, ci_low, ci_high

"""def williams_test(r12: float, r13: float, r23: float, n: int) -> Dict[str, float]:
    # Williams (1959) test for dependent correlations.
    if n <= 3 or np.isnan([r12, r13, r23]).any(): return {"t": np.nan, "df": max(1, n - 3), "p": np.nan}
        
    num = (r12 - r13) * np.sqrt((n - 3) * (1.0 + r23))
    det = 1.0 - r12**2 - r13**2 - r23**2 + 2.0 * r12 * r13 * r23
    
    if det <= 0: return {"t": np.nan, "df": n - 3, "p": np.nan}
        
    t_stat = num / (2.0 * np.sqrt((n - 1.0) / (n - 3.0)) * np.sqrt(det) + 1e-12)
    df = n - 3
    return {"t": float(t_stat), "df": int(df), "p": float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df)))}"""
    
def williams_test(r12, r13, r23, n):
    if n <= 3 or np.isnan([r12, r13, r23]).any():
        return {"t": np.nan, "df": max(1, n - 3), "p": np.nan}
    det = 1.0 - r12**2 - r13**2 - r23**2 + 2.0 * r12 * r13 * r23
    if det <= 0:
        return {"t": np.nan, "df": n - 3, "p": np.nan}
    r_bar = (r12 + r13) / 2.0
    num = (r12 - r13) * np.sqrt((n - 1.0) * (1.0 + r23))
    denom = np.sqrt(2.0 * ((n - 1.0) / (n - 3.0)) * det + (r_bar ** 2) * (1.0 - r23) ** 3)
    t_stat = num / (denom + 1e-12)
    df = n - 3
    return {"t": float(t_stat), "df": int(df),
            "p": float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df)))}


def paired_permutation_test(x: np.ndarray, y: np.ndarray, n_permutations: int = 10000, seed: int = 16) -> Dict[str, float]:
    """
    Paired permutation test for difference in means.
    Tests H0: mean(x) == mean(y) under random sign flips of differences.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 2: return {"observed_diff": np.nan, "p_value": np.nan, "n_permutations": n_permutations}
    
    observed_diff = np.mean(x) - np.mean(y)
    diffs = x - y
    
    rng = np.random.RandomState(seed)
    perm_means = np.empty(n_permutations)
    
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_means[i] = np.mean(diffs * signs)
        
    p_value = np.mean(np.abs(perm_means) >= np.abs(observed_diff))
    
    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations
    }

def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[Dict[str, float]]:
    p_vals = np.asarray(p_values, dtype=float)
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    
    adjusted = np.zeros(n)
    cum_max = 0.0
    for i, idx in enumerate(sorted_idx):
        adj = (n - i) * p_vals[idx]
        cum_max = max(cum_max, adj)
        adjusted[idx] = min(1.0, cum_max)
        
    return [{"p_adj": float(adj), "reject": bool(adj < alpha)} for adj in adjusted]

CORR_FUNCS = {
    "pearson_r": pearson_r,
    "spearman_rho": spearman_rho,
    "kendall_tau_b": kendall_tau_b,
}
