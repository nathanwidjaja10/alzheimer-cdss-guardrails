"""Statistics for `evaluate_llm_paper.py`: Wilson confidence interval and exact McNemar test.

The functions operate on results that already exist in `results/eval_raw/*.json`; no LLM is called.

Wilson interval rather than the normal approximation: at proportions of 0 or 100% the normal
interval collapses to a single point, whereas Wilson keeps a sensible bound (0/232 gives an upper
limit of about 1.6%).

Exact McNemar test rather than an independent-proportions test: two variants or models are
evaluated on the same sessions, so the outcomes are paired. Only discordant pairs matter, and the
exact binomial form stays valid when discordant counts are very small.
"""
from __future__ import annotations

from math import comb
from statistics import NormalDist

_Z95 = NormalDist().inv_cdf(0.975)  # two-sided 95%


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval (lo, hi) for the proportion k/n on a 0..1 scale; (0.0, 1.0) when n = 0."""
    if n <= 0:
        return (0.0, 1.0)
    z = _Z95 if confidence == 0.95 else NormalDist().inv_cdf(0.5 + confidence / 2)
    phat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts b and c.

    Equivalent to twice the one-sided `scipy.stats.binomtest(min(b, c), b + c, 0.5)`, computed
    with `math.comb`.
    """
    n = b + c
    if n == 0:
        return 1.0  # no discordant pairs
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def mcnemar_table(rows_x: dict[str, bool], rows_y: dict[str, bool]) -> tuple[int, int, int]:
    """Return (b, c, n_pairs) from two {session_id: passed} dictionaries.

    b: passed on X and failed on Y; c: failed on X and passed on Y; n_pairs: sessions present in
    both. Sessions present on one side only are ignored.
    """
    common = rows_x.keys() & rows_y.keys()
    b = sum(1 for sid in common if rows_x[sid] and not rows_y[sid])
    c = sum(1 for sid in common if not rows_x[sid] and rows_y[sid])
    return b, c, len(common)
