"""
fuzzyschema/engine.py
---------------------
Application-agnostic FLS engine implementations: T1 (Type-1) and IT2
(Interval Type-2). Takes antecedents/rules/output_var as plain arguments;
has no knowledge of any concrete variable schema.

KM type reduction
-----------------
ex_fuzzy's RuleBaseT2.inference() has an IndexError in consequent_centroid_r
when no element satisfies the argwhere condition on certain inputs. T2FLSEngine
bypasses this by calling compute_rule_antecedent_memberships() (which works
correctly) and then applying _km_endpoint() for type reduction directly.
"""

from abc import ABC, abstractmethod

import numpy as np
from ex_fuzzy.rules import RuleBaseT1, RuleBaseT2


# ── Input validation ─────────────────────────────────────────────────────────

def validate_input(X: np.ndarray, n_expected: int, var_names: list) -> None:
    """Raise ValueError if X is not a valid (n_samples, n_expected) input matrix."""
    if X.ndim != 2:
        raise ValueError(f"run_inference: X must be 2-dimensional, got shape {X.shape}")
    if X.shape[1] != n_expected:
        raise ValueError(
            f"run_inference: X must have {n_expected} columns "
            f"({', '.join(var_names)}), got {X.shape[1]}"
        )


# ── Karnik-Mendel type reduction ─────────────────────────────────────────────

def _km_endpoint(
    f_primary: np.ndarray,
    f_secondary: np.ndarray,
    c: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Compute one endpoint of the KM type-reduced set.

    For the left endpoint y_l: f_primary=f_u, f_secondary=f_l.
    For the right endpoint y_r: f_primary=f_l, f_secondary=f_u.
    Uses searchsorted to locate the switch point, avoiding the IndexError
    in ex_fuzzy's centroid.py when argwhere returns an empty array.
    """
    idx = np.argsort(c)
    c_s, fp_s, fs_s = c[idx], f_primary[idx], f_secondary[idx]
    n = len(c_s)

    denom = fp_s.sum()
    if denom == 0.0:
        # No primary firing at all: fall back to the secondary-weighted
        # centroid rather than NaN (handles the all-LMF-zero case).
        denom = fs_s.sum()
        if denom == 0.0:
            return np.nan
        return float(np.dot(fs_s, c_s) / denom)
    y = float(np.dot(fp_s, c_s) / denom)

    for _ in range(max_iter):
        switch = int(np.searchsorted(c_s, y, side='right')) - 1
        switch = max(0, min(switch, n - 1))
        f_mix = np.where(np.arange(n) <= switch, fp_s, fs_s)
        denom = f_mix.sum()
        if denom == 0.0:
            return np.nan
        y_new = float(np.dot(f_mix, c_s) / denom)
        if abs(y_new - y) < tol:
            return y_new
        y = y_new

    return y


# ── Abstract interface ───────────────────────────────────────────────────────

class FLSEngine(ABC):
    """Stable interface for all FLS engine variants. Depend on this, not on
    concrete engine classes directly."""

    @abstractmethod
    def run_inference(self, X: np.ndarray) -> np.ndarray:
        """Run FLS inference on a batch of input samples.

        Returns shape (n_samples,) calibrated outputs; NaN where no rules fired.
        """


# ── T1 engine ─────────────────────────────────────────────────────────────────

class T1FLSEngine(FLSEngine):
    """Type-1 FLS engine. Rule base is built once at construction."""

    def __init__(self, antecedents: list, rules: list, output_var) -> None:
        self._antecedents = antecedents
        self._n_inputs = len(antecedents)
        self._var_names = [v.name for v in antecedents]
        self._rb = RuleBaseT1(antecedents=antecedents, rules=rules, consequent=output_var)

    def run_inference(self, X: np.ndarray) -> np.ndarray:
        validate_input(X, self._n_inputs, self._var_names)
        return self._rb.inference(X)


# ── IT2 engine ────────────────────────────────────────────────────────────────

class T2FLSEngine(FLSEngine):
    """Interval Type-2 FLS engine with custom Karnik-Mendel type reduction.

    Bypasses ex_fuzzy's RuleBaseT2.inference() (broken on batch input for
    certain firing patterns). Instead: compute rule antecedent memberships
    for the full batch, apply _km_endpoint() per sample, defuzzify as the
    midpoint of the type-reduced interval.
    """

    def __init__(
        self,
        antecedents: list,
        rules: list,
        output_var,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        self._antecedents = antecedents
        self._n_inputs = len(antecedents)
        self._var_names = [v.name for v in antecedents]
        self._rb = RuleBaseT2(antecedents=antecedents, rules=rules, consequent=output_var)
        self._max_iter = max_iter
        self._tol = tol

    def run_inference(self, X: np.ndarray) -> np.ndarray:
        validate_input(X, self._n_inputs, self._var_names)
        n_samples = X.shape[0]

        firing = self._rb.compute_rule_antecedent_memberships(X)

        rule_consequents = np.array([r.consequent for r in self._rb.rules])
        centroids = self._rb.consequent_centroids
        c0 = centroids[rule_consequents, 0]
        c1 = centroids[rule_consequents, 1]
        c_l = np.minimum(c0, c1)
        c_r = np.maximum(c0, c1)

        results = np.full(n_samples, np.nan)
        for s in range(n_samples):
            f_l = firing[s, :, 0]
            f_u = firing[s, :, 1]
            y_l = _km_endpoint(f_u, f_l, c_l, self._max_iter, self._tol)
            y_r = _km_endpoint(f_l, f_u, c_r, self._max_iter, self._tol)
            if not (np.isnan(y_l) or np.isnan(y_r)):
                results[s] = (y_l + y_r) / 2.0

        return results
