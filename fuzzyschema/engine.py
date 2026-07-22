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

The same broken KM is *also* reached at construction time, which inference-level
bypassing cannot reach -- see _patch_ex_fuzzy_centroids() below.
"""

from abc import ABC, abstractmethod

import numpy as np
import ex_fuzzy.centroid
from ex_fuzzy.rules import RuleBaseT1, RuleBaseT2

from fuzzyschema.rules import DONT_CARE


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


def _reject_dont_care_antecedents(rules: list) -> None:
    """Raise ValueError if any rule carries a DONT_CARE (-1) antecedent.

    validate_rules() deliberately PERMITS DONT_CARE: it is a legitimate sparse-
    authoring / GA-seeding convenience, and RuleChromosomeCodec.expert_chromosome()
    is the one explicit place that resolves it (via specificity). But the engines
    wrap ex_fuzzy's RuleBaseT1/RuleBaseT2, which have no wildcard-resolution
    logic -- a DONT_CARE antecedent reaching them silently co-fires every rule it
    subsumes instead of most-specific-rule-wins. Engine construction therefore
    requires a dense rule base.

    Reject, don't auto-expand: expansion has exactly one home already
    (expert_chromosome), and doing it here would let a caller's mistake through
    silently instead of failing loud.
    """
    for i, rule in enumerate(rules):
        ants = [int(a) for a in rule.antecedents]
        if DONT_CARE in ants:
            raise ValueError(
                f"Rule {i}: antecedents {ants} contain DONT_CARE ({DONT_CARE}); "
                f"engine construction requires a dense rule base (no wildcard "
                f"antecedents). Resolve DONT_CARE before building an engine "
                f"(e.g. via RuleChromosomeCodec.expert_chromosome())."
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


# ── ex_fuzzy consequent-centroid patch ───────────────────────────────────────

def _centroid_t2_l(z: np.ndarray, memberships: np.ndarray) -> float:
    """Left consequent-centroid endpoint. memberships is (N, 2) = (LMF, UMF),
    matching IVFS's (LMF, UMF) construction order."""
    m = np.asarray(memberships, dtype=float)
    return _km_endpoint(m[:, 1], m[:, 0], np.asarray(z, dtype=float))


def _centroid_t2_r(z: np.ndarray, memberships: np.ndarray) -> float:
    """Right consequent-centroid endpoint. Primary/secondary are swapped
    relative to _centroid_t2_l, per _km_endpoint's documented convention."""
    m = np.asarray(memberships, dtype=float)
    return _km_endpoint(m[:, 0], m[:, 1], np.asarray(z, dtype=float))


def _patch_ex_fuzzy_centroids() -> None:
    """Rebind ex_fuzzy.centroid's IT2 centroid endpoints onto _km_endpoint.

    RuleBaseT2.__init__ computes self.consequent_centroids as a *side effect of
    construction* (ex_fuzzy/rules.py:841), so T2FLSEngine cannot bypass it the
    way run_inference() bypasses RuleBaseT2.inference() -- by the time the
    engine holds a rule base, the damage is done. The patch therefore has to be
    in place before any RuleBaseT2 is constructed, which importing this module
    guarantees.

    ex_fuzzy's own compute_centroid_t2_l/_r are broken two ways:

      1. They converge on exact float equality (`while yhat != yhat_2`) with no
         guard against sum(w) == 0. A consequent term whose LMF samples to
         all-zero on the 0.05 grid makes center_of_masses return 0/0 = NaN.
         The loop then does NOT freeze on NaN -- it oscillates in a 2-cycle,
         because the empty-argwhere IndexError fallback (k = 0) rescues yhat to
         a finite value every other iteration and routes straight back into the
         degenerate branch. A bare iteration cap is thus not a fix: it returns
         whichever half of the cycle the cap lands on.
      2. The switch point (`argwhere(...)[-1]`) selects the *last* index
         satisfying the condition. Since the domain grid is ascending, that is
         essentially always N-1, collapsing "KM" into a plain LMF-weighted
         centroid -- which yields inverted intervals (left endpoint > right)
         even on perfectly healthy input.

    _km_endpoint has neither problem: searchsorted for the switch point, a
    tolerance-bounded loop, and an explicit zero-denominator fallback.

    Patching the two leaf functions rather than compute_centroid_iv is
    deliberate: compute_centroid_iv looks them up as module globals at call
    time, so rebinding the leaves fixes every caller (both RuleBase centroid
    sites, rules.py:374 and :841) while keeping compute_centroid_t2_l/_r
    directly callable -- and therefore directly testable.

    Idempotent: safe to call more than once.
    """
    if getattr(ex_fuzzy.centroid, '_fuzzyschema_patched', False):
        return
    ex_fuzzy.centroid.compute_centroid_t2_l = _centroid_t2_l
    ex_fuzzy.centroid.compute_centroid_t2_r = _centroid_t2_r
    ex_fuzzy.centroid._fuzzyschema_patched = True


_patch_ex_fuzzy_centroids()


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
        _reject_dont_care_antecedents(rules)
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
        _reject_dont_care_antecedents(rules)
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
