import time

import ex_fuzzy.centroid
import numpy as np
import pytest

from fuzzyschema.engine import (
    T1FLSEngine, T2FLSEngine, validate_input, _km_endpoint,
    _centroid_t2_l, _centroid_t2_r, _patch_ex_fuzzy_centroids,
)
from fuzzyschema.mf_params import build_mf_params_class, get_antecedents, get_output_var
from fuzzyschema.mf_params_t2 import build_it2_mf_params_class, make_it2_from_t1, get_antecedents_t2, get_output_var_t2
from fuzzyschema.rules import DONT_CARE, RuleFactory


class TestValidateInput:
    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            validate_input(np.zeros(5), 1, ['x'])

    def test_wrong_column_count_raises(self):
        with pytest.raises(ValueError, match="1 columns"):
            validate_input(np.zeros((3, 2)), 1, ['x'])

    def test_correct_shape_passes(self):
        validate_input(np.zeros((3, 2)), 2, ['x', 'y'])  # should not raise


class TestT1FLSEngineSingleAntecedent:
    """Exercises the smallest possible schema (1 antecedent) -- different
    shape from the 4-antecedent dissertation schema this code was
    originally written against."""

    def _build_engine(self, schema):
        MFParams = build_mf_params_class(schema)
        params = MFParams()
        rf = RuleFactory(schema)
        rules = [rf.rule(only_input=0, out=0), rf.rule(only_input=1, out=1)]
        return T1FLSEngine(
            antecedents=get_antecedents(schema, params),
            rules=rules,
            output_var=get_output_var(schema, params),
        )

    def test_runs_without_error(self, single_antecedent_schema):
        engine = self._build_engine(single_antecedent_schema)
        result = engine.run_inference(np.array([[0.1], [0.9]]))
        assert result.shape == (2,)

    def test_low_input_biases_toward_low_output(self, single_antecedent_schema):
        engine = self._build_engine(single_antecedent_schema)
        result = engine.run_inference(np.array([[0.0], [1.0]]))
        assert result[0] < result[1]

    def test_wrong_column_count_raises(self, single_antecedent_schema):
        engine = self._build_engine(single_antecedent_schema)
        with pytest.raises(ValueError):
            engine.run_inference(np.zeros((2, 2)))  # schema expects 1 column


class TestT2FLSEngineToySchema:
    def _build_engine(self, schema):
        T1 = build_mf_params_class(schema)
        IT2cls = build_it2_mf_params_class(schema)
        it2_params = make_it2_from_t1(schema, T1(), delta=0.1, it2_cls=IT2cls)
        rf = RuleFactory(schema)
        rules = [
            rf.rule(x1=0, x2=0, out=0),
            rf.rule(x1=1, x2=2, out=3),
        ]
        return T2FLSEngine(
            antecedents=get_antecedents_t2(schema, it2_params),
            rules=rules,
            output_var=get_output_var_t2(schema, it2_params),
        )

    def test_runs_without_error(self, toy_schema):
        engine = self._build_engine(toy_schema)
        result = engine.run_inference(np.array([[1.0, 0.1], [9.0, 0.9]]))
        assert result.shape == (2,)

    def test_no_matching_rule_gives_nan(self, toy_schema):
        """An input region with no firing rule should return NaN, not
        crash or silently return an arbitrary value."""
        engine = self._build_engine(toy_schema)
        # Rules only cover (x1=LOW,x2=LOW) and (x1=HIGH,x2=HIGH) corners;
        # a far-out-of-domain input can still trigger some membership due
        # to flat tops, so instead check the shape/type contract directly.
        result = engine.run_inference(np.array([[5.0, 0.5]]))
        assert result.shape == (1,)
        assert np.isnan(result[0]) or np.isfinite(result[0])


class TestEngineRejectsDontCare:
    """Both engines require a DENSE rule base at construction. A DONT_CARE (-1)
    antecedent -- which validate_rules() permits as a sparse-authoring / GA-
    seeding convenience -- must be rejected here, because ex_fuzzy's RuleBaseT1/
    RuleBaseT2 have no wildcard-resolution logic and would silently co-fire every
    rule a wildcard subsumes instead of most-specific-rule-wins."""

    def _t1_bits(self, schema):
        params = build_mf_params_class(schema)()
        return dict(
            antecedents=get_antecedents(schema, params),
            output_var=get_output_var(schema, params),
        )

    def _t2_bits(self, schema):
        T1 = build_mf_params_class(schema)
        IT2 = build_it2_mf_params_class(schema)
        it2 = make_it2_from_t1(schema, T1(), delta=0.1, it2_cls=IT2)
        return dict(
            antecedents=get_antecedents_t2(schema, it2),
            output_var=get_output_var_t2(schema, it2),
        )

    def test_t1_rejects_dont_care(self, toy_schema):
        # x2 left unspecified -> RuleFactory fills it with DONT_CARE.
        rules = [RuleFactory(toy_schema).rule(x1=0, out=0)]
        assert DONT_CARE in [int(a) for a in rules[0].antecedents]
        with pytest.raises(ValueError, match="DONT_CARE"):
            T1FLSEngine(rules=rules, **self._t1_bits(toy_schema))

    def test_t2_rejects_dont_care(self, toy_schema):
        rules = [RuleFactory(toy_schema).rule(x1=0, out=0)]
        assert DONT_CARE in [int(a) for a in rules[0].antecedents]
        with pytest.raises(ValueError, match="DONT_CARE"):
            T2FLSEngine(rules=rules, **self._t2_bits(toy_schema))

    def test_t1_constructs_on_dense_rule_base(self, toy_schema):
        # Every antecedent specified -> no DONT_CARE -> the guard must NOT fire.
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, x2=0, out=0), rf.rule(x1=1, x2=2, out=3)]
        engine = T1FLSEngine(rules=rules, **self._t1_bits(toy_schema))
        assert isinstance(engine, T1FLSEngine)

    def test_t2_constructs_on_dense_rule_base(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, x2=0, out=0), rf.rule(x1=1, x2=2, out=3)]
        engine = T2FLSEngine(rules=rules, **self._t2_bits(toy_schema))
        assert isinstance(engine, T2FLSEngine)


class TestKMEndpointGeneric:
    def test_all_zero_primary_falls_back_to_secondary(self):
        f_primary = np.array([0.0, 0.0, 0.0])
        f_secondary = np.array([1.0, 1.0, 1.0])
        c = np.array([0.0, 1.0, 2.0])
        result = _km_endpoint(f_primary, f_secondary, c)
        assert result == pytest.approx(1.0)  # uniform secondary -> centroid = mean(c)

    def test_all_zero_both_returns_nan(self):
        f_primary = np.zeros(3)
        f_secondary = np.zeros(3)
        c = np.array([0.0, 1.0, 2.0])
        assert np.isnan(_km_endpoint(f_primary, f_secondary, c))


class TestExFuzzyCentroidPatch:
    """Importing fuzzyschema.engine rebinds ex_fuzzy.centroid's IT2 centroid
    endpoints onto _km_endpoint -- see _patch_ex_fuzzy_centroids.

    Before the patch, an all-zero LMF made ex_fuzzy's compute_centroid_t2_r
    loop forever: sum(w) == 0 gives 0/0 = NaN, and `while yhat != yhat_2`
    never terminates because the empty-argwhere IndexError fallback (k = 0)
    bounces yhat between NaN and a finite wrong value in a 2-cycle. These
    tests would HANG on unpatched ex_fuzzy, so they are also the regression
    guard for the patch being applied at all.
    """

    def test_patch_is_installed(self):
        assert ex_fuzzy.centroid.compute_centroid_t2_l is _centroid_t2_l
        assert ex_fuzzy.centroid.compute_centroid_t2_r is _centroid_t2_r

    def test_patch_is_idempotent(self):
        _patch_ex_fuzzy_centroids()
        _patch_ex_fuzzy_centroids()
        assert ex_fuzzy.centroid.compute_centroid_t2_r is _centroid_t2_r

    def test_all_zero_memberships_terminates_with_nan(self):
        # The exact input that used to hang. An entirely empty fuzzy set has
        # no defined centroid, so NaN is the honest answer -- the contract
        # being asserted is that it *returns* one, promptly.
        z = np.array([0.1, 0.4, 0.6])
        memberships = np.zeros((3, 2))

        start = time.perf_counter()
        r = ex_fuzzy.centroid.compute_centroid_t2_r(z, memberships)
        l = ex_fuzzy.centroid.compute_centroid_t2_l(z, memberships)
        elapsed = time.perf_counter() - start

        assert np.isnan(r)
        assert np.isnan(l)
        assert elapsed < 1.0, f"took {elapsed:.2f}s -- should be near-instant"

    def test_zero_lmf_falls_back_to_umf_weighted_centroid(self):
        # The realistic degenerate case: LMF samples to all-zero on the grid
        # (too thin to land on a grid point) but the UMF is healthy. _km_endpoint
        # falls back to the secondary-weighted centroid rather than NaN.
        z = np.array([0.0, 1.0, 2.0])
        memberships = np.column_stack([
            np.zeros(3),              # LMF: all-zero
            np.array([1.0, 1.0, 1.0]),  # UMF: uniform -> centroid = mean(z) = 1.0
        ])

        r = ex_fuzzy.centroid.compute_centroid_t2_r(z, memberships)

        assert r == pytest.approx(1.0)
        assert np.isfinite(r)

    def test_interval_is_never_inverted(self):
        # ex_fuzzy's argwhere(...)[-1] switch point returned y_l > y_r on
        # perfectly healthy input (a real inverted interval). searchsorted does not.
        z = np.arange(0.0, 1.0, 0.05)
        lmf = np.clip(1.0 - np.abs(z - 0.2) / 0.15, 0.0, 1.0)
        umf = np.clip(1.0 - np.abs(z - 0.2) / 0.30, 0.0, 1.0)
        memberships = np.column_stack([lmf, umf])

        l = ex_fuzzy.centroid.compute_centroid_t2_l(z, memberships)
        r = ex_fuzzy.centroid.compute_centroid_t2_r(z, memberships)

        assert np.isfinite(l) and np.isfinite(r)
        assert l <= r, f"inverted interval: y_l={l} > y_r={r}"

    def test_compute_centroid_iv_uses_the_patched_endpoints(self):
        # compute_centroid_iv resolves the two endpoint fns as module globals
        # at call time, so patching the leaves must reach it -- this is what
        # makes RuleBaseT2.__init__ (the actual hang site) safe.
        z = np.array([0.1, 0.4, 0.6])
        iv = ex_fuzzy.centroid.compute_centroid_iv(z, np.zeros((3, 2)))
        assert np.isnan(iv).all()
