import dataclasses

import numpy as np
import pytest

from fuzzyschema.mf_params import build_mf_params_class
from fuzzyschema.mf_params_t2 import (
    build_it2_mf_params_class, make_it2_from_t1,
    get_antecedents_t2, get_output_var_t2, _min_width,
)
from fuzzyschema.variable_config import check_trap


class TestBuildIT2MFParamsClass:
    def test_field_count_is_double_t1(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        assert len(dataclasses.fields(IT2)) == 18  # 9 T1 fields x 2 (umf/lmf)

    def test_field_names_have_umf_lmf_suffixes(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        names = {f.name for f in dataclasses.fields(IT2)}
        assert 'x1_low_umf' in names
        assert 'x1_low_lmf' in names
        assert 'y_t4_umf' in names
        assert 'y_t4_lmf' in names

    def test_containment_violation_raises(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        fields = {f.name: (0.0, 0.0, 1.0, 1.0) for f in dataclasses.fields(IT2)}
        # UMF must contain LMF: make LMF wider than UMF on one term -> violation.
        fields['x1_low_umf'] = (0.0, 0.0, 1.0, 1.0)
        fields['x1_low_lmf'] = (-1.0, 0.0, 1.0, 2.0)  # extends outside UMF
        with pytest.raises(ValueError, match="does not contain"):
            IT2(**fields)

    def test_valid_containment_passes(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        fields = {}
        for f in dataclasses.fields(IT2):
            if f.name.endswith('_umf'):
                fields[f.name] = (0.0, 1.0, 3.0, 4.0)
            else:
                fields[f.name] = (0.5, 1.5, 2.5, 3.5)
        IT2(**fields)  # should not raise


class TestMakeIT2FromT1:
    def test_produces_containing_fous(self, toy_schema):
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1()
        it2 = make_it2_from_t1(toy_schema, t1, delta=0.1, it2_cls=IT2)

        for f in dataclasses.fields(t1):
            umf = getattr(it2, f.name + '_umf')
            lmf = getattr(it2, f.name + '_lmf')
            assert umf[0] <= lmf[0]
            assert umf[1] <= lmf[1]
            assert umf[2] >= lmf[2]
            assert umf[3] >= lmf[3]

    def test_per_variable_delta_dict(self, toy_schema):
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1()
        # Large delta on x1, tiny on everything else (falls back to 0.05).
        it2 = make_it2_from_t1(toy_schema, t1, delta={'x1': 2.0}, it2_cls=IT2)

        x1_spread = it2.x1_low_umf[2] - it2.x1_low_lmf[2]
        y_spread = it2.y_t1_umf[2] - it2.y_t1_lmf[2]
        assert x1_spread > y_spread

    def test_respects_domain_bounds(self, toy_schema):
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1()
        it2 = make_it2_from_t1(toy_schema, t1, delta=5.0, it2_cls=IT2)  # large delta
        # x1 domain is (0, 10) -- UMF must not exceed it on the right.
        assert it2.x1_high_umf[3] <= 10.0
        # x1_low's default `a` (-0.1) sits below dom_min by convention (the
        # standard epsilon-below-floor trick for full membership at the
        # boundary) -- the left-edge-pinned branch must leave it unchanged,
        # not clamp it to dom_min or push it further negative.
        assert it2.x1_low_umf[0] == t1.x1_low[0]

    def test_left_edge_not_pinned_respects_domain_min(self, toy_schema):
        """
        Regression test for the eff_d domain-floor bug: expand_contract's
        general branch (a > dom_min) capped eff_d by d and (c-b)/2 but not
        by (a - dom_min), so a large delta could push umf_a below the
        variable's domain minimum even when a itself was not pinned there.
        """
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)

        # x1 domain is (0.0, 10.0). x1_low's `a` sits close to but strictly
        # above dom_min, with a flat top far enough right that (c-b)/2
        # alone would not have capped eff_d small enough under the old code.
        t1 = T1(x1_low=(0.05, 0.5, 5.0, 6.0))

        it2 = make_it2_from_t1(toy_schema, t1, delta=5.0, it2_cls=IT2)

        assert it2.x1_low_umf[0] >= 0.0  # domain min for x1


class TestIT2FromVector:
    def test_round_trip_length(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        n = len(dataclasses.fields(IT2))
        v = np.tile([1.0, 2.0, 3.0, 4.0], n)
        it2 = IT2.from_vector(v)
        assert len(dataclasses.fields(it2)) == n

    def test_wrong_length_raises(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        with pytest.raises(ValueError, match="Expected"):
            IT2.from_vector(np.zeros(3))


class TestToVectorIT2:
    def test_length_matches_from_vector_expectation(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        it2 = make_it2_from_t1(toy_schema, T1(), delta=0.1, it2_cls=IT2)
        n = len(dataclasses.fields(IT2))
        assert it2.to_vector().shape == (n * 4,)

    def test_round_trip_make_it2_from_t1_output(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        it2 = make_it2_from_t1(toy_schema, T1(), delta=0.15, it2_cls=IT2)
        it2_roundtrip = IT2.from_vector(it2.to_vector())
        for f in dataclasses.fields(IT2):
            assert getattr(it2, f.name) == pytest.approx(getattr(it2_roundtrip, f.name))

    def test_round_trip_directly_constructed_pair(self, toy_schema):
        # Direct kwargs construction (no T1 involved at all) round-trips too.
        IT2 = build_it2_mf_params_class(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        base = make_it2_from_t1(toy_schema, T1(), delta=0.1, it2_cls=IT2)
        it2 = dataclasses.replace(
            base,
            x1_low_umf=(0.0, 1.0, 5.0, 6.0),
            x1_low_lmf=(0.5, 1.5, 4.5, 5.5),
        )
        it2_roundtrip = IT2.from_vector(it2.to_vector())
        assert it2_roundtrip.x1_low_umf == pytest.approx(it2.x1_low_umf)
        assert it2_roundtrip.x1_low_lmf == pytest.approx(it2.x1_low_lmf)

    def test_round_trip_wide_umf_narrow_lmf(self, toy_schema):
        # umf=(0,1,8,10), lmf=(3,5,6,9): valid containment and internal
        # ordering, but a_l(3) > b_u(1) -- a "crossing" / wide-UMF-narrow-LMF
        # shape the old joint-sort decode could not reconstruct. The
        # UMF-anchored clamp-chain decode reaches this shape too: a_l is
        # clamped into [a_u, c_u] = [0, 8], and 3 is already inside that
        # range, so the clamp is the identity and the round trip is exact.
        IT2 = build_it2_mf_params_class(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        base = make_it2_from_t1(toy_schema, T1(), delta=0.1, it2_cls=IT2)
        it2 = dataclasses.replace(
            base,
            x1_low_umf=(0.0, 1.0, 8.0, 10.0),
            x1_low_lmf=(3.0, 5.0, 6.0, 9.0),
        )
        it2_roundtrip = IT2.from_vector(it2.to_vector())
        assert it2_roundtrip.x1_low_umf == pytest.approx(it2.x1_low_umf)
        assert it2_roundtrip.x1_low_lmf == pytest.approx(it2.x1_low_lmf)

    def test_round_trip_degenerate_lmf_equals_umf(self, toy_schema):
        # Zero-uncertainty edge case: LMF identical to UMF at every point.
        IT2 = build_it2_mf_params_class(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        base = make_it2_from_t1(toy_schema, T1(), delta=0.1, it2_cls=IT2)
        it2 = dataclasses.replace(
            base,
            x1_low_umf=(0.0, 1.0, 4.0, 6.0),
            x1_low_lmf=(0.0, 1.0, 4.0, 6.0),
        )
        it2_roundtrip = IT2.from_vector(it2.to_vector())
        assert it2_roundtrip.x1_low_umf == pytest.approx(it2.x1_low_umf)
        assert it2_roundtrip.x1_low_lmf == pytest.approx(it2.x1_low_lmf)

    def test_arbitrary_floats_always_decode_validly(self, toy_schema):
        # Repair guarantee: from_vector must still produce a fully valid
        # instance (ordering + containment) for arbitrary, unordered,
        # GA-mutation-style floats -- the clamp-chain decode must not have
        # traded away this robustness for the round-trip fix.
        IT2 = build_it2_mf_params_class(toy_schema)
        rng = np.random.default_rng(0)
        n = len(dataclasses.fields(IT2))
        for _ in range(50):
            v = rng.uniform(-5.0, 15.0, size=n * 4)
            it2 = IT2.from_vector(v)  # must not raise -- __post_init__ validates
            for f in dataclasses.fields(IT2):
                check_trap(f.name, getattr(it2, f.name))


class TestFromVectorMinWidthClamp:
    def test_degenerate_umf_genes_widened_to_at_least_epsilon(self, toy_schema):
        # All 4 raw genes for x1_low_umf land within noise of each other --
        # exactly the shape GA mutation can legally produce (see
        # GA_ROBUSTNESS_HANDOFF.md Phase 2). x1's domain is (0.0, 10.0), so
        # the domain-relative floor is 1e-3 * 10.0 = 0.01 -- but the effective
        # floor is max(that, _MIN_WIDTH_ABS), i.e. 0.06. See _min_width.
        IT2 = build_it2_mf_params_class(toy_schema)
        field_names = [f.name for f in dataclasses.fields(IT2)]
        n = len(field_names)
        v = np.tile([1.0, 2.0, 3.0, 4.0], n)  # everything else non-degenerate
        i = field_names.index('x1_low_umf')
        v[i * 4:(i + 1) * 4] = [5.0, 5.0, 5.0, 5.0]  # fully degenerate UMF genes

        it2 = IT2.from_vector(v)  # must not raise -- __post_init__ validates

        a, b, c, d = it2.x1_low_umf
        epsilon = _min_width(0.0, 10.0)
        assert d - a >= epsilon - 1e-12
        check_trap('x1_low_umf', it2.x1_low_umf)  # still a valid trapezoid
        # UMF must still contain its paired LMF after widening.
        lmf = it2.x1_low_lmf
        assert a <= lmf[0] and b <= lmf[1] and c >= lmf[2] and d >= lmf[3]

    def test_non_degenerate_umf_round_trips_unchanged(self, toy_schema):
        # Epsilon must be a no-op for normal, comfortably-wide trapezoids --
        # not just a rejection of the degenerate ones.
        IT2 = build_it2_mf_params_class(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        base = make_it2_from_t1(toy_schema, T1(), delta=0.1, it2_cls=IT2)
        it2 = dataclasses.replace(
            base,
            x1_low_umf=(0.0, 1.0, 4.0, 6.0),
            x1_low_lmf=(0.5, 1.5, 3.5, 5.5),
        )
        it2_roundtrip = IT2.from_vector(it2.to_vector())
        assert it2_roundtrip.x1_low_umf == pytest.approx(it2.x1_low_umf)
        assert it2_roundtrip.x1_low_lmf == pytest.approx(it2.x1_low_lmf)


class TestFromVectorLMFMinWidth:
    """The clamp chain in _from_vector produces a *valid* LMF but not
    necessarily a *resolvable* one: min/max against the UMF's breakpoints can
    collapse all four LMF points together. ex_fuzzy's RuleBaseT2 samples each
    consequent term on np.arange(dom0, dom1, 0.05), so a hairline LMF reads as
    all-zero there -- which is what used to hang its KM loop.
    """

    def _degenerate_lmf_vector(self, IT2, field_names):
        """Raw genes giving x1_low a wide UMF but an LMF whose 4 genes all
        collapse onto one point (the clamp chain pins them to a_u..c_u)."""
        n = len(field_names)
        v = np.tile([1.0, 2.0, 3.0, 4.0], n)
        iu = field_names.index('x1_low_umf')
        il = field_names.index('x1_low_lmf')
        v[iu * 4:(iu + 1) * 4] = [0.0, 1.0, 8.0, 9.0]   # wide, healthy UMF
        v[il * 4:(il + 1) * 4] = [5.0, 5.0, 5.0, 5.0]   # zero-width LMF genes
        return v

    def test_degenerate_lmf_widened_to_floor(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        field_names = [f.name for f in dataclasses.fields(IT2)]
        v = self._degenerate_lmf_vector(IT2, field_names)

        it2 = IT2.from_vector(v)  # must not raise

        a_l, b_l, c_l, d_l = it2.x1_low_lmf
        assert d_l - a_l >= _min_width(0.0, 10.0) - 1e-12
        check_trap('x1_low_lmf', it2.x1_low_lmf)

    def test_widened_lmf_still_contained_by_umf(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        field_names = [f.name for f in dataclasses.fields(IT2)]
        it2 = IT2.from_vector(self._degenerate_lmf_vector(IT2, field_names))

        a_u, b_u, c_u, d_u = it2.x1_low_umf
        a_l, b_l, c_l, d_l = it2.x1_low_lmf
        # Widening must never push the LMF outside the UMF that contains it --
        # __post_init__'s _check_containment would reject the instance.
        assert a_u <= a_l and b_u <= b_l and c_u >= c_l and d_u >= d_l

    def test_lmf_never_widened_past_a_narrow_umf(self, toy_schema):
        # If the UMF itself has less span than the floor (only reachable when
        # its own widening was clipped at the domain max), the LMF must settle
        # for the UMF's span rather than break containment to hit the floor.
        IT2 = build_it2_mf_params_class(toy_schema)
        field_names = [f.name for f in dataclasses.fields(IT2)]
        n = len(field_names)
        v = np.tile([1.0, 2.0, 3.0, 4.0], n)
        iu = field_names.index('x1_low_umf')
        il = field_names.index('x1_low_lmf')
        # UMF pinned hard against x1's domain max (10.0): widening can only
        # clip, so its final span is tiny.
        v[iu * 4:(iu + 1) * 4] = [10.0, 10.0, 10.0, 10.0]
        v[il * 4:(il + 1) * 4] = [10.0, 10.0, 10.0, 10.0]

        it2 = IT2.from_vector(v)  # must not raise

        a_u, _, _, d_u = it2.x1_low_umf
        a_l, _, _, d_l = it2.x1_low_lmf
        assert (d_l - a_l) <= (d_u - a_u) + 1e-12  # never wider than its UMF
        assert a_u <= a_l and d_l <= d_u           # containment holds regardless
        check_trap('x1_low_lmf', it2.x1_low_lmf)

    def test_healthy_lmf_round_trips_unchanged(self, toy_schema):
        # The floor must be a no-op on real, deliberately-chosen params --
        # otherwise from_vector would silently perturb a GA's seed chromosome
        # instead of only repairing degenerate mutants.
        IT2 = build_it2_mf_params_class(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        it2 = make_it2_from_t1(toy_schema, T1(), delta=0.1, it2_cls=IT2)

        it2_roundtrip = IT2.from_vector(it2.to_vector())

        for f in dataclasses.fields(IT2):
            assert getattr(it2_roundtrip, f.name) == pytest.approx(
                getattr(it2, f.name)
            ), f"{f.name} was perturbed by the width floor"


class TestGetAntecedentsT2:
    def test_returns_ivfs_variables(self, toy_schema):
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        it2 = make_it2_from_t1(toy_schema, T1(), delta=0.1, it2_cls=IT2)
        ants = get_antecedents_t2(toy_schema, it2)
        assert len(ants) == 2
        assert len(ants[1].linguistic_variables) == 3  # x2 has 3 terms


class TestGetOutputVarT2:
    def test_output_term_count(self, toy_schema):
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        it2 = make_it2_from_t1(toy_schema, T1(), delta=0.1, it2_cls=IT2)
        out = get_output_var_t2(toy_schema, it2)
        assert out.name == 'y'
        assert len(out.linguistic_variables) == 4
