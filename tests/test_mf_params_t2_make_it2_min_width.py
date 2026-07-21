"""
tests/test_mf_params_t2_make_it2_min_width.py
---------------------------------------------
Does make_it2_from_t1 apply the same minimum-trapezoid-width floor that
from_vector does?

from_vector has applied a width floor for a while (see
TestFromVectorMinWidthClamp / TestFromVectorLMFMinWidth in
test_mf_params_t2.py): a trapezoid narrower than ex_fuzzy's hardcoded 0.05
consequent-sampling step (RuleBaseT2 samples every consequent term on
np.arange(dom0, dom1, 0.05)) can fall entirely between two grid points, sample
to all-zero, and hang KM. make_it2_from_t1 -- the *other* constructor for the
same class -- never got that floor. Its only guard,
eff_d = min(d, (c - b) / 2.0, a - dom_min), prevents inversion and crossing,
not degenerate width.

That was harmless while `delta` was a fixed, hand-picked constant. It stops
being harmless the moment `delta` becomes a GA gene, because both branches of
expand_contract can then be driven to a grid-invisible trapezoid:

  * a narrow *core* (small c - b) caps eff_d, and the LMF contracts towards a
    point -- a rectangular core (b == a, c == d) collapses it onto one exactly;
  * a trapezoid already narrowed by a `scale` gene stays narrow in the UMF,
    because expansion only ever adds eff_d to each side, and eff_d is itself
    capped by that same narrow core.

Every test here asserts against the library's own _min_width/_widen_* helpers
rather than a copied 0.06, so a change to _MIN_WIDTH_ABS cannot silently
desync the test from the code it guards.
"""

import dataclasses

import numpy as np
import pytest

from fuzzyschema.mf_params import build_mf_params_class
from fuzzyschema.mf_params_t2 import (
    build_it2_mf_params_class, make_it2_from_t1,
    _min_width, _widen_if_degenerate, _widen_lmf_if_degenerate,
)
from fuzzyschema.variable_config import Schema, TermSpec, VariableSpec, check_trap


# ex_fuzzy's RuleBaseT2 consequent-sampling step (rules.py:836). Not imported --
# it is a hardcoded constant *inside* ex_fuzzy, so the only honest thing to do
# is restate it here as the thing the floor exists to clear.
EX_FUZZY_GRID_STEP = 0.05


# ── Invariant helpers ────────────────────────────────────────────────────────

def _assert_valid_pair(umf, lmf, dom_min, dom_max):
    """Every invariant __post_init__ enforces, plus the width floor itself.

    Neither half's floor is a flat `epsilon`, because neither is free to widen
    without limit. Both helpers only push points outward into whatever room they
    have, and each has a different ceiling on that room:

      UMF: >= min(epsilon, dom_max - a_u). _widen_if_degenerate grows `d`
           rightward, clamped at the domain max -- so a term already jammed
           against the ceiling can only reach the space left below it. See
           test_ceiling_clipped_umf_is_known_behaviour.
      LMF: >= min(epsilon, UMF span). An LMF may never grow outside the UMF
           containing it (_check_containment would reject the instance), so when
           the UMF was itself clipped, the LMF settles for the UMF's span.

    Asserting a flat epsilon on either would be asserting something the library
    does not, and cannot, promise.
    """
    check_trap('umf', umf)
    check_trap('lmf', lmf)

    a_u, b_u, c_u, d_u = umf
    a_l, b_l, c_l, d_l = lmf
    assert a_u <= a_l and b_u <= b_l and c_u >= c_l and d_u >= d_l, \
        f"UMF {umf} does not contain LMF {lmf}"

    epsilon = _min_width(dom_min, dom_max)
    assert (d_u - a_u) >= min(epsilon, dom_max - a_u) - 1e-12, \
        f"UMF {umf} is below the width floor"
    assert (d_l - a_l) >= min(epsilon, d_u - a_u) - 1e-12, \
        f"LMF {lmf} is below the width floor"


def _pair(it2, field: str):
    return getattr(it2, field + '_umf'), getattr(it2, field + '_lmf')


# ── 1. The bug this closes ───────────────────────────────────────────────────

class TestDegenerateOutputsAreWidened:
    """Each case below produced a trapezoid narrower than ex_fuzzy's grid step
    before the floor was added to expand_contract -- two of them collapsed one
    to a literal point."""

    def test_narrow_core_collapses_lmf_to_a_point(self, toy_schema):
        # General branch (a > dom_min). A rectangular core: eff_d is capped at
        # (c - b) / 2 = 0.2, which is exactly enough for the LMF's contraction
        # to meet in the middle -- lmf becomes (4.2, 4.2, 4.2, 4.2), width 0.
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1(x1_low=(4.0, 4.0, 4.4, 4.4))  # x1 domain is (0.0, 10.0)

        it2 = make_it2_from_t1(toy_schema, t1, delta=1.0, it2_cls=IT2)

        umf, lmf = _pair(it2, 'x1_low')
        assert (lmf[3] - lmf[0]) > 0.0, "LMF is still a zero-width point"
        _assert_valid_pair(umf, lmf, 0.0, 10.0)

    def test_domain_pinned_left_edge_collapses_lmf(self, toy_schema):
        # Pinned branch (a <= dom_min), where eff_d is capped at (c - b) rather
        # than (c - b) / 2. delta >= the flat top drives lmf_c and lmf_d both
        # back onto lmf_b -- lmf becomes (0, 0, 0, 0). The flat-top guard does
        # NOT rescue this: it only fires when c and d sit at dom_max, and here
        # the term ends at 0.5, well short of x2's ceiling of 1.0.
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1(x2_low=(0.0, 0.0, 0.5, 0.5))  # x2 domain is (0.0, 1.0)

        it2 = make_it2_from_t1(toy_schema, t1, delta={'x2': 0.5}, it2_cls=IT2)

        umf, lmf = _pair(it2, 'x2_low')
        assert (lmf[3] - lmf[0]) > 0.0, "LMF is still a zero-width point"
        _assert_valid_pair(umf, lmf, 0.0, 1.0)

    def test_scale_narrowed_trapezoid_umf_below_grid_step(self, toy_schema):
        # A trapezoid a `scale` gene has already squeezed to 0.03 wide. The UMF
        # can only ever be (trap width + 2 * eff_d) wide, and eff_d is capped at
        # (c - b) / 2 = 0.005 -- so the UMF lands at 0.04, under the 0.05 grid
        # step, however generous delta is. Both halves are invisible to
        # ex_fuzzy's consequent sampler.
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1(x1_low=(5.0, 5.01, 5.02, 5.03))

        it2 = make_it2_from_t1(toy_schema, t1, delta=0.005, it2_cls=IT2)

        umf, lmf = _pair(it2, 'x1_low')
        assert (umf[3] - umf[0]) > EX_FUZZY_GRID_STEP, \
            "UMF still fits between two consequent-grid points"
        _assert_valid_pair(umf, lmf, 0.0, 10.0)


# ── 2. The compounding case ──────────────────────────────────────────────────

# The full range a multiplicative delta_scale gene will be allowed to explore.
# The point is not that one adversarial delta breaks it -- it is that a trapezoid
# already narrowed by `scale` is degenerate at EVERY delta, because eff_d is
# capped by that trapezoid's own core ((c - b) / 2 = 0.005) no matter how large
# delta grows. Sweeping the range is what distinguishes "the floor holds where we
# happened to look" from "the floor holds everywhere delta_scale can reach".
DELTA_SCALE_RANGE = [1e-4, 0.01, 0.1, 1.0, 3.0, 10.0]


class TestFloorHoldsAcrossTheDeltaScaleRange:
    @pytest.mark.parametrize('delta', DELTA_SCALE_RANGE)
    def test_scale_narrowed_trapezoid_at_every_delta(self, toy_schema, delta):
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1(x1_low=(5.0, 5.01, 5.02, 5.03))  # 0.03 wide: already sub-grid

        it2 = make_it2_from_t1(toy_schema, t1, delta=delta, it2_cls=IT2)

        umf, lmf = _pair(it2, 'x1_low')
        _assert_valid_pair(umf, lmf, 0.0, 10.0)

    @pytest.mark.parametrize('delta', DELTA_SCALE_RANGE)
    def test_narrow_core_at_every_delta(self, toy_schema, delta):
        # The other degeneracy route (rectangular core) across the same sweep:
        # harmless at small delta, collapses the LMF at large delta. Together
        # with the case above -- degenerate at every delta -- the two cover both
        # ends of the range rather than one lucky sample.
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1(x1_low=(4.0, 4.0, 4.4, 4.4))

        it2 = make_it2_from_t1(toy_schema, t1, delta=delta, it2_cls=IT2)

        umf, lmf = _pair(it2, 'x1_low')
        _assert_valid_pair(umf, lmf, 0.0, 10.0)


# ── 3. The no-op regression check ────────────────────────────────────────────
#
# The floor must not perturb params an expert deliberately chose -- if it did, it
# would be silently redesigning the seed FLS rather than repairing degenerate
# search points.
#
# The real consuming application's DEFAULT_IT2_MF_PARAMS cannot be imported here
# (it is application-specific, and this library must be testable with zero
# knowledge of any consumer), so the schema below is a synthetic stand-in matched
# to it on every axis the floor actually depends on.
#
# It is deliberately NOT the `app_shaped_schema` fixture in
# test_mf_optimize_mutation_parity.py, and should not be merged with it. That one
# matches the real chromosome's gene COUNTS (17 terms / 34 IT2 fields / 136 MF
# genes) -- which this one also does -- but every variable in it sits on the same
# (0.0, 1.0) domain, with uniformly wide terms. Both properties are load-bearing
# here and wrong for this test:
#
#   * _min_width is max(1e-3 * domain_width, 0.06). A uniform-domain schema
#     exercises only one of those two terms. The domains below span (0, 1) to
#     (0, 100), so the absolute floor binds on some variables and the
#     domain-relative floor on others.
#   * Uniformly wide terms cannot reproduce the thin (~0.08) output LMFs of a
#     real expert design, which are what sit closest to the floor -- i.e. the
#     only place a no-op claim is actually at risk.
#
# Widening the parity fixture to cover these would change the MF gene bounds it
# draws its population from, perturbing the mutation rates that module exists to
# pin. Two schemas, two jobs.

def _var(name: str, domain, defaults) -> VariableSpec:
    return VariableSpec(
        name=name,
        domain=domain,
        terms=tuple(
            TermSpec(f"T{i}", f"{name}_t{i}", default=d)
            for i, d in enumerate(defaults)
        ),
    )


@pytest.fixture
def mixed_scale_schema() -> Schema:
    """4 antecedents + 1 output, 17 terms -> 34 IT2 fields -> 136 MF genes, on
    domains whose scales differ by two orders of magnitude."""
    return Schema(
        antecedents=(
            _var("v0", (0.0, 12.0), [
                (-0.1, 0.0, 2.0, 4.0), (2.0, 4.0, 6.0, 8.0), (6.0, 8.0, 12.0, 12.0)]),
            _var("v1", (0.0, 1.0), [
                (-0.01, 0.0, 0.2, 0.4), (0.2, 0.4, 0.6, 0.8), (0.6, 0.8, 1.0, 1.0)]),
            _var("v2", (0.0, 1.0), [
                (-0.01, 0.0, 0.25, 0.45), (0.25, 0.45, 0.6, 0.8), (0.6, 0.8, 1.0, 1.0)]),
            _var("v3", (0.0, 100.0), [
                (-1.0, 0.0, 20.0, 40.0), (20.0, 40.0, 60.0, 80.0),
                (60.0, 80.0, 100.0, 100.0)]),
        ),
        output=_var("out", (0.0, 1.0), [
            (-0.01, 0.0, 0.08, 0.15), (0.10, 0.13, 0.27, 0.30),
            (0.35, 0.38, 0.42, 0.45), (0.55, 0.62, 0.68, 0.75),
            (0.80, 0.88, 1.0, 1.0)]),
    )


# Per-variable FOU deltas of the same magnitudes a real expert design uses --
# spanning two orders of magnitude, like the domains they belong to.
EXPERT_DELTAS = {'v0': 0.30, 'v1': 0.07, 'v2': 0.08, 'v3': 2.50, 'out': 0.01}


class TestFloorIsANoOpOnExpertParams:
    def test_synthetic_schema_reproduces_the_real_chromosome_shape(self, mixed_scale_schema):
        """If this drifts, the no-op claim below is being made about the wrong
        shape of MF chromosome."""
        IT2 = build_it2_mf_params_class(mixed_scale_schema)
        n_terms = sum(len(v.terms) for v in mixed_scale_schema.all_vars)
        assert n_terms == 17
        assert len(dataclasses.fields(IT2)) == 34
        assert len(dataclasses.fields(IT2)) * 4 == 136

    def test_floor_is_a_noop_on_expert_deltas(self, mixed_scale_schema):
        # Assert the floor is the IDENTITY on every field, rather than merely
        # that every field ends up above it. Both helpers early-return when the
        # trapezoid is already wide enough, so re-applying them to a healthy
        # output must change nothing -- and this catches a helper that stopped
        # early-returning, which a "widths all exceed the floor" assertion would
        # sail straight past.
        T1 = build_mf_params_class(mixed_scale_schema)
        IT2 = build_it2_mf_params_class(mixed_scale_schema)
        it2 = make_it2_from_t1(mixed_scale_schema, T1(), EXPERT_DELTAS, IT2)

        field_domain = mixed_scale_schema.field_domains()
        for f in dataclasses.fields(IT2):
            if not f.name.endswith('_umf'):
                continue
            base = f.name[:-4]
            dom_min, dom_max = field_domain[base]
            epsilon = _min_width(dom_min, dom_max)
            umf, lmf = _pair(it2, base)

            assert _widen_if_degenerate(*umf, epsilon, dom_max) == umf, \
                f"{base}_umf was perturbed by the width floor"
            assert _widen_lmf_if_degenerate(*lmf, umf[0], umf[3], epsilon) == lmf, \
                f"{base}_lmf was perturbed by the width floor"

    def test_tightest_expert_trap_margin_over_floor(self, mixed_scale_schema):
        # The no-op above holds, but not by much: the thinnest output LMFs of a
        # real expert design are ~0.08 wide against a 0.06 floor -- a margin of
        # only 1.33x. Pinned here so that raising _MIN_WIDTH_ABS past ~0.08
        # fails loudly, instead of silently widening deliberately-chosen MF
        # params on every make_it2_from_t1 call.
        T1 = build_mf_params_class(mixed_scale_schema)
        IT2 = build_it2_mf_params_class(mixed_scale_schema)
        it2 = make_it2_from_t1(mixed_scale_schema, T1(), EXPERT_DELTAS, IT2)

        field_domain = mixed_scale_schema.field_domains()
        margins = []
        for f in dataclasses.fields(IT2):
            base = f.name[:-4]
            dom_min, dom_max = field_domain[base]
            a, _, _, d = getattr(it2, f.name)
            margins.append((d - a) / _min_width(dom_min, dom_max))

        assert min(margins) > 1.0, "an expert trapezoid is at or under the floor"
        assert min(margins) == pytest.approx(1.33, abs=0.01)


# ── 4. Invariants hold on every widened output ───────────────────────────────

class TestInvariantsSurviveWidening:
    def test_domain_pinned_left_edge_preserved(self, toy_schema):
        # x1_low's default `a` (-0.1) sits below dom_min by convention (the
        # epsilon-below-floor trick for full membership at the boundary). The
        # floor widens only rightward on a UMF, so it must not lift `a` off that
        # pin -- the same property test_respects_domain_bounds guards for the
        # pre-existing clamps.
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1()

        it2 = make_it2_from_t1(toy_schema, t1, delta=0.1, it2_cls=IT2)

        assert it2.x1_low_umf[0] == t1.x1_low[0]

    def test_ceiling_clipped_umf_is_known_behaviour(self, toy_schema):
        # KNOWN LIMIT, pinned deliberately rather than papered over.
        #
        # _widen_if_degenerate widens a UMF by pushing `d` right, clamped at
        # dom_max. A term jammed against the domain ceiling therefore cannot
        # reach the absolute floor -- there is no room left to grow into. The LMF
        # then correctly settles for min(epsilon, UMF span) instead of breaking
        # containment to hit the floor.
        #
        # This is pre-existing behaviour shared with from_vector (see
        # _widen_lmf_if_degenerate's docstring: the min() "can only bind when the
        # UMF's own widening was itself clipped at the domain max"), not
        # something make_it2_from_t1's floor introduced. Closing it would mean
        # changing a helper from_vector depends on, which is a separate decision.
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        t1 = T1(x2_high=(0.97, 0.98, 1.0, 1.0))  # x2 domain is (0.0, 1.0)

        it2 = make_it2_from_t1(toy_schema, t1, delta={'x2': 0.01}, it2_cls=IT2)

        umf, lmf = _pair(it2, 'x2_high')
        epsilon = _min_width(0.0, 1.0)
        umf_span = umf[3] - umf[0]

        assert umf_span < epsilon           # clipped at the ceiling, as documented
        assert umf[3] == 1.0                # ...because it is already at dom_max
        assert (lmf[3] - lmf[0]) == pytest.approx(min(epsilon, umf_span))
        assert lmf[2] == 1.0 and lmf[3] == 1.0  # flat top still pinned at dom_max
        _assert_valid_pair(umf, lmf, 0.0, 1.0)  # ordering + containment regardless

    def test_random_traps_always_construct(self, toy_schema):
        # Repair guarantee: whatever a shift/scale/delta_scale codec throws at
        # make_it2_from_t1, the result must still be a constructible IT2MFParams
        # -- __post_init__ runs check_trap on all 18 fields and _check_containment
        # on all 9 pairs, so merely getting an instance back is already most of
        # the assertion.
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        rng = np.random.default_rng(0)

        for _ in range(200):
            # A valid but arbitrarily-shaped T1 trapezoid on x1's (0, 10) domain,
            # including hairlines and rectangles.
            pts = np.sort(rng.uniform(0.0, 10.0, size=4))
            if rng.random() < 0.3:          # force a narrow core sometimes
                pts[2] = pts[1] + rng.uniform(0.0, 0.01)
                pts = np.sort(pts)
            t1 = T1(x1_low=tuple(float(p) for p in pts))
            delta = float(rng.choice(DELTA_SCALE_RANGE))

            it2 = make_it2_from_t1(toy_schema, t1, delta=delta, it2_cls=IT2)

            umf, lmf = _pair(it2, 'x1_low')
            _assert_valid_pair(umf, lmf, 0.0, 10.0)
