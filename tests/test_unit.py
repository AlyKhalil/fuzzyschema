"""
tests/test_unit.py
------------------
Tests for FLSUnit / HierarchicalFLS. Synthetic schemas only -- see conftest.

The two properties worth the most care here, because a break in either is
SILENT (no exception, just wrong numbers, for an entire GA run):

  1. No shared mutable state. build_engine/run_inference must never mutate the
     unit's default_params or the list default_rules_fn returns. Tested against
     BOTH engine types: T2 has a separate construction path (RuleBaseT2's
     consequent-centroid computation, already patched once in engine.py), so T1
     passing is not evidence T2 passes.

  2. The chain actually chains. Unit B's 'prev' antecedent must read unit A's
     OUTPUT column, not a raw column that happens to be nearby. The conftest
     schemas name B's antecedent 'prev' rather than 'a_out' precisely so that
     an implementation ignoring column_map cannot pass by name-coincidence.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from ex_fuzzy.rules import RuleSimple

from fuzzyschema.chromosome import RuleChromosomeCodec
from fuzzyschema.engine import T1FLSEngine, T2FLSEngine
from fuzzyschema.unit import FLSUnit, HierarchicalFLS

from conftest import dense_rules_fn, make_unit

ENGINE_TYPES = ['t1', 't2']


def _snapshot(unit) -> tuple:
    """Everything about a unit's defaults that an engine build could plausibly
    corrupt: the params' full float vector, and the rule base's antecedent
    arrays + consequents."""
    params_vec = unit.default_params.to_vector().copy()
    rules = unit.default_rules_fn()
    rules_repr = [([int(a) for a in r.antecedents], int(r.consequent)) for r in rules]
    return params_vec, rules_repr


# ── Construction ──────────────────────────────────────────────────────────────

class TestFLSUnitConstruction:

    @pytest.mark.parametrize('engine_type,expected_cls',
                             [('t1', T1FLSEngine), ('t2', T2FLSEngine)])
    def test_build_engine_returns_the_engine_for_its_type(
        self, gapped_schema, engine_type, expected_cls,
    ):
        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                         engine_type=engine_type)

        assert isinstance(unit.build_engine(), expected_cls)

    def test_unknown_engine_type_raises_at_construction(self, gapped_schema):
        # Must fail when the unit is built, not on the first GA candidate.
        with pytest.raises(ValueError, match="unknown engine_type"):
            make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                      engine_type='t3')

    def test_chromosome_codec_matches_the_units_schema(self, schema_a):
        unit = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))

        codec = unit.chromosome_codec()

        assert isinstance(codec, RuleChromosomeCodec)
        assert codec.schema is schema_a
        assert codec.chrom_len == 2 * 2          # two 2-term antecedents
        assert codec.n_consequents == 2

    def test_chromosome_codec_is_stateless(self, schema_a):
        unit = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))

        assert unit.chromosome_codec() is not unit.chromosome_codec()

    def test_build_engine_rejects_rules_that_dont_match_the_schema(self, schema_a):
        # schema_a has 2 antecedents; a 1-antecedent rule base is a mismatch and
        # must be caught by validate_rules, not surface as an ex_fuzzy crash.
        unit = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))

        with pytest.raises(ValueError, match="antecedent length"):
            unit.build_engine(rules=[RuleSimple([0], consequent=0)])


# ── run_inference ─────────────────────────────────────────────────────────────

class TestRunInference:

    @pytest.mark.parametrize('engine_type', ENGINE_TYPES)
    def test_scores_track_the_rule_bases_consequent(self, gapped_schema, engine_type):
        # The mutation-worthy version of "returns finite numbers": the SAME
        # inputs, run against a rule base mapping everything to the LOW output
        # term vs. one mapping everything to HIGH, must produce systematically
        # lower scores. A stubbed/ignored rule base, or params read from the
        # wrong term, breaks this; a shape-only assertion would not notice.
        df = pd.DataFrame({'x': [0.1, 0.9]})

        all_low = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: 0),
                            engine_type=engine_type)
        all_high = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: 1),
                             engine_type=engine_type)

        low_scores = all_low.run_inference(df)
        high_scores = all_high.run_inference(df)

        assert low_scores.shape == (2,)
        assert np.all(np.isfinite(low_scores)) and np.all(np.isfinite(high_scores))
        assert np.all(high_scores > low_scores)

    @pytest.mark.parametrize('engine_type', ENGINE_TYPES)
    def test_rules_argument_overrides_the_default_rule_base(self, gapped_schema, engine_type):
        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: 0),
                         engine_type=engine_type)
        df = pd.DataFrame({'x': [0.1, 0.9]})

        default_scores = unit.run_inference(df)
        overridden = unit.run_inference(df, rules=dense_rules_fn(1, lambda c: 1)())

        assert np.all(overridden > default_scores)

    @pytest.mark.parametrize('engine_type', ENGINE_TYPES)
    def test_params_argument_overrides_the_default_params(self, gapped_schema, engine_type):
        # Shift the output terms' trapezoids upward; with the rule base held
        # fixed, every score must move up. Catches params being ignored.
        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                         engine_type=engine_type)
        df = pd.DataFrame({'x': [0.1, 0.9]})

        default_scores = unit.run_inference(df)

        shifted = _shift_output_terms(unit, +0.05)
        shifted_scores = unit.run_inference(df, params=shifted)

        assert np.all(shifted_scores > default_scores)

    def test_missing_input_column_raises_naming_it(self, schema_a):
        unit = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        df = pd.DataFrame({'in1': [0.1]})  # in2 absent

        with pytest.raises(ValueError, match="missing input column"):
            unit.run_inference(df)

    def test_extra_columns_are_ignored(self, gapped_schema):
        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]))

        bare = unit.run_inference(pd.DataFrame({'x': [0.1, 0.9]}))
        with_extra = unit.run_inference(
            pd.DataFrame({'x': [0.1, 0.9], 'unrelated': ['a', 'b']})
        )

        np.testing.assert_array_equal(bare, with_extra)


def _shift_output_terms(unit, delta: float):
    """Return a copy of unit.default_params with every OUTPUT term's trapezoid
    shifted by `delta`, leaving antecedent terms untouched. Used to prove the
    params argument is actually consumed."""
    out_fields = {t.field for t in unit.schema.output.terms}
    kwargs = {}
    for f in dataclasses.fields(unit.default_params):
        val = getattr(unit.default_params, f.name)
        base = f.name[:-4] if f.name.endswith(('_umf', '_lmf')) else f.name
        kwargs[f.name] = tuple(v + delta for v in val) if base in out_fields else val
    return type(unit.default_params)(**kwargs)


# ── fallback_fn ───────────────────────────────────────────────────────────────

class TestFallback:
    """gapped_schema's LOW/HIGH terms leave ~(0.35, 0.65) uncovered, so x=0.5
    fires no rule and the engine scores it NaN. That is the only way to reach
    these paths deliberately."""

    @pytest.mark.parametrize('engine_type', ENGINE_TYPES)
    def test_nan_propagates_when_fallback_is_none(self, gapped_schema, engine_type):
        # Documented, deliberate behaviour: a zero-firing row is NOT coerced to
        # 0.0 and does NOT raise -- the caller must be able to see the gap.
        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                         engine_type=engine_type, fallback_fn=None)

        scores = unit.run_inference(pd.DataFrame({'x': [0.1, 0.5, 0.9]}))

        assert np.isnan(scores[1])
        assert not np.isnan(scores[0]) and not np.isnan(scores[2])
        assert scores[1] != 0.0 or np.isnan(scores[1])  # explicitly not 0.0

    @pytest.mark.parametrize('engine_type', ENGINE_TYPES)
    def test_fallback_fills_only_the_zero_firing_rows(self, gapped_schema, engine_type):
        # The rows that DID fire must keep their engine scores. A fallback that
        # was handed the whole frame and written back wholesale would clobber
        # them -- this is what pins the "NaN rows only" contract.
        rules = dense_rules_fn(1, lambda c: c[0])
        df = pd.DataFrame({'x': [0.1, 0.5, 0.9]})

        without = make_unit('u', gapped_schema, rules, engine_type=engine_type)
        with_fb = make_unit('u', gapped_schema, rules, engine_type=engine_type,
                            fallback_fn=lambda d: np.full(len(d), 0.42))

        base = without.run_inference(df)
        filled = with_fb.run_inference(df)

        assert filled[1] == pytest.approx(0.42)
        np.testing.assert_allclose(filled[[0, 2]], base[[0, 2]])

    def test_fallback_receives_only_the_nan_rows(self, gapped_schema):
        # Assert on what the callable actually saw, not just on the result:
        # exactly the zero-firing row, with its original index preserved.
        seen = {}

        def fallback(d):
            seen['index'] = list(d.index)
            seen['x'] = list(d['x'])
            return np.zeros(len(d))

        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                         fallback_fn=fallback)

        unit.run_inference(pd.DataFrame({'x': [0.1, 0.5, 0.9]}))

        assert seen['index'] == [1]
        assert seen['x'] == [0.5]

    def test_fallback_is_not_called_when_no_row_is_nan(self, gapped_schema):
        calls = []

        def fallback(d):
            calls.append(len(d))
            return np.zeros(len(d))

        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                         fallback_fn=fallback)

        unit.run_inference(pd.DataFrame({'x': [0.1, 0.9]}))  # both rows fire

        assert calls == []

    def test_fallback_returning_the_wrong_length_raises(self, gapped_schema):
        # e.g. a fallback written against the full frame: it would return 3
        # values for 1 NaN row. Must raise, never broadcast or truncate.
        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                         fallback_fn=lambda d: np.zeros(3))

        with pytest.raises(ValueError, match="fallback_fn returned shape"):
            unit.run_inference(pd.DataFrame({'x': [0.1, 0.5, 0.9]}))

    def test_fallback_values_are_positional_not_index_aligned(self, gapped_schema):
        # A non-default index must not silently re-align the fallback's values.
        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                         fallback_fn=lambda d: np.array([0.77]))
        df = pd.DataFrame({'x': [0.1, 0.5, 0.9]}, index=[10, 20, 30])

        scores = unit.run_inference(df)

        assert scores[1] == pytest.approx(0.77)


# ── HierarchicalFLS chaining ──────────────────────────────────────────────────

class TestChaining:

    def test_two_unit_chain_feeds_a_out_into_bs_prev_antecedent(
        self, schema_a, schema_b,
    ):
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        unit_b = make_unit('b', schema_b, dense_rules_fn(2, lambda c: c[0]))
        hfls = HierarchicalFLS([
            ('a', unit_a, {'in1': 'in1', 'in2': 'in2'}),
            ('b', unit_b, {'prev': 'a_out', 'in3': 'in3'}),
        ])
        df = pd.DataFrame({
            'in1': [0.1, 0.9, 0.1],
            'in2': [0.1, 0.9, 0.9],
            'in3': [0.9, 0.1, 0.9],
        })

        out = hfls.run_inference(df)

        # A's column is exactly what A produces standalone.
        a_standalone = unit_a.run_inference(df)
        np.testing.assert_allclose(out['a_out'].to_numpy(), a_standalone)

        # B's column is exactly what B produces when its 'prev' antecedent is
        # fed A's scores -- i.e. the chain really passed A's OUTPUT along, and
        # not some raw column. Note B's antecedent is 'prev', so this frame is
        # built by renaming, exactly as column_map specifies.
        b_input = pd.DataFrame({'prev': a_standalone, 'in3': df['in3']})
        np.testing.assert_allclose(out['b_out'].to_numpy(),
                                   unit_b.run_inference(b_input))

    def test_changing_only_as_inputs_changes_bs_output(self, schema_a, schema_b):
        # The behavioural proof that B consumes A: perturb in1/in2, which feed
        # ONLY unit A, hold in3 fixed, and B's scores must move. If column_map
        # silently fed B a raw column instead, B would be unaffected.
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        unit_b = make_unit('b', schema_b, dense_rules_fn(2, lambda c: c[0]))
        hfls = HierarchicalFLS([
            ('a', unit_a, {'in1': 'in1', 'in2': 'in2'}),
            ('b', unit_b, {'prev': 'a_out', 'in3': 'in3'}),
        ])

        low_a = hfls.run_inference(pd.DataFrame({'in1': [0.1], 'in2': [0.1], 'in3': [0.9]}))
        high_a = hfls.run_inference(pd.DataFrame({'in1': [0.9], 'in2': [0.9], 'in3': [0.9]}))

        assert high_a['a_out'][0] > low_a['a_out'][0]
        assert high_a['b_out'][0] > low_a['b_out'][0]

    def test_three_unit_chain_propagates_through_both_hops(
        self, schema_a, schema_b, schema_c,
    ):
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        unit_b = make_unit('b', schema_b, dense_rules_fn(2, lambda c: c[0]))
        unit_c = make_unit('c', schema_c, dense_rules_fn(1, lambda c: c[0]))
        hfls = HierarchicalFLS([
            ('a', unit_a, {'in1': 'in1', 'in2': 'in2'}),
            ('b', unit_b, {'prev': 'a_out', 'in3': 'in3'}),
            ('c', unit_c, {'prev2': 'b_out'}),
        ])
        df = pd.DataFrame({'in1': [0.1, 0.9], 'in2': [0.1, 0.9], 'in3': [0.1, 0.9]})

        out = hfls.run_inference(df)

        assert list(out.columns) == ['in1', 'in2', 'in3', 'a_out', 'b_out', 'c_out']
        # Monotone chain: the all-high row must beat the all-low row at EVERY
        # hop, which can only happen if each unit read the previous one's output.
        for col in ('a_out', 'b_out', 'c_out'):
            assert out[col][1] > out[col][0]

    def test_output_column_name_comes_from_the_schema_not_the_unit_name(
        self, schema_a,
    ):
        # The confirmed naming rule: the column is schema.output.name
        # ('a_out'), NOT the unit_name in the units list ('some_other_name').
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        hfls = HierarchicalFLS([
            ('some_other_name', unit_a, {'in1': 'in1', 'in2': 'in2'}),
        ])

        out = hfls.run_inference(pd.DataFrame({'in1': [0.1], 'in2': [0.1]}))

        assert 'a_out' in out.columns
        assert 'some_other_name' not in out.columns
        assert 'some_other_name_output' not in out.columns

    def test_a_units_output_overwrites_a_preexisting_column_of_that_name(
        self, schema_a, schema_b,
    ):
        # If the caller's frame already has an 'a_out' column, the unit's own
        # scores must win -- otherwise B would silently consume stale data.
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        unit_b = make_unit('b', schema_b, dense_rules_fn(2, lambda c: c[0]))
        hfls = HierarchicalFLS([
            ('a', unit_a, {'in1': 'in1', 'in2': 'in2'}),
            ('b', unit_b, {'prev': 'a_out', 'in3': 'in3'}),
        ])
        df = pd.DataFrame({
            'in1': [0.9], 'in2': [0.9], 'in3': [0.9],
            'a_out': [-999.0],  # decoy
        })

        out = hfls.run_inference(df)

        np.testing.assert_allclose(out['a_out'].to_numpy(), unit_a.run_inference(df))
        assert out['a_out'][0] != -999.0

    def test_caller_dataframe_is_not_mutated(self, schema_a):
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        hfls = HierarchicalFLS([('a', unit_a, {'in1': 'in1', 'in2': 'in2'})])
        df = pd.DataFrame({'in1': [0.1], 'in2': [0.1]})
        before = df.copy()

        out = hfls.run_inference(df)

        assert 'a_out' not in df.columns
        assert 'a_out' in out.columns
        pd.testing.assert_frame_equal(df, before)


# ── column_map validation ─────────────────────────────────────────────────────

class TestColumnMapValidation:

    def test_column_map_missing_an_antecedent_raises_at_construction(self, schema_a):
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))

        with pytest.raises(ValueError, match="missing antecedent"):
            HierarchicalFLS([('a', unit_a, {'in1': 'in1'})])  # in2 unmapped

    def test_column_map_with_an_unknown_key_raises_at_construction(self, schema_a):
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))

        with pytest.raises(ValueError, match="not antecedents"):
            HierarchicalFLS([
                ('a', unit_a, {'in1': 'in1', 'in2': 'in2', 'in9': 'in9'}),
            ])

    def test_mapped_column_absent_from_dataframe_raises_with_a_clear_message(
        self, schema_a,
    ):
        # The failure the design doc calls out: must raise immediately, naming
        # unit/antecedent/column/available -- not a bare KeyError or a NaN column.
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        hfls = HierarchicalFLS([('a', unit_a, {'in1': 'in1', 'in2': 'typo_col'})])

        with pytest.raises(ValueError) as exc:
            hfls.run_inference(pd.DataFrame({'in1': [0.1], 'in2': [0.1]}))

        msg = str(exc.value)
        assert "'a'" in msg and "'in2'" in msg and "'typo_col'" in msg
        assert "Available columns" in msg

    def test_unit_ordered_before_its_producer_raises(self, schema_a, schema_b):
        # B is listed FIRST, so 'a_out' does not exist yet when B runs. This is
        # the ordering mistake the column-presence check exists to catch.
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        unit_b = make_unit('b', schema_b, dense_rules_fn(2, lambda c: c[0]))
        hfls = HierarchicalFLS([
            ('b', unit_b, {'prev': 'a_out', 'in3': 'in3'}),
            ('a', unit_a, {'in1': 'in1', 'in2': 'in2'}),
        ])

        with pytest.raises(ValueError, match="a_out"):
            hfls.run_inference(pd.DataFrame({'in1': [0.1], 'in2': [0.1], 'in3': [0.1]}))

    def test_duplicate_unit_names_raise_at_construction(self, schema_a, schema_b):
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: c[0]))
        unit_b = make_unit('b', schema_b, dense_rules_fn(2, lambda c: c[0]))

        with pytest.raises(ValueError, match="duplicate unit name"):
            HierarchicalFLS([
                ('dup', unit_a, {'in1': 'in1', 'in2': 'in2'}),
                ('dup', unit_b, {'prev': 'a_out', 'in3': 'in3'}),
            ])


# ── overrides ─────────────────────────────────────────────────────────────────

class TestOverrides:

    @pytest.fixture
    def chain(self, schema_a, schema_b):
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: 0))
        unit_b = make_unit('b', schema_b, dense_rules_fn(2, lambda c: 0))
        hfls = HierarchicalFLS([
            ('a', unit_a, {'in1': 'in1', 'in2': 'in2'}),
            ('b', unit_b, {'prev': 'a_out', 'in3': 'in3'}),
        ])
        df = pd.DataFrame({'in1': [0.1, 0.9], 'in2': [0.1, 0.9], 'in3': [0.1, 0.9]})
        return hfls, unit_a, unit_b, df

    def test_rules_override_alone_leaves_params_at_the_units_default(self, chain):
        # The explicit judgment call: supplying 'rules' must NOT require also
        # supplying 'params'. Result must equal running the unit standalone with
        # the overridden rules and the unit's DEFAULT params.
        hfls, unit_a, _, df = chain

        out = hfls.run_inference(
            df, overrides={'a': {'rules': dense_rules_fn(2, lambda c: 1)()}},
        )

        expected = unit_a.run_inference(df, rules=dense_rules_fn(2, lambda c: 1)())
        np.testing.assert_allclose(out['a_out'].to_numpy(), expected)
        # And it genuinely changed something vs. the unoverridden run.
        base = hfls.run_inference(df)
        assert np.all(out['a_out'].to_numpy() > base['a_out'].to_numpy())

    def test_params_override_alone_leaves_rules_at_the_units_default(self, chain):
        hfls, unit_a, _, df = chain
        shifted = _shift_output_terms(unit_a, +0.05)

        out = hfls.run_inference(df, overrides={'a': {'params': shifted}})

        expected = unit_a.run_inference(df, params=shifted)
        np.testing.assert_allclose(out['a_out'].to_numpy(), expected)
        base = hfls.run_inference(df)
        assert np.all(out['a_out'].to_numpy() > base['a_out'].to_numpy())

    def test_overriding_one_unit_leaves_the_others_defaults_alone(self, chain):
        # The freeze-vs-co-optimize mechanism: an override for 'b' must not
        # perturb 'a'. (a runs upstream of b, so a's column must be identical.)
        hfls, _, _, df = chain

        base = hfls.run_inference(df)
        out = hfls.run_inference(
            df, overrides={'b': {'rules': dense_rules_fn(2, lambda c: 1)()}},
        )

        np.testing.assert_array_equal(out['a_out'].to_numpy(),
                                      base['a_out'].to_numpy())
        assert np.all(out['b_out'].to_numpy() > base['b_out'].to_numpy())

    def test_empty_overrides_equals_no_overrides(self, chain):
        hfls, _, _, df = chain

        np.testing.assert_array_equal(
            hfls.run_inference(df, overrides={})['b_out'].to_numpy(),
            hfls.run_inference(df)['b_out'].to_numpy(),
        )

    def test_override_for_an_unknown_unit_raises(self, chain):
        hfls, _, _, df = chain

        with pytest.raises(ValueError, match="unknown unit"):
            hfls.run_inference(df, overrides={'stage_typo': {'rules': []}})

    def test_override_with_an_unknown_key_raises(self, chain):
        # 'param' instead of 'params': silently ignoring this would mean the
        # override the caller thought they applied simply didn't happen.
        hfls, unit_a, _, df = chain

        with pytest.raises(ValueError, match="unknown key"):
            hfls.run_inference(df, overrides={'a': {'param': unit_a.default_params}})


# ── No shared mutable state ───────────────────────────────────────────────────

class TestNoSharedMutableState:
    """The silent-corruption class of bug. A GA calls run_inference once per
    candidate chromosome with different overrides each time; if that mutated the
    unit's own defaults, every later candidate would be scored against a moving
    baseline, with no exception and no visible symptom."""

    @pytest.mark.parametrize('engine_type', ENGINE_TYPES)
    def test_repeated_run_inference_with_different_overrides_leaves_defaults_intact(
        self, schema_a, schema_b, engine_type,
    ):
        unit_a = make_unit('a', schema_a, dense_rules_fn(2, lambda c: 0),
                           engine_type=engine_type)
        unit_b = make_unit('b', schema_b, dense_rules_fn(2, lambda c: 0),
                           engine_type=engine_type)
        hfls = HierarchicalFLS([
            ('a', unit_a, {'in1': 'in1', 'in2': 'in2'}),
            ('b', unit_b, {'prev': 'a_out', 'in3': 'in3'}),
        ])
        df = pd.DataFrame({'in1': [0.1, 0.9], 'in2': [0.9, 0.1], 'in3': [0.1, 0.9]})

        before_a, before_b = _snapshot(unit_a), _snapshot(unit_b)
        first_default = hfls.run_inference(df)['b_out'].to_numpy()

        hfls.run_inference(df, overrides={'a': {'rules': dense_rules_fn(2, lambda c: 1)()}})
        hfls.run_inference(df, overrides={
            'b': {'rules': dense_rules_fn(2, lambda c: 1)(),
                  'params': _shift_output_terms(unit_b, +0.05)},
        })

        after_a, after_b = _snapshot(unit_a), _snapshot(unit_b)
        np.testing.assert_array_equal(before_a[0], after_a[0])   # params vector
        np.testing.assert_array_equal(before_b[0], after_b[0])
        assert before_a[1] == after_a[1]                          # rule base
        assert before_b[1] == after_b[1]

        # The real test of the property: a default run AFTER two overridden runs
        # must reproduce the default run from BEFORE them, exactly.
        np.testing.assert_array_equal(hfls.run_inference(df)['b_out'].to_numpy(),
                                      first_default)

    @pytest.mark.parametrize('engine_type', ENGINE_TYPES)
    def test_build_engine_does_not_mutate_the_rules_list_it_is_given(
        self, gapped_schema, engine_type,
    ):
        # ex_fuzzy's RuleBase de-duplicates its rules; that must not be visible
        # in the caller's list (it builds a new one -- verified, but pinned here
        # so an ex_fuzzy upgrade that changes it fails loudly).
        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                         engine_type=engine_type)
        rules = dense_rules_fn(1, lambda c: c[0])()
        before = [([int(a) for a in r.antecedents], int(r.consequent)) for r in rules]
        n_before = len(rules)

        unit.build_engine(rules=rules)

        assert len(rules) == n_before
        assert [([int(a) for a in r.antecedents], int(r.consequent)) for r in rules] == before

    @pytest.mark.parametrize('engine_type', ENGINE_TYPES)
    def test_repeated_default_builds_are_independent(self, gapped_schema, engine_type):
        # Two engines built from the same defaults must not share a rules list.
        unit = make_unit('u', gapped_schema, dense_rules_fn(1, lambda c: c[0]),
                         engine_type=engine_type)

        e1, e2 = unit.build_engine(), unit.build_engine()

        assert e1._rb.rules is not e2._rb.rules
