import dataclasses

import numpy as np
import pytest

from fuzzyschema.mf_params import build_mf_params_class, get_antecedents, get_output_var
from fuzzyschema.variable_config import Schema, VariableSpec, TermSpec


class TestBuildMFParamsClass:
    def test_field_count_matches_schema(self, toy_schema):
        # 2 + 3 antecedent terms + 4 output terms = 9
        MFParams = build_mf_params_class(toy_schema)
        assert len(dataclasses.fields(MFParams)) == 9

    def test_field_names_match_termspec_fields(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        names = {f.name for f in dataclasses.fields(MFParams)}
        assert names == {
            'x1_low', 'x1_high', 'x2_low', 'x2_med', 'x2_high',
            'y_t1', 'y_t2', 'y_t3', 'y_t4',
        }

    def test_no_arg_construction_uses_defaults(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        p = MFParams()
        assert p.x1_low == (-0.1, 0.0, 4.0, 6.0)
        assert p.y_t4 == (3.5, 4.0, 5.0, 5.0)

    def test_keyword_override(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        p = MFParams(x1_low=(0.0, 0.0, 3.0, 5.0))
        assert p.x1_low == (0.0, 0.0, 3.0, 5.0)
        assert p.x1_high == (4.0, 6.0, 10.0, 10.0)  # untouched

    def test_missing_default_raises(self):
        schema = Schema(
            antecedents=(VariableSpec("x", (0.0, 1.0), (TermSpec("A", "x_a"),)),),
            output=VariableSpec("y", (0.0, 1.0), (TermSpec("B", "y_b"),)),
        )
        with pytest.raises(ValueError, match="no default"):
            build_mf_params_class(schema)

    def test_invalid_trap_rejected_on_construction(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        with pytest.raises(ValueError):
            MFParams(x1_low=(5.0, 0.0, 4.0, 6.0))  # a > b

    def test_two_schemas_produce_independent_classes(self, toy_schema, single_antecedent_schema):
        ClassA = build_mf_params_class(toy_schema, class_name="A")
        ClassB = build_mf_params_class(single_antecedent_schema, class_name="B")
        assert ClassA is not ClassB
        assert {f.name for f in dataclasses.fields(ClassA)} != {f.name for f in dataclasses.fields(ClassB)}

    def test_single_antecedent_schema(self, single_antecedent_schema):
        MFParams = build_mf_params_class(single_antecedent_schema)
        assert len(dataclasses.fields(MFParams)) == 4  # 2 in + 2 out
        p = MFParams()
        assert p.in_low == (-0.01, 0.0, 0.3, 0.5)


class TestFromVector:
    def test_round_trip_sorted(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        n = len(dataclasses.fields(MFParams))
        # Deliberately unsorted within each 4-block; from_vector must sort.
        v = np.tile([3.0, 1.0, 4.0, 2.0], n)
        p = MFParams.from_vector(v)
        for f in dataclasses.fields(p):
            trap = getattr(p, f.name)
            assert list(trap) == sorted(trap)

    def test_wrong_length_raises(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        with pytest.raises(ValueError, match="Expected"):
            MFParams.from_vector(np.zeros(5))


class TestToVector:
    def test_length_matches_from_vector_expectation(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        p = MFParams()
        n = len(dataclasses.fields(MFParams))
        assert p.to_vector().shape == (n * 4,)

    def test_round_trip_default_instance(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        p = MFParams()
        p2 = MFParams.from_vector(p.to_vector())
        for f in dataclasses.fields(MFParams):
            assert getattr(p, f.name) == getattr(p2, f.name)

    def test_round_trip_custom_instance(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        p = MFParams(x1_low=(0.0, 1.0, 2.0, 3.0), x2_high=(0.1, 0.2, 0.3, 0.9))
        p2 = MFParams.from_vector(p.to_vector())
        assert p2.x1_low == (0.0, 1.0, 2.0, 3.0)
        assert p2.x2_high == (0.1, 0.2, 0.3, 0.9)

    def test_field_order_matches_declaration(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        p = MFParams()
        v = p.to_vector()
        expected = np.concatenate(
            [np.asarray(getattr(p, f.name), dtype=float) for f in dataclasses.fields(MFParams)]
        )
        assert np.array_equal(v, expected)


class TestGetAntecedents:
    def test_returns_one_per_antecedent(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        ants = get_antecedents(toy_schema, MFParams())
        assert len(ants) == 2
        assert [a.name for a in ants] == ['x1', 'x2']

    def test_term_counts_match_schema(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        ants = get_antecedents(toy_schema, MFParams())
        assert len(ants[0].linguistic_variables) == 2  # x1: LOW, HIGH
        assert len(ants[1].linguistic_variables) == 3  # x2: LOW, MED, HIGH


class TestGetOutputVar:
    def test_output_var_name_and_terms(self, toy_schema):
        MFParams = build_mf_params_class(toy_schema)
        out = get_output_var(toy_schema, MFParams())
        assert out.name == 'y'
        assert len(out.linguistic_variables) == 4
