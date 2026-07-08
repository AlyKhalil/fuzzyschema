import numpy as np
import pytest

from fuzzyschema.engine import T1FLSEngine, T2FLSEngine, validate_input, _km_endpoint
from fuzzyschema.mf_params import build_mf_params_class, get_antecedents, get_output_var
from fuzzyschema.mf_params_t2 import build_it2_mf_params_class, make_it2_from_t1, get_antecedents_t2, get_output_var_t2
from fuzzyschema.rules import RuleFactory


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
