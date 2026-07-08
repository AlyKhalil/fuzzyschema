import pytest

from fuzzyschema.variable_config import Schema, VariableSpec, TermSpec, check_trap


class TestCheckTrap:
    def test_valid_trap_passes(self):
        check_trap("x", (0.0, 1.0, 2.0, 3.0))  # should not raise

    def test_equal_breakpoints_pass(self):
        check_trap("x", (0.0, 0.0, 1.0, 1.0))  # should not raise

    def test_violates_a_le_b(self):
        with pytest.raises(ValueError, match="violates"):
            check_trap("x", (1.0, 0.0, 2.0, 3.0))

    def test_violates_c_le_d(self):
        with pytest.raises(ValueError, match="violates"):
            check_trap("x", (0.0, 1.0, 3.0, 2.0))

    def test_error_message_includes_name(self):
        with pytest.raises(ValueError, match="my_field"):
            check_trap("my_field", (1.0, 0.0, 2.0, 3.0))


class TestTermSpec:
    def test_default_is_optional(self):
        t = TermSpec("LOW", "x_low")
        assert t.default is None

    def test_frozen(self):
        t = TermSpec("LOW", "x_low")
        with pytest.raises(Exception):
            t.label = "HIGH"


class TestSchemaAllVars:
    def test_all_vars_uneven_terms(self, toy_schema):
        names = [v.name for v in toy_schema.all_vars]
        assert names == ["x1", "x2", "y"]

    def test_all_vars_single_antecedent(self, single_antecedent_schema):
        names = [v.name for v in single_antecedent_schema.all_vars]
        assert names == ["only_input", "only_output"]

    def test_all_vars_is_antecedents_plus_output(self, toy_schema):
        assert toy_schema.all_vars[:-1] == toy_schema.antecedents
        assert toy_schema.all_vars[-1] is toy_schema.output


class TestSchemaInputVarNames:
    def test_matches_antecedent_order(self, toy_schema):
        assert toy_schema.input_var_names == ["x1", "x2"]

    def test_single_antecedent(self, single_antecedent_schema):
        assert single_antecedent_schema.input_var_names == ["only_input"]

    def test_excludes_output(self, toy_schema):
        assert "y" not in toy_schema.input_var_names


class TestSchemaFrozen:
    def test_schema_is_frozen(self, toy_schema):
        with pytest.raises(Exception):
            toy_schema.output = toy_schema.antecedents[0]


class TestSchemaFieldDomains:
    def test_all_fields_present(self, toy_schema):
        fd = toy_schema.field_domains()
        expected_fields = {
            'x1_low', 'x1_high', 'x2_low', 'x2_med', 'x2_high',
            'y_t1', 'y_t2', 'y_t3', 'y_t4',
        }
        assert set(fd.keys()) == expected_fields

    def test_fields_of_same_variable_share_domain(self, toy_schema):
        fd = toy_schema.field_domains()
        assert fd['x2_low'] == fd['x2_med'] == fd['x2_high'] == (0.0, 1.0)

    def test_domains_match_variable_spec(self, toy_schema):
        fd = toy_schema.field_domains()
        assert fd['x1_low'] == (0.0, 10.0)
        assert fd['y_t1'] == (0.0, 5.0)

    def test_single_antecedent_schema(self, single_antecedent_schema):
        fd = single_antecedent_schema.field_domains()
        assert fd['in_low'] == fd['in_high'] == (0.0, 1.0)
        assert fd['out_low'] == (0.0, 1.0)
