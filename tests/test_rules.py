import pytest

from fuzzyschema.rules import DONT_CARE, RuleFactory, build_term_index, validate_rules


class TestBuildTermIndex:
    def test_labels_map_to_position(self, toy_schema):
        idx = build_term_index(toy_schema.antecedents[1])  # x2: LOW, MED, HIGH
        assert idx == {'LOW': 0, 'MED': 1, 'HIGH': 2}


class TestRuleFactory:
    def test_unspecified_antecedents_default_to_dont_care(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rule = rf.rule(x1=0, out=0)  # x2 unspecified
        assert list(rule.antecedents) == [0, DONT_CARE]

    def test_all_antecedents_specified(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rule = rf.rule(x1=1, x2=2, out=3)
        assert list(rule.antecedents) == [1, 2]
        assert rule.consequent == 3

    def test_unknown_variable_name_raises(self, toy_schema):
        rf = RuleFactory(toy_schema)
        with pytest.raises(ValueError, match="Unknown antecedent"):
            rf.rule(nonexistent=0, out=0)

    def test_single_antecedent_schema(self, single_antecedent_schema):
        rf = RuleFactory(single_antecedent_schema)
        rule = rf.rule(only_input=1, out=0)
        assert list(rule.antecedents) == [1]

    def test_term_idx_and_output_idx_derived_correctly(self, toy_schema):
        rf = RuleFactory(toy_schema)
        assert rf.term_idx['x1'] == {'LOW': 0, 'HIGH': 1}
        assert rf.output_idx == {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}


class TestValidateRules:
    def test_valid_rule_base_passes(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, x2=0, out=0), rf.rule(x1=1, x2=2, out=3)]
        validate_rules(rules, toy_schema)  # should not raise

    def test_wrong_antecedent_length_raises(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rule = rf.rule(x1=0, out=0)
        rule.antecedents = rule.antecedents[:1]  # truncate to wrong length
        with pytest.raises(ValueError, match="antecedent length"):
            validate_rules([rule], toy_schema)

    def test_out_of_range_antecedent_index_raises(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rule = rf.rule(x2=5, out=0)  # x2 only has 3 terms (0,1,2)
        with pytest.raises(ValueError, match="not in"):
            validate_rules([rule], toy_schema)

    def test_out_of_range_consequent_raises(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rule = rf.rule(x1=0, out=99)  # output only has 4 terms
        with pytest.raises(ValueError, match="consequent"):
            validate_rules([rule], toy_schema)

    def test_duplicate_antecedents_raise(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, x2=0, out=0), rf.rule(x1=0, x2=0, out=1)]
        with pytest.raises(ValueError, match="duplicate"):
            validate_rules(rules, toy_schema)

    def test_dont_care_positions_do_not_count_as_duplicates(self, toy_schema):
        rf = RuleFactory(toy_schema)
        # (0, DONT_CARE) and (0, 1) are different antecedent tuples -- not duplicates.
        rules = [rf.rule(x1=0, out=0), rf.rule(x1=0, x2=1, out=1)]
        validate_rules(rules, toy_schema)  # should not raise
