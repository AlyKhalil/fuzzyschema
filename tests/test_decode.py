"""
tests/test_decode.py
--------------------
Tests for decode.rules_to_terms. Synthetic schemas only (toy_schema:
antecedents x1 [LOW, HIGH] and x2 [LOW, MED, HIGH], output y [T1..T4]).
"""

import pytest
from ex_fuzzy.rules import RuleSimple

from fuzzyschema.decode import rules_to_terms
from fuzzyschema.rules import DONT_CARE, RuleFactory


class TestRulesToTerms:

    def test_labels_resolve_to_the_right_schema_position(self, toy_schema):
        # x1=HIGH (index 1), x2=MED (index 1), out=T3 (index 2). Both
        # antecedents use index 1 deliberately: if the decoder ever indexed
        # into the wrong variable's label list, x1 would come back 'MED' (a
        # label x1 does not even have) rather than 'HIGH'.
        rules = [RuleSimple([1, 1], consequent=2)]

        assert rules_to_terms(rules, toy_schema) == [
            {'antecedents': {'x1': 'HIGH', 'x2': 'MED'}, 'consequent': 'T3'}
        ]

    def test_every_term_index_maps_to_its_own_label(self, toy_schema):
        # Walk every (x1, x2) combination and check the decoded label against
        # the schema's own TermSpec list -- catches an off-by-one or a
        # reversed label list, which a single hand-picked rule might not.
        rules = [
            RuleSimple([i, j], consequent=0)
            for i in range(2)
            for j in range(3)
        ]
        decoded = rules_to_terms(rules, toy_schema)

        x1_terms, x2_terms = toy_schema.antecedents[0].terms, toy_schema.antecedents[1].terms
        expected = [
            {'x1': x1_terms[i].label, 'x2': x2_terms[j].label}
            for i in range(2)
            for j in range(3)
        ]
        assert [d['antecedents'] for d in decoded] == expected

    def test_every_consequent_index_maps_to_its_own_output_label(self, toy_schema):
        rules = [RuleSimple([0, 0], consequent=c) for c in range(4)]
        decoded = rules_to_terms(rules, toy_schema)

        assert [d['consequent'] for d in decoded] == ['T1', 'T2', 'T3', 'T4']

    def test_dont_care_decodes_to_none_not_the_last_term(self, toy_schema):
        # The whole point of this function over ga.rules_to_readable: a
        # DONT_CARE (-1) must not read back as the last term via negative
        # indexing. x2's last term is 'HIGH', so a regression here shows up
        # as 'HIGH' rather than None.
        rules = [RuleSimple([0, DONT_CARE], consequent=1)]

        decoded = rules_to_terms(rules, toy_schema)

        assert decoded[0]['antecedents'] == {'x1': 'LOW', 'x2': None}
        assert decoded[0]['antecedents']['x2'] is None

    def test_unconstrained_antecedents_still_appear_as_keys(self, toy_schema):
        # An all-DONT_CARE rule still lists every schema antecedent -- the
        # caller can rely on the key set being the schema's, always.
        rules = [RuleSimple([DONT_CARE, DONT_CARE], consequent=3)]

        decoded = rules_to_terms(rules, toy_schema)

        assert decoded[0] == {
            'antecedents': {'x1': None, 'x2': None},
            'consequent': 'T4',
        }

    def test_decodes_a_rulefactory_expert_rule_base(self, toy_schema):
        # RuleFactory defaults unspecified antecedents to DONT_CARE, so this
        # is the realistic shape of an expert rule base -- the case that
        # motivated this function existing.
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, out=0), rf.rule(x2=2, out=3)]

        decoded = rules_to_terms(rules, toy_schema)

        assert decoded == [
            {'antecedents': {'x1': 'LOW', 'x2': None}, 'consequent': 'T1'},
            {'antecedents': {'x1': None, 'x2': 'HIGH'}, 'consequent': 'T4'},
        ]

    def test_rule_order_is_preserved(self, toy_schema):
        rules = [RuleSimple([1, 2], consequent=3), RuleSimple([0, 0], consequent=0)]

        decoded = rules_to_terms(rules, toy_schema)

        assert [d['consequent'] for d in decoded] == ['T4', 'T1']

    def test_empty_rule_base_decodes_to_empty_list(self, toy_schema):
        assert rules_to_terms([], toy_schema) == []

    def test_works_on_a_differently_shaped_schema(self, single_antecedent_schema):
        # Same decoder, a schema with a different antecedent count/term counts
        # -- the library-level property that no schema shape is baked in.
        rules = [RuleSimple([1], consequent=0)]

        assert rules_to_terms(rules, single_antecedent_schema) == [
            {'antecedents': {'only_input': 'HIGH'}, 'consequent': 'LOW'}
        ]

    # ── raise paths ──────────────────────────────────────────────────────────

    def test_out_of_range_antecedent_index_raises(self, toy_schema):
        # x1 has 2 terms; index 2 is out of range. Must raise rather than
        # IndexError-crash or silently mislabel.
        rules = [RuleSimple([2, 0], consequent=0)]

        with pytest.raises(ValueError, match="out of range"):
            rules_to_terms(rules, toy_schema)

    def test_out_of_range_consequent_index_raises(self, toy_schema):
        rules = [RuleSimple([0, 0], consequent=4)]  # y has 4 terms: 0..3

        with pytest.raises(ValueError, match="consequent index 4"):
            rules_to_terms(rules, toy_schema)

    def test_wrong_antecedent_length_raises(self, toy_schema):
        # A rule base built for a 3-antecedent schema, decoded against a
        # 2-antecedent one: caught explicitly, not read as a short/long row.
        rules = [RuleSimple([0, 0, 0], consequent=0)]

        with pytest.raises(ValueError, match="antecedent length 3 != 2"):
            rules_to_terms(rules, toy_schema)

    def test_negative_index_below_dont_care_raises(self, toy_schema):
        # -2 is not DONT_CARE and not a valid term. Without the explicit
        # range check this would negative-index to the second-to-last label.
        rules = [RuleSimple([0, -2], consequent=0)]

        with pytest.raises(ValueError, match="out of range"):
            rules_to_terms(rules, toy_schema)
