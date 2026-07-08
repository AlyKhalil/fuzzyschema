import dataclasses

import numpy as np
import pytest

from fuzzyschema.chromosome import (
    RuleChromosomeCodec, mf_chromosome_bounds,
    build_combined_chromosome, split_combined_chromosome,
)
from fuzzyschema.mf_params import build_mf_params_class
from fuzzyschema.mf_params_t2 import build_it2_mf_params_class
from fuzzyschema.rules import RuleFactory


class TestCodecShape:
    def test_chrom_len_is_product_of_term_counts(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        assert codec.n_terms == [2, 3]
        assert codec.chrom_len == 6  # 2 * 3

    def test_n_consequents_matches_output_terms(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        assert codec.n_consequents == 4

    def test_single_antecedent_schema(self, single_antecedent_schema):
        codec = RuleChromosomeCodec(single_antecedent_schema)
        assert codec.n_terms == [2]
        assert codec.chrom_len == 2


class TestGeneIndex:
    def test_exhaustive_round_trip_is_bijective(self, toy_schema):
        """Every (a0, a1) combination must map to a unique index in
        [0, chrom_len), covering the full range exactly once. This is the
        stride-math correctness check the uneven [2, 3] shape is meant to
        stress -- a uniform term-count schema (e.g. 3 everywhere) could
        mask a stride bug that only shows up with uneven sizes."""
        codec = RuleChromosomeCodec(toy_schema)
        seen = set()
        for a0 in range(2):
            for a1 in range(3):
                idx = codec.gene_index([a0, a1])
                assert 0 <= idx < codec.chrom_len
                seen.add(idx)
        assert seen == set(range(codec.chrom_len))

    def test_known_indices(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        # idx = a0 * n_terms[1] + a1 = a0 * 3 + a1
        assert codec.gene_index([0, 0]) == 0
        assert codec.gene_index([0, 2]) == 2
        assert codec.gene_index([1, 0]) == 3
        assert codec.gene_index([1, 2]) == 5

    def test_single_antecedent(self, single_antecedent_schema):
        codec = RuleChromosomeCodec(single_antecedent_schema)
        assert codec.gene_index([0]) == 0
        assert codec.gene_index([1]) == 1


class TestDecode:
    def test_active_and_inactive_genes(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        chrom = np.zeros(6)
        chrom[codec.gene_index([0, 0])] = 1  # active, consequent 0
        chrom[codec.gene_index([1, 2])] = 4  # active, consequent 3
        # all other genes stay 0 -> disabled

        rules = codec.decode(chrom)
        decoded = {(tuple(int(x) for x in r.antecedents), int(r.consequent)) for r in rules}
        assert decoded == {((0, 0), 0), ((1, 2), 3)}

    def test_all_zero_chromosome_decodes_to_no_rules(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        assert codec.decode(np.zeros(6)) == []

    def test_float_genes_are_rounded(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        chrom = np.zeros(6)
        chrom[codec.gene_index([0, 1])] = 2.6  # should round to 3 -> consequent 2
        rules = codec.decode(chrom)
        assert len(rules) == 1
        assert int(rules[0].consequent) == 2

    def test_out_of_range_values_clipped(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        chrom = np.zeros(6)
        chrom[0] = 99  # way above n_consequents (4) -> clipped to 4 -> consequent 3
        rules = codec.decode(chrom)
        assert int(rules[0].consequent) == 3

    def test_wrong_length_raises(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        with pytest.raises(ValueError, match="Expected"):
            codec.decode(np.zeros(5))


class TestExpertChromosome:
    def test_dont_care_expands_over_all_positions(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        rf = RuleFactory(toy_schema)
        rule = rf.rule(x2=1, out=2)  # x1 = DONT_CARE -> expands over x1 in {0, 1}

        chrom = codec.expert_chromosome(lambda: [rule])
        assert chrom[codec.gene_index([0, 1])] == 3  # consequent 2 + 1
        assert chrom[codec.gene_index([1, 1])] == 3
        # Cells with x2 != 1 must remain untouched.
        assert chrom[codec.gene_index([0, 0])] == 0

    def test_more_specific_rule_wins_on_overlap(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        rf = RuleFactory(toy_schema)
        general = rf.rule(x2=1, out=2)          # specificity 1, covers (0,1) and (1,1)
        specific = rf.rule(x1=0, x2=1, out=3)   # specificity 2, covers only (0,1)

        chrom = codec.expert_chromosome(lambda: [general, specific])
        # (0, 1): both rules apply -- the more specific one (out=3) must win.
        assert chrom[codec.gene_index([0, 1])] == 4  # consequent 3 + 1
        # (1, 1): only the general rule applies.
        assert chrom[codec.gene_index([1, 1])] == 3  # consequent 2 + 1

    def test_output_is_valid_input_to_decode(self, toy_schema):
        """expert_chromosome's output must itself round-trip through decode --
        this is what run_ga actually does when seeding the initial population."""
        codec = RuleChromosomeCodec(toy_schema)
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, x2=0, out=0), rf.rule(x1=1, x2=2, out=3)]

        chrom = codec.expert_chromosome(lambda: rules)
        decoded = codec.decode(chrom)
        decoded_pairs = {(tuple(int(x) for x in r.antecedents), int(r.consequent)) for r in decoded}
        assert decoded_pairs == {((0, 0), 0), ((1, 2), 3)}


class TestBounds:
    def test_bounds_shape_and_range(self, toy_schema):
        codec = RuleChromosomeCodec(toy_schema)
        lower, upper = codec.bounds()
        assert len(lower) == len(upper) == 6
        assert (lower == 0).all()
        assert (upper == 4).all()  # n_consequents


class TestMFChromosomeBounds:
    def test_t1_shape_and_values(self, toy_schema):
        T1 = build_mf_params_class(toy_schema)
        lower, upper = mf_chromosome_bounds(T1, toy_schema)
        n = len(dataclasses.fields(T1))
        assert lower.shape == upper.shape == (n * 4,)
        # x1's two fields (x1_low, x1_high) are declared first; domain (0,10).
        assert (lower[:8] == 0.0).all()
        assert (upper[:8] == 10.0).all()

    def test_t1_fields_of_same_variable_share_bounds(self, toy_schema):
        T1 = build_mf_params_class(toy_schema)
        lower, upper = mf_chromosome_bounds(T1, toy_schema)
        fields = [f.name for f in dataclasses.fields(T1)]
        # x2 has 3 terms (x2_low, x2_med, x2_high), domain (0,1) -- all three
        # fields' 4-blocks must carry the same bounds.
        x2_idxs = [i for i, name in enumerate(fields) if name.startswith('x2_')]
        for i in x2_idxs:
            assert (lower[i * 4:(i + 1) * 4] == 0.0).all()
            assert (upper[i * 4:(i + 1) * 4] == 1.0).all()

    def test_it2_shape_double_t1_and_suffix_stripped_correctly(self, toy_schema):
        IT2 = build_it2_mf_params_class(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        lower_it2, upper_it2 = mf_chromosome_bounds(IT2, toy_schema)
        lower_t1, upper_t1 = mf_chromosome_bounds(T1, toy_schema)
        assert lower_it2.shape == upper_it2.shape == (len(lower_t1) * 2,)

        # Every umf/lmf field pair for the same term must share bounds with
        # that term's T1 field (suffix stripped correctly before lookup).
        fields_it2 = [f.name for f in dataclasses.fields(IT2)]
        idx = fields_it2.index('x1_high_umf')
        assert lower_it2[idx * 4] == 0.0 and upper_it2[idx * 4] == 10.0
        idx = fields_it2.index('x1_high_lmf')
        assert lower_it2[idx * 4] == 0.0 and upper_it2[idx * 4] == 10.0

    def test_mismatched_schema_raises(self, toy_schema, single_antecedent_schema):
        # mf_params_cls generated from one schema, bounds requested against
        # a different, incompatible schema -- field names won't match.
        T1 = build_mf_params_class(toy_schema)
        with pytest.raises(ValueError, match="not found in schema.field_domains"):
            mf_chromosome_bounds(T1, single_antecedent_schema)


class TestCombinedChromosome:
    def test_build_concatenates_rule_then_mf_block(self):
        # Layout contract: rule block first, mf block second. This is what
        # ga.py and any seeding code must agree on -- verify it explicitly
        # rather than relying on it being "obviously" the concat order.
        rule_chrom = np.array([1.0, 2.0, 3.0])
        mf_chrom = np.array([0.1, 0.2])
        combined = build_combined_chromosome(rule_chrom, mf_chrom)
        assert list(combined) == [1.0, 2.0, 3.0, 0.1, 0.2]

    def test_build_casts_to_float(self):
        # rule chromosomes are conventionally int-valued gene indices;
        # mf chromosomes are always float. The combined array must be a
        # single consistent dtype (float) since pymoo operates on one
        # array per individual -- a mixed-dtype array would silently
        # truncate the mf block's fractional values if built the wrong way.
        combined = build_combined_chromosome(np.array([1, 2, 3]), np.array([0.5, 1.5]))
        assert combined.dtype == np.float64

    def test_split_is_exact_inverse_of_build(self):
        rule_chrom = np.array([1.0, 0.0, 3.0, 2.0])
        mf_chrom = np.array([0.1, 0.2, 0.3])
        combined = build_combined_chromosome(rule_chrom, mf_chrom)
        rule_part, mf_part = split_combined_chromosome(
            combined, rule_len=len(rule_chrom), mf_len=len(mf_chrom)
        )
        assert np.array_equal(rule_part, rule_chrom)
        assert np.array_equal(mf_part, mf_chrom)

    def test_wrong_length_raises(self):
        # A silent mis-slice here would corrupt both the decoded rule base
        # and the decoded MF params with no obvious symptom at the call
        # site -- this must fail loudly, not just return misaligned slices.
        combined = np.zeros(10)
        with pytest.raises(ValueError, match="expected length"):
            split_combined_chromosome(combined, rule_len=6, mf_len=6)  # 12 != 10

    def test_round_trip_with_real_codec_and_mf_bounds_lengths(self, toy_schema):
        # End-to-end sanity check using the real lengths a GA run would
        # actually use: RuleChromosomeCodec.chrom_len for the rule block,
        # and mf_chromosome_bounds' length for the mf block.
        codec = RuleChromosomeCodec(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        lower, _ = mf_chromosome_bounds(T1, toy_schema)

        rule_chrom = np.ones(codec.chrom_len)
        mf_chrom = np.arange(len(lower), dtype=float)
        combined = build_combined_chromosome(rule_chrom, mf_chrom)

        rule_part, mf_part = split_combined_chromosome(
            combined, rule_len=codec.chrom_len, mf_len=len(lower)
        )
        assert np.array_equal(rule_part, rule_chrom)
        assert np.array_equal(mf_part, mf_chrom)
        # And the rule part must still be a valid input to decode().
        codec.decode(rule_part)
