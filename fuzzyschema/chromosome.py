"""
fuzzyschema/chromosome.py
-------------------------
Schema-driven rule-space chromosome codec: encodes/decodes a fixed-length
integer chromosome over the full antecedent-term rule space (one gene per
fully-specified antecedent combination), independent of any application's
rule content or fitness function.

Gene value convention:
  0      -> rule disabled (not included in the decoded rule base)
  1 .. N -> rule active, consequent = value - 1  (N = number of output terms)

This is application-agnostic: it only needs a Schema and (for seeding) a
callable that returns the current expert rule base as RuleSimple objects.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Callable, Type

import numpy as np
from ex_fuzzy.rules import RuleSimple

from fuzzyschema.rules import DONT_CARE
from fuzzyschema.variable_config import Schema


class RuleChromosomeCodec:
    """
    Encode/decode fully-specified rule-space chromosomes for one Schema.

    chrom_len = product of len(var.terms) for var in schema.antecedents.
    n_consequents = len(schema.output.terms).
    """

    def __init__(self, schema: Schema):
        self.schema = schema
        self.n_terms: list = [len(v.terms) for v in schema.antecedents]
        self.n_consequents: int = len(schema.output.terms)

        chrom_len = 1
        for n in self.n_terms:
            chrom_len *= n
        self.chrom_len: int = chrom_len

    # ── gene indexing ─────────────────────────────────────────────────────

    def gene_index(self, ants: list) -> int:
        """
        Map a fully-specified antecedent combination to its gene position.

        No DONT_CARE values allowed; all values must be in [0, n_terms[i]).
        Formula: index = sum(ants[i] * prod(n_terms[i+1:]))
        """
        idx = 0
        stride = 1
        for i in reversed(range(len(ants))):
            idx += ants[i] * stride
            stride *= self.n_terms[i]
        return idx

    # ── decode ────────────────────────────────────────────────────────────

    def _decode_rules(self, rule_genes: np.ndarray) -> list:
        rules = []
        for combo in itertools.product(*[range(n) for n in self.n_terms]):
            idx = self.gene_index(list(combo))
            gene = int(rule_genes[idx])
            if gene == 0:
                continue
            consequent = gene - 1
            rules.append(RuleSimple(np.array(list(combo)), consequent=consequent))
        return rules

    def decode(self, chrom: np.ndarray) -> list:
        """
        Decode a chrom_len-element chromosome into active RuleSimple objects.

        Gene values are rounded to the nearest integer and clipped to
        [0, n_consequents]. GA libraries typically operate in float space;
        rounding is applied internally.
        """
        if len(chrom) != self.chrom_len:
            raise ValueError(
                f"Expected chromosome of length {self.chrom_len} (one gene per "
                f"antecedent combination), got {len(chrom)}"
            )
        rule_genes = np.clip(
            np.round(np.asarray(chrom, dtype=float)).astype(int),
            0,
            self.n_consequents,
        )
        return self._decode_rules(rule_genes)

    # ── seed chromosome ──────────────────────────────────────────────────

    def expert_chromosome(self, rules_fn: Callable[[], list]) -> np.ndarray:
        """
        Build a seed chromosome from the expert rule base returned by
        `rules_fn()`. DONT_CARE antecedents are expanded over all term
        values for that position. When two expert rules expand into the
        same cell, the more specific rule (fewer DONT_CARE positions) wins;
        ties are broken by last occurrence in rules_fn().
        """
        rule_genes = np.zeros(self.chrom_len, dtype=int)
        winning_spec = np.full(self.chrom_len, -1, dtype=int)

        for rule in rules_fn():
            ants = [int(x) for x in rule.antecedents]
            consequent = int(rule.consequent)
            specificity = sum(1 for a in ants if a != DONT_CARE)

            ranges = [
                [ants[i]] if ants[i] != DONT_CARE else list(range(self.n_terms[i]))
                for i in range(len(self.schema.antecedents))
            ]
            for combo in itertools.product(*ranges):
                idx = self.gene_index(list(combo))
                if specificity > winning_spec[idx]:
                    rule_genes[idx] = consequent + 1
                    winning_spec[idx] = specificity

        return rule_genes.astype(float)

    # ── bounds ────────────────────────────────────────────────────────────

    def bounds(self):
        """Return (lower, upper) float arrays: all genes in [0, n_consequents]."""
        lower = np.zeros(self.chrom_len)
        upper = np.full(self.chrom_len, float(self.n_consequents))
        return lower, upper


# ── MF chromosome bounds ─────────────────────────────────────────────────────

def mf_chromosome_bounds(mf_params_cls: Type, schema: Schema):
    """
    Return (lower, upper) float arrays sized to mf_params_cls's chromosome
    length (n_fields * 4), giving each field's 4 floats the domain bounds of
    its owning variable, via Schema.field_domains() -- the single source of
    truth also used by make_it2_from_t1's domain clamping.

    Works for classes generated by either build_mf_params_class (T1, bare
    field names) or build_it2_mf_params_class (IT2, field names suffixed
    _umf/_lmf) -- IT2 field names are stripped of their _umf/_lmf suffix
    before the lookup, since field_domains() is keyed by the base term
    field name (one domain shared by both the umf and lmf field of a term).

    Note: from_vector's decode does not itself clip to these bounds (T1
    only sorts; IT2's clamp-chain only enforces containment/ordering, not
    the variable's domain) -- these bounds are what make a GA's search
    space respect the domain, by construction of the optimizer's box
    constraints, not by any check inside from_vector itself.
    """
    field_domain = schema.field_domains()
    lower: list = []
    upper: list = []
    for f in dataclasses.fields(mf_params_cls):
        base = f.name[:-4] if f.name.endswith(('_umf', '_lmf')) else f.name
        if base not in field_domain:
            raise ValueError(
                f"mf_chromosome_bounds: field '{f.name}' (base '{base}') not "
                f"found in schema.field_domains() -- mf_params_cls must be "
                f"generated from this same schema."
            )
        dom_min, dom_max = field_domain[base]
        lower.extend([dom_min] * 4)
        upper.extend([dom_max] * 4)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


# ── Combined (rule + MF) chromosome helpers ─────────────────────────────────
#
# When MF-optimization mode is enabled (item #6), a single GA individual
# needs to carry two logically separate blocks:
#   - a "rule block" of length RuleChromosomeCodec.chrom_len (decoded via
#     RuleChromosomeCodec.decode)
#   - an "mf block" of length n_mf_fields * 4 (decoded via
#     mf_params_cls.from_vector)
#
# pymoo (and GA libraries generally) operate on one flat float array per
# individual -- there's no native concept of "this array is actually two
# sub-chromosomes". These two helpers are the single place that defines how
# the two blocks are joined and split, so ga.py, seeding code, and any
# future caller all agree on the layout (rule block first, mf block second)
# instead of each reimplementing the concat/slice and risking a mismatch.

def build_combined_chromosome(rule_chrom: np.ndarray, mf_chrom: np.ndarray) -> np.ndarray:
    """
    Concatenate a rule-block chromosome and an mf-block chromosome into one
    flat array, in the fixed layout (rule block first, mf block second)
    that split_combined_chromosome expects.

    Used both to build a combined *seed* (e.g. expert_chromosome() output
    concatenated with an expert MFParams/IT2MFParams instance's
    to_vector() output) and, conceptually, to describe what a combined
    individual's gene layout looks like once the GA is running.
    """
    return np.concatenate([
        np.asarray(rule_chrom, dtype=float),
        np.asarray(mf_chrom, dtype=float),
    ])


def split_combined_chromosome(chrom: np.ndarray, rule_len: int, mf_len: int):
    """
    Inverse of build_combined_chromosome: slice a flat combined chromosome
    back into (rule_block, mf_block), given the two blocks' known lengths.

    rule_len is RuleChromosomeCodec.chrom_len for the schema in use;
    mf_len is len(mf_chromosome_bounds(mf_params_cls, schema)[0]) (i.e.
    n_mf_fields * 4) for the mf_params_cls in use. Both are fixed, known
    quantities once a schema and mf_params_cls are chosen -- they are
    passed in explicitly here rather than re-derived, so this function has
    no dependency on RuleChromosomeCodec or mf_params_cls itself and stays
    a plain, allocation-free array operation.

    Raises if chrom's length doesn't match rule_len + mf_len, since a
    silent mis-slice here would corrupt both the decoded rule base and the
    decoded MF params without any obvious symptom at the call site.
    """
    expected = rule_len + mf_len
    if len(chrom) != expected:
        raise ValueError(
            f"split_combined_chromosome: expected length {expected} "
            f"(rule_len={rule_len} + mf_len={mf_len}), got {len(chrom)}"
        )
    return chrom[:rule_len], chrom[rule_len:rule_len + mf_len]
