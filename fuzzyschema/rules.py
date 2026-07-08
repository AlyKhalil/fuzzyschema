"""
fuzzyschema/rules.py
--------------------
Schema-driven rule-authoring helpers: term-index derivation, a DONT_CARE-
defaulting rule factory, and rule-base validation. Application-agnostic --
the actual rule *content* (which term combinations fire, and what they mean)
always belongs in the consuming application, not here.
"""

from __future__ import annotations

import numpy as np
from ex_fuzzy.rules import RuleSimple

from fuzzyschema.variable_config import Schema, VariableSpec

DONT_CARE: int = -1


def build_term_index(var: VariableSpec) -> dict:
    """label -> integer index within var.terms."""
    return {t.label: i for i, t in enumerate(var.terms)}


class RuleFactory:
    """
    Builds RuleSimple objects by antecedent variable name + output term
    index, defaulting unspecified antecedents to DONT_CARE.

    Usage:
        rf = RuleFactory(schema)
        rf.rule(lidar_conf=L_LOW, camera_conf=C_LOW, out=O_LOW)

    Keyword names must match schema antecedent VariableSpec.name exactly.
    A consuming application typically wraps `.rule()` in a local `_rule()`
    helper with its own preferred (possibly abbreviated) parameter names --
    that mapping is application-specific and belongs in the consumer, not here.
    """

    def __init__(self, schema: Schema):
        self.schema = schema
        self._pos = {v.name: i for i, v in enumerate(schema.antecedents)}
        self.term_idx = {v.name: build_term_index(v) for v in schema.antecedents}
        self.output_idx = build_term_index(schema.output)

    def rule(self, *, out: int, **kwargs) -> RuleSimple:
        ants = [DONT_CARE] * len(self.schema.antecedents)
        for var_name, idx in kwargs.items():
            if var_name not in self._pos:
                raise ValueError(
                    f"Unknown antecedent variable '{var_name}'. "
                    f"Valid names: {sorted(self._pos)}"
                )
            ants[self._pos[var_name]] = idx
        return RuleSimple(np.array(ants), consequent=out)


def validate_rules(rules: list, schema: Schema) -> None:
    """
    Validate a rule base against `schema`.

    Checks:
      - Antecedent array length matches len(schema.antecedents).
      - All antecedent indices are DONT_CARE or in valid range for that
        variable's term count.
      - All consequent indices are in range for schema.output.terms.
      - No duplicate antecedent arrays.

    Raises ValueError with a descriptive message on the first violation found.
    """
    antecedents = schema.antecedents
    n_ants = len(antecedents)
    valid_ranges = [set(range(len(var.terms))) for var in antecedents]
    n_output = len(schema.output.terms)
    seen: set = set()

    for i, rule in enumerate(rules):
        ants = rule.antecedents

        if len(ants) != n_ants:
            raise ValueError(f"Rule {i}: antecedent length {len(ants)} != {n_ants}")

        for pos, idx in enumerate(ants):
            idx = int(idx)
            if idx != DONT_CARE and idx not in valid_ranges[pos]:
                raise ValueError(
                    f"Rule {i}, position {pos} ({antecedents[pos].name}): "
                    f"index {idx} not in {sorted(valid_ranges[pos])}"
                )

        c = int(rule.consequent)
        if not (0 <= c < n_output):
            raise ValueError(f"Rule {i}: consequent {c} not in [0, {n_output})")

        key = tuple(int(x) for x in ants)
        if key in seen:
            raise ValueError(f"Rule {i}: duplicate antecedents {key}")
        seen.add(key)
