"""
fuzzyschema/decode.py
---------------------
Pure schema decode of a rule base into linguistic term labels.

This module deliberately stops at *labels*: it produces the structured,
schema-resolved form of a rule base and nothing else -- no prose, no
templating, no LLM call. A consuming application that wants a natural-language
explanation wraps this (see the app repo's explain.py); the wrapper is where
prompt/phrasing decisions belong, not here.

Relationship to ga.rules_to_readable
------------------------------------
ga.rules_to_readable serves a different purpose (the JSON artefact written by a
GA run) and is only ever handed codec.decode() output, whose antecedents are
fully specified by construction. It indexes labels[int(ants[i])] directly, so a
DONT_CARE (-1) would silently read back as the *last* term via Python's negative
indexing. rules_to_terms is the decoder to use for any rule base that may carry
DONT_CARE -- an expert rule base built via RuleFactory always can.
"""

from __future__ import annotations

from typing import List, Optional

from fuzzyschema.rules import DONT_CARE
from fuzzyschema.variable_config import Schema


def rules_to_terms(rules: list, schema: Schema) -> List[dict]:
    """
    Decode a rule base into linguistic term labels, resolved against `schema`.

    Returns one dict per rule, in the order the rules were given:

        [
          {'antecedents': {'x1': 'LOW', 'x2': None}, 'consequent': 'T3'},
          {'antecedents': {'x1': 'HIGH', 'x2': 'MED'}, 'consequent': 'T1'},
        ]

    Every antecedent variable in the schema appears as a key for every rule,
    including the ones the rule does not constrain. A DONT_CARE antecedent
    decodes to ``None`` -- an explicit "this variable is unconstrained", left
    for the caller to phrase ("any", "*", omitted entirely...) rather than
    given a label here.

    The nested 'antecedents' map (rather than one flat dict per rule) keeps an
    antecedent named 'consequent' from colliding with the consequent key.

    Raises ValueError if a rule's antecedent length disagrees with the schema,
    or if any term index is out of range for its variable -- an out-of-range
    index is a schema/rule-base mismatch, and silently mislabelling it is the
    exact failure this function exists to avoid.
    """
    ant_labels = [[t.label for t in v.terms] for v in schema.antecedents]
    out_labels = [t.label for t in schema.output.terms]
    n_ants = len(schema.antecedents)

    decoded = []
    for i, rule in enumerate(rules):
        ants = rule.antecedents
        if len(ants) != n_ants:
            raise ValueError(
                f"Rule {i}: antecedent length {len(ants)} != {n_ants} "
                f"(schema has antecedents {schema.input_var_names})"
            )

        terms: dict = {}
        for pos in range(n_ants):
            idx = int(ants[pos])
            var = schema.antecedents[pos]
            if idx == DONT_CARE:
                label: Optional[str] = None
            elif 0 <= idx < len(ant_labels[pos]):
                label = ant_labels[pos][idx]
            else:
                raise ValueError(
                    f"Rule {i}, position {pos} ({var.name}): term index {idx} "
                    f"out of range for {len(ant_labels[pos])} terms "
                    f"({', '.join(ant_labels[pos])})"
                )
            terms[var.name] = label

        c = int(rule.consequent)
        if not (0 <= c < len(out_labels)):
            raise ValueError(
                f"Rule {i}: consequent index {c} out of range for "
                f"{len(out_labels)} output terms ({', '.join(out_labels)})"
            )

        decoded.append({'antecedents': terms, 'consequent': out_labels[c]})

    return decoded
