"""
fuzzyschema/variable_config.py
------------------------------
Generic schema primitives for declaring an IT2-FLS's variable structure.

This module is application-agnostic: it has no knowledge of any specific
variable names, domains, or term counts. A consuming application defines
its own concrete Schema built from these primitives -- see the README for
a worked example.

TermSpec.default carries the expert-anchored T1 trapezoid for that term,
so a Schema alone is sufficient to derive an MFParams class (mf_params.py)
with the correct field names AND defaults, with no hand-written dataclass.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

# ── Shared type alias ────────────────────────────────────────────────────────

Trap = Tuple[float, float, float, float]


# ── Validation utility ───────────────────────────────────────────────────────

def check_trap(name: str, t: Trap) -> None:
    """Raise ValueError if trapezoid violates a <= b <= c <= d."""
    a, b, c, d = t
    if not (a <= b <= c <= d):
        raise ValueError(
            f"{name}: trapezoid {t} violates a \u2264 b \u2264 c \u2264 d"
        )


# ── Spec dataclasses ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TermSpec:
    """
    One linguistic term within a fuzzy variable.

    label   : string used as the ex_fuzzy FS/IVFS name and in rule authoring.
    field   : field name to use in the generated MFParams dataclass (T1
              trapezoid). IT2 field names are derived as field + '_umf'
              and field + '_lmf'.
    default : expert-anchored (a, b, c, d) T1 trapezoid used as the default
              value for `field` in the generated MFParams class. Optional
              so existing call sites that only pass (label, field) keep
              working; a schema intended for MFParams generation must set it.
    """
    label: str
    field: str
    default: Optional[Trap] = None


@dataclass(frozen=True)
class VariableSpec:
    """
    Complete specification for one fuzzy variable.

    name   : variable name — the ex_fuzzy fuzzyVariable name, and (in a
             concrete application) typically also a DataFrame column name.
    domain : (min, max) — passed to ex_fuzzy as list(domain).
    terms  : ordered tuple of TermSpec. Order defines the integer index
             used to reference terms in rule antecedents.
    """
    name:   str
    domain: Tuple[float, float]
    terms:  Tuple[TermSpec, ...]


@dataclass(frozen=True)
class Schema:
    """
    A complete variable schema for one FLS (or one stage of a hierarchical
    FLS): its antecedents and its single output variable.

    all_vars is antecedents + (output,), matching the ALL_VARS convention
    used by IT2 builders and make_it2_from_t1.
    """
    antecedents: Tuple[VariableSpec, ...]
    output:      VariableSpec

    @property
    def all_vars(self) -> Tuple[VariableSpec, ...]:
        return (*self.antecedents, self.output)

    @property
    def input_var_names(self) -> list:
        return [v.name for v in self.antecedents]

    def field_domains(self) -> dict:
        """Map each term's field name to its owning variable's (min, max) domain."""
        return {term.field: var.domain for var in self.all_vars for term in var.terms}
