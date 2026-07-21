"""
fuzzyschema/mf_params.py
------------------------
Schema-driven T1 membership-function parameter class + ex_fuzzy variable
builders. Application-agnostic: takes a Schema (variable_config.Schema)
and produces a dataclass with one Trap field per (variable, term) pair,
defaults taken from each TermSpec.default.

The generated class's methods (__post_init__, from_vector) are written
once, generically, and attached to every generated class — they are not
duplicated per schema.
"""

from __future__ import annotations

import dataclasses
from typing import Type

import numpy as np
from ex_fuzzy.fuzzy_sets import FS, fuzzyVariable

from fuzzyschema.variable_config import Schema, Trap, check_trap


def _post_init(self) -> None:
    for f in dataclasses.fields(self):
        check_trap(f"{type(self).__name__}.{f.name}", getattr(self, f.name)) # getattr(x, 'y', def) equivalent to x.y; 
        # can be used to access attributes and methods of an object, if attr 'y' does not exist the given default 'def'
        # is returned; if no 'def' raises AttributeError


def _from_vector(cls, v: np.ndarray) -> "MFParams":
    """
    Decode a flat GA chromosome (n_fields x 4 floats) into an instance of cls.

    Each block of 4 values is sorted ascending before construction, repairing
    any ordering violation introduced by mutation without rejecting the
    chromosome. Chromosome layout: fields in declaration order, 4 floats each.
    """
    fields = [f.name for f in dataclasses.fields(cls)]
    n = len(fields)
    if len(v) != n * 4:
        raise ValueError(f"Expected {n * 4} floats, got {len(v)}")
    
    chunks = [tuple(sorted(v[i * 4:(i + 1) * 4])) for i in range(n)]

    return cls(**dict(zip(fields, chunks)))


def _to_vector(self) -> np.ndarray:
    """
    Encode this instance as a flat chromosome vector, in the same field
    order and layout _from_vector expects. Since __post_init__ already
    guarantees every field's 4 values are ascending, this is a plain
    concatenation -- no sorting needed, and from_vector(to_vector(self))
    always reconstructs an equal instance for T1 (no interleave to invert).
    """
    fields = [f.name for f in dataclasses.fields(self)]
    return np.concatenate([np.asarray(getattr(self, f), dtype=float) for f in fields])


def build_mf_params_class(schema: Schema, class_name: str = "MFParams") -> Type:
    """
    Generate a dataclass with one Trap field per (variable, term) pair in
    `schema`, defaulted from each TermSpec.default.

    Every TermSpec used in `schema` must have `default` set, or construction
    of the generated class will raise (make_dataclass requires an explicit
    default here since fields are built programmatically, not hand-typed).
    """
    fields = []
    for var in schema.all_vars:
        for term in var.terms:
            if term.default is None:
                raise ValueError(
                    f"TermSpec '{var.name}.{term.label}' (field={term.field!r}) "
                    f"has no default trapezoid; cannot build {class_name}."
                )
            
            fields.append(
                (term.field, Trap, dataclasses.field(default=term.default))
            )

    cls = dataclasses.make_dataclass(
        class_name,
        fields,
        namespace={
            "__post_init__": _post_init, # checks default Trapezoid validity
            "from_vector": classmethod(_from_vector),
            "to_vector": _to_vector,
        },
    )

    return cls


# ── Generic T1 variable builders ─────────────────────────────────────────────
# Take the schema explicitly rather than reading a module-level global, so
# the same functions serve any number of schemas/stages.

def get_antecedents(schema: Schema, params) -> list:
    """Build the ordered list of ex_fuzzy antecedent fuzzyVariables from params."""
    result = []
    for var in schema.antecedents:
        fsets = [
            FS(term.label, list(getattr(params, term.field)), domain=list(var.domain))
            for term in var.terms
        ]

        result.append(fuzzyVariable(var.name, fsets))

    return result


def get_output_var(schema: Schema, params) -> fuzzyVariable:
    """Build the ex_fuzzy output fuzzyVariable from params."""
    fsets = [
        FS(term.label, list(getattr(params, term.field)), domain=list(schema.output.domain))
        for term in schema.output.terms
    ]

    return fuzzyVariable(schema.output.name, fsets)
