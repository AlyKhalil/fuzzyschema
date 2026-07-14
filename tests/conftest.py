"""
conftest.py
------------
Shared fixtures for fuzzyschema's own test suite.

TOY_SCHEMA is deliberately NOT the same shape as any known consumer
(e.g. the dissertation's 4-antecedent, uniform-3-term schema): it has
2 antecedents with UNEVEN term counts (2 and 3) and a 4-term output.
This is the actual point of testing the library itself, rather than
only testing consumer schemas that happen to exercise one shape --
uneven term counts are exactly where stride/indexing math (chromosome.py)
and pairing logic (mf_params_t2.py) are most likely to have an
off-by-one that a uniform 3-term-everywhere schema would never surface.
"""

import itertools

import pytest
from ex_fuzzy.rules import RuleSimple

from fuzzyschema.mf_params import build_mf_params_class
from fuzzyschema.mf_params_t2 import build_it2_mf_params_class, make_it2_from_t1
from fuzzyschema.unit import FLSUnit
from fuzzyschema.variable_config import Schema, VariableSpec, TermSpec


@pytest.fixture
def toy_schema() -> Schema:
    return Schema(
        antecedents=(
            VariableSpec(
                name="x1",
                domain=(0.0, 10.0),
                terms=(
                    TermSpec("LOW",  "x1_low",  default=(-0.1, 0.0, 4.0, 6.0)),
                    TermSpec("HIGH", "x1_high", default=(4.0, 6.0, 10.0, 10.0)),
                ),
            ),
            VariableSpec(
                name="x2",
                domain=(0.0, 1.0),
                terms=(
                    TermSpec("LOW",  "x2_low",  default=(-0.01, 0.0, 0.2, 0.4)),
                    TermSpec("MED",  "x2_med",  default=(0.2, 0.4, 0.6, 0.8)),
                    TermSpec("HIGH", "x2_high", default=(0.6, 0.8, 1.0, 1.0)),
                ),
            ),
        ),
        output=VariableSpec(
            name="y",
            domain=(0.0, 5.0),
            terms=(
                TermSpec("T1", "y_t1", default=(-0.1, 0.0, 1.0, 1.5)),
                TermSpec("T2", "y_t2", default=(1.0, 1.5, 2.0, 2.5)),
                TermSpec("T3", "y_t3", default=(2.0, 2.5, 3.5, 4.0)),
                TermSpec("T4", "y_t4", default=(3.5, 4.0, 5.0, 5.0)),
            ),
        ),
    )


@pytest.fixture
def single_antecedent_schema() -> Schema:
    """Smallest possible non-trivial schema: 1 antecedent, 2 terms."""
    return Schema(
        antecedents=(
            VariableSpec(
                name="only_input",
                domain=(0.0, 1.0),
                terms=(
                    TermSpec("LOW",  "in_low",  default=(-0.01, 0.0, 0.3, 0.5)),
                    TermSpec("HIGH", "in_high", default=(0.5, 0.7, 1.0, 1.0)),
                ),
            ),
        ),
        output=VariableSpec(
            name="only_output",
            domain=(0.0, 1.0),
            terms=(
                TermSpec("LOW",  "out_low",  default=(-0.01, 0.0, 0.4, 0.6)),
                TermSpec("HIGH", "out_high", default=(0.4, 0.6, 1.0, 1.0)),
            ),
        ),
    )


# ── FLSUnit / HierarchicalFLS fixtures ───────────────────────────────────────
#
# The chain schemas below are built so that a broken column_map cannot pass by
# accident. Unit B's antecedent that consumes unit A's output is named 'prev',
# NOT 'a_out' -- so the mapping {'prev': 'a_out'} is genuinely a rename. Had B's
# antecedent simply been called 'a_out', an implementation that ignored
# column_map entirely and matched columns by antecedent name would still pass
# every chaining test. Same for unit C's 'prev2' <- 'b_out'.


def _two_term_var(name: str, domain=(0.0, 1.0)) -> VariableSpec:
    """A 0-1 variable with LOW/HIGH terms that leave a GAP over roughly
    (0.35, 0.65): no term has any membership there. That gap is what makes a
    zero-firing (NaN) row reachable on demand -- see gapped_schema."""
    return VariableSpec(
        name=name,
        domain=domain,
        terms=(
            TermSpec("LOW",  f"{name}_low",  default=(0.0, 0.0, 0.2, 0.35)),
            TermSpec("HIGH", f"{name}_high", default=(0.65, 0.8, 1.0, 1.0)),
        ),
    )


def make_unit(name, schema, rules_fn, engine_type='t1', fallback_fn=None) -> FLSUnit:
    """Build an FLSUnit for `schema` with default params derived from the
    schema's own TermSpec defaults (T1), converted to IT2 when engine_type is
    't2'. Keeps every test from repeating the params-construction boilerplate."""
    t1_cls = build_mf_params_class(schema)
    t1_params = t1_cls()

    if engine_type == 't2':
        it2_cls = build_it2_mf_params_class(schema)
        params = make_it2_from_t1(schema, t1_params, 0.05, it2_cls)
        mf_cls = it2_cls
    else:
        params, mf_cls = t1_params, t1_cls

    return FLSUnit(
        name=name,
        schema=schema,
        engine_type=engine_type,
        mf_params_cls=mf_cls,
        default_params=params,
        default_rules_fn=rules_fn,
        fallback_fn=fallback_fn,
    )


@pytest.fixture
def schema_a() -> Schema:
    """Unit A: raw inputs in1, in2 -> output column 'a_out'."""
    return Schema(
        antecedents=(_two_term_var("in1"), _two_term_var("in2")),
        output=_two_term_var("a_out"),
    )


@pytest.fixture
def schema_b() -> Schema:
    """Unit B: antecedent 'prev' (fed from A's 'a_out' column) + raw 'in3'
    -> output column 'b_out'."""
    return Schema(
        antecedents=(_two_term_var("prev"), _two_term_var("in3")),
        output=_two_term_var("b_out"),
    )


@pytest.fixture
def schema_c() -> Schema:
    """Unit C: antecedent 'prev2' (fed from B's 'b_out' column) -> 'c_out'."""
    return Schema(
        antecedents=(_two_term_var("prev2"),),
        output=_two_term_var("c_out"),
    )


def dense_rules_fn(n_ants: int, consequent_of):
    """A rule base covering every antecedent-term combination (so no row can
    zero-fire), with each combination's consequent chosen by `consequent_of`.
    Returns a FRESH list on every call, as an FLSUnit's default_rules_fn must."""

    def rules_fn():
        return [
            RuleSimple(list(combo), consequent=consequent_of(combo))
            for combo in itertools.product(*[range(2)] * n_ants)
        ]

    return rules_fn


@pytest.fixture
def gapped_schema() -> Schema:
    """Single antecedent whose LOW/HIGH terms leave an uncovered gap: an input
    of ~0.5 fires no rule at all, which is how the NaN/fallback paths are
    exercised deliberately rather than by accident."""
    return Schema(
        antecedents=(_two_term_var("x"),),
        output=_two_term_var("y"),
    )
