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

import pytest

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
