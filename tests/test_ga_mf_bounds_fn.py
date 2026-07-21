"""
tests/test_ga_mf_bounds_fn.py
-----------------------------
Can run_ga optimise an MF parameterisation whose chromosome is NOT
n_fields * 4 genes long?

Until mf_bounds_fn existed it could not, and the reason was structural rather
than incidental. run_ga derived the MF block's bounds -- and therefore its
length -- from exactly one place, mf_chromosome_bounds, which emits four floats
per dataclass field (chromosome.py: `lower.extend([dom_min] * 4)`). So mf_len
was always a multiple of 4: one gene per trapezoid breakpoint. That is the right
default for the raw-breakpoint parameterisation, and the wrong one for any codec
that optimises something else -- e.g. a constrained per-term shift + scale plus a
per-variable FOU-width multiplier, whose chromosome is 17 + 17 + 5 = 39 genes.
39 % 4 == 3, so no naming of the adapter's fields could ever have produced it.

Two properties matter here, and they pull in opposite directions:

  1. omitting mf_bounds_fn must change NOTHING for existing callers, and
  2. supplying it must free mf_len from the multiple-of-4 shape entirely.

test_ga.py's TestMFOptimize already covers the default path end-to-end and must
keep passing untouched; this module pins (1) as an explicit equality and proves
(2) with a deliberately non-multiple-of-4 MF block.
"""

import dataclasses
import json
import os

import numpy as np
import pytest

from fuzzyschema.chromosome import (
    RuleChromosomeCodec, mf_chromosome_bounds, split_combined_chromosome,
)
from fuzzyschema.ga import run_ga
from fuzzyschema.mf_params import build_mf_params_class


def _combined_fitness(codec, rule_len, mf_len):
    """Deterministic, engine-free fitness over both blocks -- mirrors
    test_ga.py's helper of the same name."""
    def _fitness(chrom: np.ndarray) -> float:
        rule_part, mf_part = split_combined_chromosome(chrom, rule_len, mf_len)
        return float(len(codec.decode(rule_part))) + 0.001 * float(mf_part[0])
    return _fitness


# ── 1. Omitting mf_bounds_fn is a no-op ──────────────────────────────────────

class TestDefaultPathUnchanged:
    def test_omitting_mf_bounds_fn_matches_passing_the_default_explicitly(
        self, toy_schema, tmp_path,
    ):
        # Two runs at the same seed: one leaving mf_bounds_fn at its default, one
        # passing mf_chromosome_bounds by hand. Identical results are what makes
        # "no behavioural change for existing callers" a proof rather than a
        # claim -- the GA is fully deterministic given seed, so any divergence in
        # bounds, chromosome length, or sampling would show up here immediately.
        codec = RuleChromosomeCodec(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        mf_lower, _ = mf_chromosome_bounds(T1, toy_schema)
        fitness = _combined_fitness(codec, codec.chrom_len, len(mf_lower))

        common = dict(
            schema=toy_schema, fitness_fn=fitness, mf_params_cls=T1,
            pop_size=6, n_gen=3, seed=7,
            run_dir_base=str(tmp_path), verbose=False,
        )
        default = run_ga(run_name='default_bounds', **common)
        explicit = run_ga(
            run_name='explicit_bounds', mf_bounds_fn=mf_chromosome_bounds, **common,
        )

        assert default['best_chromosome'] == pytest.approx(explicit['best_chromosome'])
        assert default['best_score'] == explicit['best_score']
        assert default['history'] == explicit['history']

    def test_default_run_records_no_custom_bounds_fn(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        mf_lower, _ = mf_chromosome_bounds(T1, toy_schema)

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_combined_fitness(codec, codec.chrom_len, len(mf_lower)),
            mf_params_cls=T1,
            pop_size=6, n_gen=2,
            run_dir_base=str(tmp_path), run_name='default_bounds_config',
            verbose=False,
        )

        with open(os.path.join(result['run_dir'], 'ga_config.json')) as f:
            config = json.load(f)
        assert config['custom_mf_bounds_fn'] is False
        assert config['mf_chromosome_len'] == len(mf_lower)


# ── 2. A custom bounds fn frees mf_len from the n_fields * 4 shape ───────────

# Deliberately not a multiple of 4. The real motivating case is 39 genes
# (17 shift + 17 scale + 5 delta_scale); 7 is the same property in miniature and
# keeps the test fast. If mf_len were still being derived from a dataclass field
# count, no value of this constant could make the run work.
STUB_MF_LEN = 7

_STUB_LOWER = np.array([-1.0] * STUB_MF_LEN)
_STUB_UPPER = np.array([2.0] * STUB_MF_LEN)


@dataclasses.dataclass
class _StubTunedParams:
    """What the stub codec's from_vector hands back.

    ga.py's _mf_params_to_readable does `list(getattr(p, f.name))` over
    dataclasses.fields(p), so from_vector's OUTPUT must be a dataclass whose
    field values are iterable -- note this constrains the output, not the codec
    class itself. A real adapter would return an actual IT2MFParams here (decoded
    trapezoids), which satisfies that for free and makes ga_best_mf_params.json
    inspectable in the same shape as the default path's.
    """
    genes: tuple


class _StubCodec:
    """The minimal interface run_ga demands of an mf_params_cls once
    mf_chromosome_bounds is bypassed: a from_vector classmethod, and nothing
    else. It is not a dataclass, and none of its names resolve against the
    schema -- both of which the default bounds path would have required."""

    @classmethod
    def from_vector(cls, v: np.ndarray) -> _StubTunedParams:
        assert len(v) == STUB_MF_LEN, f"expected {STUB_MF_LEN} genes, got {len(v)}"
        return _StubTunedParams(genes=tuple(float(x) for x in v))


def _stub_bounds_fn(mf_params_cls, schema):
    """Bounds that owe nothing to schema.field_domains() -- shift/scale/
    delta_scale genes have their own ranges, not variable domains."""
    return _STUB_LOWER, _STUB_UPPER


class _StubSeed:
    """mf_seed_fn's return value is only ever asked for .to_vector(); run_ga
    never type-checks it against mf_params_cls."""

    def to_vector(self) -> np.ndarray:
        return np.zeros(STUB_MF_LEN)


class TestCustomBoundsFn:
    def test_allows_non_multiple_of_4_mf_block(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_combined_fitness(codec, codec.chrom_len, STUB_MF_LEN),
            mf_params_cls=_StubCodec,
            mf_bounds_fn=_stub_bounds_fn,
            pop_size=6, n_gen=3, seed=1,
            run_dir_base=str(tmp_path), run_name='stub_bounds',
            verbose=False,
        )

        assert STUB_MF_LEN % 4 != 0  # the whole point: unreachable by the default
        assert len(result['best_chromosome']) == codec.chrom_len + STUB_MF_LEN
        assert isinstance(result['best_mf_params'], _StubTunedParams)
        assert len(result['best_mf_params'].genes) == STUB_MF_LEN

    def test_mf_genes_stay_within_the_custom_bounds(self, toy_schema, tmp_path):
        # pymoo's box constraints come from the (lower, upper) run_ga concatenates,
        # so a custom bounds fn must actually reach the optimiser -- not just size
        # the block. If mf_bounds_fn were used for length but the default bounds
        # for xl/xu, the genes would drift outside [-1, 2] and this would fail.
        codec = RuleChromosomeCodec(toy_schema)

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_combined_fitness(codec, codec.chrom_len, STUB_MF_LEN),
            mf_params_cls=_StubCodec,
            mf_bounds_fn=_stub_bounds_fn,
            pop_size=8, n_gen=3, seed=3,
            run_dir_base=str(tmp_path), run_name='stub_bounds_range',
            verbose=False,
        )

        mf_part = result['best_chromosome'][codec.chrom_len:]
        assert (mf_part >= _STUB_LOWER - 1e-9).all()
        assert (mf_part <= _STUB_UPPER + 1e-9).all()

    def test_seeding_and_artefacts_work_on_the_custom_block(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_combined_fitness(codec, codec.chrom_len, STUB_MF_LEN),
            mf_params_cls=_StubCodec,
            mf_bounds_fn=_stub_bounds_fn,
            mf_seed_fn=_StubSeed,
            pop_size=6, n_gen=2, seed=5,
            run_dir_base=str(tmp_path), run_name='stub_bounds_seeded',
            verbose=False,
        )

        with open(os.path.join(result['run_dir'], 'ga_config.json')) as f:
            config = json.load(f)
        assert config['custom_mf_bounds_fn'] is True
        assert config['mf_chromosome_len'] == STUB_MF_LEN
        assert config['chromosome_len'] == codec.chrom_len + STUB_MF_LEN

        # _save_artefacts round-trips the MF block through from_vector and
        # _mf_params_to_readable -- the path most likely to have assumed a
        # trapezoid-shaped params object.
        with open(os.path.join(result['run_dir'], 'ga_best_mf_params.json')) as f:
            saved = json.load(f)
        assert len(saved['genes']) == STUB_MF_LEN


# ── 3. The argument is never silently ignored ────────────────────────────────

class TestGuard:
    def test_mf_bounds_fn_without_mf_params_cls_raises(self, toy_schema):
        # Mirrors the existing mf_seed_fn guard: an argument that cannot take
        # effect is a caller error, not something to quietly drop.
        codec = RuleChromosomeCodec(toy_schema)

        with pytest.raises(ValueError, match="mf_bounds_fn"):
            run_ga(
                schema=toy_schema,
                fitness_fn=lambda c: float(len(codec.decode(c))),
                mf_bounds_fn=_stub_bounds_fn,
                pop_size=4, n_gen=1,
                verbose=False,
            )
