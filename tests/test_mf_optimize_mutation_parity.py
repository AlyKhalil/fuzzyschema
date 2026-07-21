"""
tests/test_mf_optimize_mutation_parity.py
-----------------------------------------
Does the rule block mutate at the same per-gene rate whether or not an MF block
is bolted onto the chromosome?

`tests/test_ga.py::TestMakeMutation` already guards the *raw* per-gene rate and
is correct as far as it goes -- but it measures `~np.isclose(Xp, X)` on a
population of non-integer floats, so a 2.0 -> 2.03 perturbation counts as a
mutation. The rule base never sees that mutation: `RuleChromosomeCodec.decode`
np.round()s every rule gene, and 2.03 rounds straight back to 2. The raw rate is
therefore an upper bound on the rate that reaches the phenotype, and a
regression could halve the *decoded* rate while leaving the raw rate untouched.

This module measures the DECODED rate -- the one that decides whether a rule can
actually be pruned -- and asserts parity between rule-only and combined
(--mf-optimize) mode. It also pins the raw-vs-decoded gap as characterised
behaviour, and guards the two properties the parity argument rests on: that the
initial population's rule genes are drawn identically in both modes, and that
the GA's per-generation rule_counts log agrees with codec.decode().

Everything here is seeded and runs in seconds. No GA search is used to measure a
mutation rate.
"""

import json
import os

import numpy as np
import pytest
from pymoo.core.population import Population
from pymoo.core.problem import Problem

from fuzzyschema.chromosome import (
    RuleChromosomeCodec, mf_chromosome_bounds, split_combined_chromosome,
)
from fuzzyschema.ga import _make_callback, _make_mutation, _make_sampling, run_ga
from fuzzyschema.mf_params import build_mf_params_class
from fuzzyschema.mf_params_t2 import build_it2_mf_params_class, make_it2_from_t1
from fuzzyschema.variable_config import Schema, TermSpec, VariableSpec


# ── Synthetic schema ─────────────────────────────────────────────────────────
#
# Shape-matched to the chromosome a real --mf-optimize run carries (81 rule
# genes, 136 MF genes), and -- the part that actually matters here -- matched on
# n_consequents too.
#
# n_consequents is not cosmetic: it sets the rule genes' upper bound, hence the
# gene range a polynomial-mutation deltaq has to traverse to cross a 0.5 rounding
# boundary, hence the decoded mutation rate this module measures. A schema with
# the right gene *count* but the wrong n_consequents would report a different
# decoded rate and quietly invalidate every number below.
#
# The content is deliberately generic (x0..x3 / y, LOW-MED-HIGH / A..E): the
# library must be testable with zero knowledge of any consuming application.

def _var(name: str, term_names: tuple, lo: float = 0.0, hi: float = 1.0) -> VariableSpec:
    step = (hi - lo) / len(term_names)
    return VariableSpec(
        name=name,
        domain=(lo, hi),
        terms=tuple(
            TermSpec(t, f"{name}_{t.lower()}", default=(
                lo + i * step,
                lo + i * step + 0.1 * step,
                lo + (i + 1) * step - 0.1 * step,
                lo + (i + 1) * step,
            ))
            for i, t in enumerate(term_names)
        ),
    )


@pytest.fixture
def app_shaped_schema() -> Schema:
    """4 antecedents x 3 terms = 81 rule genes; 5 output terms => n_consequents=5;
    17 terms x 2 (umf/lmf) x 4 breakpoints = 136 IT2 MF genes."""
    return Schema(
        antecedents=tuple(_var(f"x{i}", ("LOW", "MED", "HIGH")) for i in range(4)),
        output=_var("y", ("A", "B", "C", "D", "E")),
    )


RULE_LEN, MF_LEN, N_CONSEQUENTS = 81, 136, 5
ETA = 20.0


def test_synthetic_schema_reproduces_the_real_chromosome_shape(app_shaped_schema):
    """If this drifts, every rate below is measuring the wrong search space."""
    codec = RuleChromosomeCodec(app_shaped_schema)
    it2_cls = build_it2_mf_params_class(app_shaped_schema)
    mf_lower, _ = mf_chromosome_bounds(it2_cls, app_shaped_schema)

    assert codec.chrom_len == RULE_LEN
    assert codec.n_consequents == N_CONSEQUENTS
    assert len(mf_lower) == MF_LEN


# ── Measurement helpers ──────────────────────────────────────────────────────

def _problem(mf_len: int) -> Problem:
    """The bounds layout run_ga builds: rule genes first, bounded
    [0, n_consequents]; MF genes second."""
    xl = np.concatenate([np.zeros(RULE_LEN), np.zeros(mf_len)])
    xu = np.concatenate([np.full(RULE_LEN, float(N_CONSEQUENTS)), np.full(mf_len, 1.0)])

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=RULE_LEN + mf_len, n_obj=1, xl=xl, xu=xu)

    return _P()


def _dense_rule_population(mf_len: int, n_ind: int = 30, seed: int = 7) -> np.ndarray:
    """A population shaped like the one the GA actually converges onto: rule genes
    are exact integers in [1, n_consequents] (every rule active, as in the expert
    seed), MF genes interior to their bounds.

    Integer-valued rule genes are the whole point -- measuring on the mid-domain
    *floats* that test_ga.py uses would hide the rounding behaviour under test.
    """
    rng = np.random.default_rng(seed)
    rule = rng.integers(1, N_CONSEQUENTS + 1, size=(n_ind, RULE_LEN)).astype(float)
    if not mf_len:
        return rule
    return np.concatenate([rule, rng.uniform(0.3, 0.7, (n_ind, mf_len))], axis=1)


def _decode_genes(rule_block: np.ndarray) -> np.ndarray:
    """The rule genes as codec.decode() sees them: rounded, clipped. 0 == rule off."""
    return np.clip(np.round(rule_block), 0, N_CONSEQUENTS).astype(int)


def _measure(mf_len: int, trials: int = 300, seed: int = 0) -> dict:
    """Drive the REAL operator through Mutation.do() -- the path the GA calls, and
    the only one that applies the per-individual gate -- and count, over the rule
    block only:

      raw     genes whose float value moved at all
      decoded genes whose value moved *and survived np.round* (the rule base differs)
      drop    genes whose decoded value became 0 (a rule left the active set)
    """
    X = _dense_rule_population(mf_len)
    problem = _problem(mf_len)
    op = _make_mutation(RULE_LEN, mf_len, eta=ETA)
    rng = np.random.default_rng(seed)

    before_raw = X[:, :RULE_LEN]
    before_dec = _decode_genes(before_raw)

    raw = decoded = drop = 0
    for _ in range(trials):
        Xp = op.do(problem, Population.new(X=X.copy()), random_state=rng).get('X')
        after_raw = Xp[:, :RULE_LEN]
        after_dec = _decode_genes(after_raw)

        raw += int((~np.isclose(after_raw, before_raw)).sum())
        decoded += int((after_dec != before_dec).sum())
        drop += int(((after_dec == 0) & (before_dec != 0)).sum())

    return {'raw': raw, 'decoded': decoded, 'drop': drop,
            'n': X.shape[0] * RULE_LEN * trials}


def _diff_ci(a: dict, b: dict, key: str, z: float = 1.96):
    """95% CI on (rate_a - rate_b) for a two-proportion comparison. If it spans 0
    the two modes' rates are statistically consistent."""
    pa, pb = a[key] / a['n'], b[key] / b['n']
    se = np.sqrt(pa * (1 - pa) / a['n'] + pb * (1 - pb) / b['n'])
    d = pa - pb
    return d, d - z * se, d + z * se


# ── The parity assertions ────────────────────────────────────────────────────

class TestRuleBlockMutationParity:
    """The audit question: is the rule block mutated as hard under --mf-optimize
    as it is on its own?"""

    def test_decoded_rule_mutation_rate_at_parity_across_modes(self):
        """The assertion the whole audit turns on.

        Fails if the rule slice, prob_var, or per-block rate regresses -- e.g. if
        the two blocks ever share one 1/n_var rate again, the rule block would
        fall from 1/81 to 1/217 and its decoded rate with it (~0.0012 -> ~0.0004),
        which the interval below is far too tight to absorb.
        """
        rule_only = _measure(mf_len=0)
        combined = _measure(mf_len=MF_LEN)

        diff, lo, hi = _diff_ci(rule_only, combined, 'decoded')
        assert lo <= 0.0 <= hi, (
            f"decoded rule-gene mutation rate differs between modes: "
            f"rule-only={rule_only['decoded'] / rule_only['n']:.6f} vs "
            f"combined={combined['decoded'] / combined['n']:.6f} "
            f"(diff {diff:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}])"
        )

    def test_raw_rule_mutation_rate_at_parity_across_modes(self):
        rule_only = _measure(mf_len=0)
        combined = _measure(mf_len=MF_LEN)

        diff, lo, hi = _diff_ci(rule_only, combined, 'raw')
        assert lo <= 0.0 <= hi, (
            f"raw rule-gene mutation rate differs between modes "
            f"(diff {diff:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}])"
        )

    def test_raw_rate_tracks_the_nominal_one_over_rule_len_in_both_modes(self):
        """Both modes should sit near the nominal 1/rule_len, not at 1/n_var. A
        shared-rate regression drags the combined mode's raw rate to ~1/217."""
        nominal = 1.0 / RULE_LEN
        shared = 1.0 / (RULE_LEN + MF_LEN)

        for mf_len in (0, MF_LEN):
            m = _measure(mf_len=mf_len)
            rate = m['raw'] / m['n']
            assert rate == pytest.approx(nominal, rel=0.15)
            assert rate > 1.5 * shared

    def test_decoded_rate_is_an_order_of_magnitude_below_raw(self):
        """Rounding suppression, pinned as *characterised* behaviour rather than a
        latent surprise: polynomial mutation with eta=20 makes small perturbations,
        and codec.decode's np.round sends most of them straight back to the gene's
        original integer. Roughly 9 in 10 rule-gene mutations never reach the rule
        base at all.

        This is a property of the rule block only -- MF genes are continuous, so
        nothing is discarded at decode. It is an expressivity statement, not a
        claim about fitness impact.

        A band, not a point: the exact ratio depends on eta and n_consequents, and
        pinning it precisely would make this a change-detector rather than a test.
        """
        for mf_len in (0, MF_LEN):
            m = _measure(mf_len=mf_len)
            raw_rate = m['raw'] / m['n']
            decoded_rate = m['decoded'] / m['n']

            assert 0.05 * raw_rate <= decoded_rate <= 0.25 * raw_rate, (
                f"raw={raw_rate:.6f} decoded={decoded_rate:.6f} "
                f"(ratio {decoded_rate / raw_rate:.3f}, expected ~0.1)"
            )
            # And a rule DROP -- the only way the rule count can fall -- is rarer
            # still, since it needs a gene sitting at 1 to be pushed below 0.5.
            assert m['drop'] < m['decoded']


class TestInitialPopulationParity:
    """If combined-mode sampling narrowed rule-gene diversity, less rule movement
    downstream would have nothing to do with the mutation operator."""

    def test_initial_rule_genes_identical_across_modes(self, app_shaped_schema):
        """_make_sampling draws the rule block from the seeded RNG *before* it
        draws the MF block, so adding an MF block cannot perturb the rule genes.
        Guarding the draw order matters: swapping the two statements would leave
        every existing test passing while silently changing the rule genes every
        --mf-optimize run starts from.
        """
        codec = RuleChromosomeCodec(app_shaped_schema)
        it2_cls = build_it2_mf_params_class(app_shaped_schema)
        mf_lower, mf_upper = mf_chromosome_bounds(it2_cls, app_shaped_schema)
        t1_params = build_mf_params_class(app_shaped_schema)()

        pop_size, seed = 30, 42
        # No expert rule seed: individual 0 would otherwise be a constant in both
        # modes and could mask a divergence in the randomly-drawn individuals.
        rules_fn = list

        sampling_rule_only = _make_sampling(
            codec, rules_fn, None, None, None, None, pop_size, seed,
        )
        sampling_combined = _make_sampling(
            codec, rules_fn, it2_cls,
            lambda: make_it2_from_t1(app_shaped_schema, t1_params, 0.05, it2_cls),
            mf_lower, mf_upper, pop_size, seed,
        )

        X_rule = sampling_rule_only._do(None, pop_size)
        X_comb = sampling_combined._do(None, pop_size)

        assert X_rule.shape == (pop_size, RULE_LEN)
        assert X_comb.shape == (pop_size, RULE_LEN + MF_LEN)
        np.testing.assert_array_equal(X_rule, X_comb[:, :RULE_LEN])

        # ...and the diversity that actually matters is the decoded rule count,
        # which must therefore be spread, not constant (a degenerate sampler that
        # returned the same chromosome to both modes would pass the check above).
        counts = [len(codec.decode(X_comb[i, :RULE_LEN])) for i in range(pop_size)]
        assert min(counts) < max(counts)


class TestPerIndividualRuleCountLogging:
    """_GALogger records a rule count for every individual, every generation --
    without which 'the best individual sat at 81 rules for 50 generations' cannot
    be told apart from 'no pruning mutation was ever produced'."""

    def test_logged_rule_counts_match_independent_decode(self, app_shaped_schema):
        """The assertion that catches the logger drifting from the real decode
        logic. The population below is built so the sloppy implementations fail:

          - genes at 0.49 and 0.51 straddle the rounding boundary, so a logger
            counting raw `!= 0` instead of np.round-ed values gets both wrong;
          - rule counts differ per individual, so a logger that decoded only the
            best and broadcast it gets all but one wrong;
          - the best individual is not index 0, so `rule_counts[0]` is not n_rules.
        """
        codec = RuleChromosomeCodec(app_shaped_schema)
        it2_cls = build_it2_mf_params_class(app_shaped_schema)
        mf_lower, _ = mf_chromosome_bounds(it2_cls, app_shaped_schema)
        mf_len = len(mf_lower)

        rule_blocks = np.array([
            np.zeros(RULE_LEN),                     # 0 rules
            np.full(RULE_LEN, 0.49),                # rounds to 0 -> 0 rules
            np.full(RULE_LEN, 0.51),                # rounds to 1 -> 81 rules
            np.full(RULE_LEN, 3.0),                 # 81 rules
            np.concatenate([np.zeros(40), np.full(RULE_LEN - 40, 2.0)]),  # 41 rules
        ])
        X = np.concatenate(
            [rule_blocks, np.full((len(rule_blocks), mf_len), 0.5)], axis=1,
        )
        # Minimised objective: index 3 is the best (most negative).
        F = np.array([[-1.0], [-2.0], [-3.0], [-9.0], [-4.0]])
        best_idx = 3

        class _StubAlgorithm:
            def __init__(self):
                self.pop = Population.new(X=X, F=F)
                self.n_gen = 1

        logger = _make_callback(codec, RULE_LEN, mf_len, verbose=False)
        logger.notify(_StubAlgorithm())

        entry = logger.history[0]
        expected = [
            len(codec.decode(split_combined_chromosome(x, RULE_LEN, mf_len)[0]))
            for x in X
        ]

        assert entry['rule_counts'] == expected
        assert expected == [0, 0, RULE_LEN, RULE_LEN, RULE_LEN - 40]
        assert entry['n_rules'] == entry['rule_counts'][best_idx]

    def test_history_carries_a_rule_count_per_individual_every_generation(
        self, app_shaped_schema, tmp_path,
    ):
        """End-to-end through run_ga: the artefact on disk has the data, and keeps
        the invariant that history stays chromosome-free (ints only, no genes)."""
        codec = RuleChromosomeCodec(app_shaped_schema)
        pop_size, n_gen = 8, 3

        result = run_ga(
            schema=app_shaped_schema,
            fitness_fn=lambda chrom: float(len(codec.decode(chrom))),
            pop_size=pop_size,
            n_gen=n_gen,
            seed=123,
            run_dir_base=str(tmp_path),
            run_name='rule_counts',
            verbose=False,
        )

        with open(os.path.join(result['run_dir'], 'ga_history.json')) as f:
            history = json.load(f)

        assert len(history) == n_gen
        for entry in history:
            assert len(entry['rule_counts']) == pop_size
            assert all(isinstance(c, int) for c in entry['rule_counts'])
            assert all(0 <= c <= RULE_LEN for c in entry['rule_counts'])
            # The logged best must be one of the individuals actually present.
            assert entry['n_rules'] in entry['rule_counts']
