import dataclasses
import json
import os

import numpy as np
import pytest

from pymoo.core.population import Population

from fuzzyschema.chromosome import RuleChromosomeCodec, mf_chromosome_bounds
from fuzzyschema.ga import _make_mutation, run_ga, rules_to_readable
from fuzzyschema.mf_params import build_mf_params_class
from fuzzyschema.mf_params_t2 import build_it2_mf_params_class, make_it2_from_t1
from fuzzyschema.rules import RuleFactory


def _n_active_rules_fitness(codec):
    """Trivial fitness: maximise the number of active rules. Deterministic
    and cheap, but non-trivial enough that a working GA should visibly
    improve on a fully-random start within a handful of generations."""
    def _fitness(chrom: np.ndarray) -> float:
        return float(len(codec.decode(chrom)))
    return _fitness


def _combined_fitness(codec, rule_len, mf_len):
    """Trivial combined-mode fitness: rewards active rules plus a small
    bonus scaled by how far the first MF gene sits from zero -- enough to
    exercise both blocks of the chromosome without needing a real engine."""
    from fuzzyschema.chromosome import split_combined_chromosome

    def _fitness(chrom: np.ndarray) -> float:
        rule_part, mf_part = split_combined_chromosome(chrom, rule_len, mf_len)
        return float(len(codec.decode(rule_part))) + 0.001 * float(mf_part[0])
    return _fitness


class TestRunGA:
    def test_runs_and_returns_expected_keys(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)
        result = run_ga(
            schema=toy_schema,
            fitness_fn=_n_active_rules_fitness(codec),
            pop_size=8,
            n_gen=3,
            run_dir_base=str(tmp_path),
            run_name='smoke_test',
            verbose=False,
        )
        assert set(result.keys()) == {
            'best_chromosome', 'best_score', 'best_rules', 'history', 'run_dir',
            'n_failures', 'failure_rate',
        }
        assert len(result['best_chromosome']) == codec.chrom_len
        assert len(result['history']) == 3

    def test_seeding_from_rules_fn_included_in_config(self, toy_schema, tmp_path):
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, x2=0, out=0)]

        result = run_ga(
            schema=toy_schema,
            fitness_fn=lambda chrom: 0.0,
            rules_fn=lambda: rules,
            pop_size=6,
            n_gen=2,
            run_dir_base=str(tmp_path),
            run_name='seeded_test',
            verbose=False,
        )
        with open(os.path.join(result['run_dir'], 'ga_config.json')) as f:
            config = json.load(f)
        assert config['seeded_from_rules_fn'] is True

    def test_artefacts_are_written(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)
        result = run_ga(
            schema=toy_schema,
            fitness_fn=_n_active_rules_fitness(codec),
            pop_size=6,
            n_gen=2,
            run_dir_base=str(tmp_path),
            run_name='artefact_test',
            verbose=False,
        )
        run_dir = result['run_dir']
        for fname in ['ga_config.json', 'ga_best_chromosome.npy', 'ga_best_rules.json', 'ga_history.json']:
            assert os.path.exists(os.path.join(run_dir, fname)), f"missing {fname}"

    def test_final_eval_fn_is_called_with_best_chromosome_and_run_dir(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)
        calls = []

        def _final_eval(best_chromosome, run_dir):
            calls.append((best_chromosome, run_dir))

        run_ga(
            schema=toy_schema,
            fitness_fn=_n_active_rules_fitness(codec),
            pop_size=6,
            n_gen=2,
            run_dir_base=str(tmp_path),
            run_name='final_eval_test',
            verbose=False,
            final_eval_fn=_final_eval,
        )
        assert len(calls) == 1
        assert len(calls[0][0]) == codec.chrom_len

    def test_final_eval_fn_exception_does_not_abort_run(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)

        def _broken_final_eval(best_chromosome, run_dir):
            raise RuntimeError("boom")

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_n_active_rules_fitness(codec),
            pop_size=6,
            n_gen=2,
            run_dir_base=str(tmp_path),
            run_name='broken_final_eval_test',
            verbose=False,
            final_eval_fn=_broken_final_eval,
        )
        assert 'best_score' in result  # run still completed and returned normally


class TestRunGAWithMFOptimization:
    def test_mf_seed_fn_without_mf_params_cls_raises(self, toy_schema, tmp_path):
        # mf_seed_fn has nothing to seed without mf_params_cls -- must fail
        # loudly rather than silently ignoring mf_seed_fn.
        T1 = build_mf_params_class(toy_schema)
        with pytest.raises(ValueError, match="mf_seed_fn"):
            run_ga(
                schema=toy_schema,
                fitness_fn=lambda chrom: 0.0,
                mf_seed_fn=lambda: T1(),
                run_dir_base=str(tmp_path),
                verbose=False,
            )

    def test_combined_chromosome_length_and_result_keys(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        mf_lower, _ = mf_chromosome_bounds(T1, toy_schema)

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_combined_fitness(codec, codec.chrom_len, len(mf_lower)),
            mf_params_cls=T1,
            pop_size=8,
            n_gen=3,
            run_dir_base=str(tmp_path),
            run_name='mf_smoke_test',
            verbose=False,
        )
        # rule-only mode's key set plus best_mf_params, nothing else.
        assert set(result.keys()) == {
            'best_chromosome', 'best_score', 'best_rules', 'history', 'run_dir',
            'n_failures', 'failure_rate', 'best_mf_params',
        }
        assert len(result['best_chromosome']) == codec.chrom_len + len(mf_lower)
        assert isinstance(result['best_mf_params'], T1)

    def test_it2_mf_params_cls_works_too(self, toy_schema, tmp_path):
        # Symmetric with T1 -- build_it2_mf_params_class-generated classes
        # are just as valid an mf_params_cls as build_mf_params_class ones.
        codec = RuleChromosomeCodec(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        mf_lower, _ = mf_chromosome_bounds(IT2, toy_schema)

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_combined_fitness(codec, codec.chrom_len, len(mf_lower)),
            mf_params_cls=IT2,
            pop_size=8,
            n_gen=2,
            run_dir_base=str(tmp_path),
            run_name='mf_it2_smoke_test',
            verbose=False,
        )
        assert isinstance(result['best_mf_params'], IT2)
        assert len(result['best_chromosome']) == codec.chrom_len + len(mf_lower)

    def test_rule_only_mode_has_no_mf_key(self, toy_schema, tmp_path):
        # Confirms mf_params_cls=None really does reproduce the exact
        # rule-only key set -- no stray best_mf_params=None leaking in.
        codec = RuleChromosomeCodec(toy_schema)
        result = run_ga(
            schema=toy_schema,
            fitness_fn=_n_active_rules_fitness(codec),
            pop_size=6,
            n_gen=2,
            run_dir_base=str(tmp_path),
            run_name='rule_only_no_mf_key_test',
            verbose=False,
        )
        assert 'best_mf_params' not in result

    def test_mf_gene_bounds_respected_in_final_chromosome(self, toy_schema, tmp_path):
        # The MF block of the best chromosome must stay within
        # mf_chromosome_bounds -- this is enforced by pymoo's box
        # constraints (xl/xu passed to the Problem), not by any check
        # inside from_vector, so worth confirming end-to-end.
        codec = RuleChromosomeCodec(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        mf_lower, mf_upper = mf_chromosome_bounds(T1, toy_schema)

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_combined_fitness(codec, codec.chrom_len, len(mf_lower)),
            mf_params_cls=T1,
            pop_size=8,
            n_gen=3,
            run_dir_base=str(tmp_path),
            run_name='mf_bounds_test',
            verbose=False,
        )
        mf_part = result['best_chromosome'][codec.chrom_len:]
        assert (mf_part >= mf_lower - 1e-9).all()
        assert (mf_part <= mf_upper + 1e-9).all()

    def test_seeding_from_mf_seed_fn_included_in_config(self, toy_schema, tmp_path):
        T1 = build_mf_params_class(toy_schema)
        IT2 = build_it2_mf_params_class(toy_schema)
        codec = RuleChromosomeCodec(toy_schema)
        expert_it2 = make_it2_from_t1(toy_schema, T1(), delta=0.1, it2_cls=IT2)
        mf_lower, _ = mf_chromosome_bounds(IT2, toy_schema)

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_combined_fitness(codec, codec.chrom_len, len(mf_lower)),
            mf_params_cls=IT2,
            mf_seed_fn=lambda: expert_it2,
            pop_size=6,
            n_gen=2,
            run_dir_base=str(tmp_path),
            run_name='mf_seeded_test',
            verbose=False,
        )
        with open(os.path.join(result['run_dir'], 'ga_config.json')) as f:
            config = json.load(f)
        assert config['seeded_from_mf_seed_fn'] is True
        assert config['optimize_mf_params'] is True
        assert config['mf_chromosome_len'] == len(mf_lower)

    def test_mf_artefact_is_written(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)
        T1 = build_mf_params_class(toy_schema)
        mf_lower, _ = mf_chromosome_bounds(T1, toy_schema)

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_combined_fitness(codec, codec.chrom_len, len(mf_lower)),
            mf_params_cls=T1,
            pop_size=6,
            n_gen=2,
            run_dir_base=str(tmp_path),
            run_name='mf_artefact_test',
            verbose=False,
        )
        mf_path = os.path.join(result['run_dir'], 'ga_best_mf_params.json')
        assert os.path.exists(mf_path)
        with open(mf_path) as f:
            saved = json.load(f)
        # One entry per T1 field, each a 4-element trapezoid.
        assert set(saved.keys()) == {f.name for f in dataclasses.fields(T1)}
        assert all(len(v) == 4 for v in saved.values())

    def test_rule_only_mode_never_writes_mf_artefact(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)
        result = run_ga(
            schema=toy_schema,
            fitness_fn=_n_active_rules_fitness(codec),
            pop_size=6,
            n_gen=2,
            run_dir_base=str(tmp_path),
            run_name='rule_only_no_mf_artefact_test',
            verbose=False,
        )
        mf_path = os.path.join(result['run_dir'], 'ga_best_mf_params.json')
        assert not os.path.exists(mf_path)


class TestRunGAParallelism:
    def test_n_jobs_none_default_matches_explicit_none(self, toy_schema, tmp_path):
        # n_jobs=None must be indistinguishable from not passing n_jobs at
        # all -- the backward-compat guarantee for existing callers.
        codec = RuleChromosomeCodec(toy_schema)
        common = dict(
            schema=toy_schema,
            fitness_fn=_n_active_rules_fitness(codec),
            pop_size=8,
            n_gen=3,
            seed=7,
            run_dir_base=str(tmp_path),
            verbose=False,
        )
        result_default = run_ga(run_name='n_jobs_omitted', **common)
        result_explicit_none = run_ga(run_name='n_jobs_none', n_jobs=None, **common)
        assert result_default['best_score'] == result_explicit_none['best_score']
        np.testing.assert_array_equal(
            result_default['best_chromosome'], result_explicit_none['best_chromosome'],
        )

    def test_n_jobs_2_matches_serial_result(self, toy_schema, tmp_path):
        # Parallelism changes wall-clock, not the optimisation outcome --
        # same seed + deterministic fitness_fn must find the same best
        # chromosome whether evaluated serially or across 2 joblib workers.
        codec = RuleChromosomeCodec(toy_schema)
        common = dict(
            schema=toy_schema,
            fitness_fn=_n_active_rules_fitness(codec),
            pop_size=8,
            n_gen=3,
            seed=7,
            run_dir_base=str(tmp_path),
            verbose=False,
        )
        serial = run_ga(run_name='serial', n_jobs=None, **common)
        parallel = run_ga(run_name='parallel', n_jobs=2, **common)
        assert serial['best_score'] == parallel['best_score']
        np.testing.assert_array_equal(serial['best_chromosome'], parallel['best_chromosome'])


class TestRunGAErrorHandling:
    def test_fitness_fn_exception_does_not_crash_run_and_reports_failures(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)

        def _flaky_fitness(chrom: np.ndarray) -> float:
            if float(chrom[0]) >= 1.0:
                raise ValueError("bad chromosome")
            return float(len(codec.decode(chrom)))

        pop_size, n_gen = 8, 3
        result = run_ga(
            schema=toy_schema,
            fitness_fn=_flaky_fitness,
            pop_size=pop_size,
            n_gen=n_gen,
            run_dir_base=str(tmp_path),
            run_name='flaky_test',
            verbose=False,
        )
        errors_path = os.path.join(result['run_dir'], 'ga_errors.jsonl')
        assert os.path.exists(errors_path)
        with open(errors_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) > 0
        assert lines[0]['exception_type'] == 'ValueError'
        assert lines[0]['message'] == 'bad chromosome'
        assert set(lines[0].keys()) == {
            'timestamp', 'pid', 'chromosome', 'exception_type', 'message', 'traceback',
        }
        assert result['n_failures'] == len(lines)
        assert result['failure_rate'] == pytest.approx(len(lines) / (pop_size * n_gen))

    def test_always_raising_fitness_fn_triggers_smoke_test_and_raises_immediately(self, toy_schema, tmp_path):
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, x2=0, out=0)]
        calls = []

        def _always_raises(chrom: np.ndarray) -> float:
            calls.append(chrom)
            raise RuntimeError("fitness_fn is fundamentally broken")

        with pytest.raises(RuntimeError, match="fundamentally broken"):
            run_ga(
                schema=toy_schema,
                fitness_fn=_always_raises,
                rules_fn=lambda: rules,
                pop_size=8,
                n_gen=3,
                run_dir_base=str(tmp_path),
                run_name='smoke_fail_test',
                verbose=False,
            )
        # Smoke test calls fitness_fn exactly once, then raises before the
        # pymoo problem/algorithm are even built -- no generations ran.
        assert len(calls) == 1
        run_dir = os.path.join(str(tmp_path), 'smoke_fail_test')
        assert not os.path.exists(os.path.join(run_dir, 'ga_history.json'))
        assert not os.path.exists(os.path.join(run_dir, 'ga_errors.jsonl'))

    def test_smoke_test_false_skips_preflight_and_proceeds_into_ga_loop(self, toy_schema, tmp_path, capsys):
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, x2=0, out=0)]
        calls = []

        def _always_raises(chrom: np.ndarray) -> float:
            calls.append(chrom)
            raise RuntimeError("fitness_fn is fundamentally broken")

        pop_size, n_gen = 6, 2
        result = run_ga(
            schema=toy_schema,
            fitness_fn=_always_raises,
            rules_fn=lambda: rules,
            pop_size=pop_size,
            n_gen=n_gen,
            run_dir_base=str(tmp_path),
            run_name='smoke_disabled_test',
            verbose=False,
            smoke_test=False,
        )
        captured = capsys.readouterr()
        # Proceeded well past the single smoke-test-style call -- every
        # individual, every generation, was actually evaluated (and failed).
        assert len(calls) > 1
        assert result['n_failures'] == len(calls)
        assert result['failure_rate'] == pytest.approx(len(calls) / (pop_size * n_gen))
        assert 'WARNING' in captured.out
        errors_path = os.path.join(result['run_dir'], 'ga_errors.jsonl')
        assert os.path.exists(errors_path)


class TestRunGAKeyboardInterrupt:
    def test_interrupted_run_keeps_actual_best_not_seed_placeholder(self, toy_schema, tmp_path):
        # Regression test: a KeyboardInterrupt raised mid-GA must not fall
        # through to the pre-loop seed placeholder (all-zero chromosome,
        # score 0.0) -- it must recover the actual best individual found
        # across whatever generations completed before the interrupt.
        codec = RuleChromosomeCodec(toy_schema)
        pop_size, n_gen = 8, 20
        calls = {'n': 0}
        # Interrupt partway through generation 2, after generation 1 has
        # definitely completed and been recorded in the callback's history.
        raise_after = pop_size + 2

        def _interrupting_fitness(chrom: np.ndarray) -> float:
            calls['n'] += 1
            if calls['n'] > raise_after:
                raise KeyboardInterrupt()
            return float(len(codec.decode(chrom)))

        final_eval_calls = []

        def _final_eval(best_chromosome, run_dir):
            final_eval_calls.append(np.array(best_chromosome, copy=True))

        result = run_ga(
            schema=toy_schema,
            fitness_fn=_interrupting_fitness,
            pop_size=pop_size,
            n_gen=n_gen,
            run_dir_base=str(tmp_path),
            run_name='interrupted_test',
            verbose=False,
            final_eval_fn=_final_eval,
        )

        # The interrupt actually fired, and at least one generation ran.
        assert calls['n'] > raise_after
        assert len(result['history']) >= 1
        assert len(result['history']) < n_gen

        # best_score/best_chromosome must reflect the real best individual
        # found (not the seed placeholder: all-zero chromosome, score 0.0).
        assert result['best_score'] > 0.0
        assert not np.allclose(result['best_chromosome'], 0.0)

        # The fitness function counts active rules, so a correctly-recovered
        # best_chromosome must decode to exactly best_score rules.
        assert len(result['best_rules']) == result['best_score']

        # Post-interrupt code path behaves like a normal completion: final
        # evaluation runs against the recovered best, not the placeholder.
        assert len(final_eval_calls) == 1
        np.testing.assert_array_equal(final_eval_calls[0], result['best_chromosome'])

        # Artefacts on disk reflect the recovered best too.
        run_dir = result['run_dir']
        with open(os.path.join(run_dir, 'ga_best_rules.json')) as f:
            saved = json.load(f)
        assert saved['score'] == result['best_score']
        assert saved['n_rules'] == len(result['best_rules'])
        with open(os.path.join(run_dir, 'ga_history.json')) as f:
            saved_history = json.load(f)
        assert len(saved_history) == len(result['history'])


class TestRulesToReadable:
    def test_labels_and_consequent(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=0, x2=2, out=1)]
        readable = rules_to_readable(toy_schema, rules)
        assert readable == [{'x1': 'LOW', 'x2': 'HIGH', 'consequent': 'T2'}]

    def test_sorted_deterministically(self, toy_schema):
        rf = RuleFactory(toy_schema)
        rules = [rf.rule(x1=1, x2=0, out=0), rf.rule(x1=0, x2=0, out=1)]
        readable = rules_to_readable(toy_schema, rules)
        assert readable[0]['x1'] == 'HIGH'  # 'HIGH' < 'LOW' alphabetically
        assert readable[1]['x1'] == 'LOW'
        assert len(readable) == 2


# ── Mutation operator ─────────────────────────────────────────────────────────

# The dissertation's real chromosome shape (4 antecedents x 3 terms = 81 rule
# genes; 34 MF fields x 4 breakpoints = 136 MF genes), used here so the rates
# under test are the ones an actual --mf-optimize run sees.
RULE_LEN, MF_LEN = 81, 136


def _mutation_problem(rule_len, mf_len):
    """Minimal pymoo Problem carrying the same bounds layout run_ga builds:
    rule genes first (bounded [0, n_consequents]), MF genes second."""
    from pymoo.core.problem import Problem

    xl = np.concatenate([np.zeros(rule_len), np.zeros(mf_len)])
    xu = np.concatenate([np.full(rule_len, 3.0), np.full(mf_len, 1.0)])

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=rule_len + mf_len, n_obj=1, xl=xl, xu=xu)

    return _P()


def _measure(op, problem, X, trials=400, seed=0):
    """Measure realised mutation rates through Mutation.do() -- the public path
    the GA itself calls -- NOT through _do().

    This distinction is the whole point. _do() computes the per-gene
    perturbation but Mutation.do() *then* applies the per-individual gate
    (self.prob) and only writes back the individuals that pass it. Measuring
    _do() in isolation reports the nominal per-gene rate and is blind to the
    gate entirely -- which is exactly how a prob/prob_var mix-up that cut the
    effective rate ~100x went unnoticed.

    Returns (P(individual changed), per-gene rate in rule block, per-gene rate
    in MF block).
    """
    rng = np.random.default_rng(seed)
    n_ind = changed = 0
    gene_hits = np.zeros(problem.n_var)

    for _ in range(trials):
        Xp = op.do(problem, Population.new(X=X.copy()), random_state=rng).get('X')
        diff = ~np.isclose(Xp, X)
        changed += diff.any(axis=1).sum()
        gene_hits += diff.sum(axis=0)
        n_ind += len(X)

    rule_rate = gene_hits[:RULE_LEN].sum() / (n_ind * RULE_LEN)
    mf_rate = (gene_hits[RULE_LEN:].sum() / (n_ind * MF_LEN)
               if problem.n_var > RULE_LEN else 0.0)
    return changed / n_ind, rule_rate, mf_rate


def _mid_domain_X(problem, n=50, seed=1):
    """Non-degenerate starting population: every gene strictly interior to its
    bounds, so a mutation always has room to move it and cannot be masked by
    mut_pm's xl == xu guard or clipped back onto its own starting value."""
    xl, xu = problem.xl, problem.xu
    u = np.random.default_rng(seed).uniform(0.3, 0.7, size=(n, problem.n_var))
    return xl + u * (xu - xl)


class TestMakeMutation:
    def test_rule_only_returns_polynomial_mutation_with_per_gene_prob_var(self):
        """Rule-only mode: the per-INDIVIDUAL gate is 1.0 (mutate everyone) and
        the per-GENE rate is 1/rule_len -- passed as prob_var, not prob."""
        from pymoo.core.variable import get
        from pymoo.operators.mutation.pm import PolynomialMutation

        op = _make_mutation(RULE_LEN, 0, eta=20.0)

        assert isinstance(op, PolynomialMutation)
        assert get(op.prob) == 1.0
        assert get(op.prob_var) == pytest.approx(1.0 / RULE_LEN)

    def test_rule_only_effective_per_gene_rate_is_one_over_rule_len(self):
        """Regression guard for the prob/prob_var mix-up.

        The old call site -- PolynomialMutation(prob=1.0/n_var) -- set the
        per-individual gate to 1/81 while the per-gene rate defaulted to 1/81
        as well, so the two multiplied and the realised per-gene rate was
        ~(1/81)**2 ~= 0.00012 rather than 0.0123. This test measures the
        realised rate through .do() and fails loudly against that code.
        """
        problem = _mutation_problem(RULE_LEN, 0)
        X = _mid_domain_X(problem)
        op = _make_mutation(RULE_LEN, 0, eta=20.0)

        ind_rate, rule_rate, _ = _measure(op, problem, X)

        assert rule_rate == pytest.approx(1.0 / RULE_LEN, rel=0.15)
        # Every individual passes the gate, so P(>=1 gene changed) is just
        # 1-(1-1/81)^81 ~= 0.63 -- not the ~0.007 the old operator produced.
        assert ind_rate == pytest.approx(1 - (1 - 1 / RULE_LEN) ** RULE_LEN, rel=0.1)

    def test_combined_mode_returns_block_chunked_variant(self):
        from pymoo.core.mutation import Mutation
        from pymoo.operators.mutation.pm import PolynomialMutation

        op = _make_mutation(RULE_LEN, MF_LEN, eta=20.0)

        assert isinstance(op, Mutation)
        assert not isinstance(op, PolynomialMutation)
        assert op.rule_prob == pytest.approx(1.0 / RULE_LEN)
        assert op.mf_prob == pytest.approx(1.0 / MF_LEN)

    def test_combined_mode_hits_target_per_block_rates(self):
        """Each block mutates at its own 1/block_len rate. A single shared
        1/n_var rate would give BOTH blocks 1/217 = 0.0046, starving the rule
        block; this asserts the rule block keeps the 1/81 it would get if it
        were the only thing being optimised."""
        problem = _mutation_problem(RULE_LEN, MF_LEN)
        X = _mid_domain_X(problem)
        op = _make_mutation(RULE_LEN, MF_LEN, eta=20.0)

        _, rule_rate, mf_rate = _measure(op, problem, X)

        assert rule_rate == pytest.approx(1.0 / RULE_LEN, rel=0.15)
        assert mf_rate == pytest.approx(1.0 / MF_LEN, rel=0.15)
        # And specifically NOT the diluted shared rate.
        shared = 1.0 / (RULE_LEN + MF_LEN)
        assert rule_rate > 1.5 * shared

    def test_combined_mode_respects_per_block_bounds(self):
        """Rule genes are bounded [0, 3] and MF genes [0, 1] here; the operator
        slices problem.xl/xu per block, so a slice/ordering mismatch would push
        MF genes past 1.0 without erroring."""
        problem = _mutation_problem(RULE_LEN, MF_LEN)
        X = _mid_domain_X(problem)
        op = _make_mutation(RULE_LEN, MF_LEN, eta=20.0)

        Xp = op.do(problem, Population.new(X=X.copy()),
                   random_state=np.random.default_rng(3)).get('X')

        assert np.all(Xp >= problem.xl - 1e-12)
        assert np.all(Xp <= problem.xu + 1e-12)
        assert Xp[:, :RULE_LEN].max() <= 3.0 + 1e-12
        assert Xp[:, RULE_LEN:].max() <= 1.0 + 1e-12

    def test_combined_mode_is_deterministic_given_the_same_random_state(self):
        """The block-chunked operator draws from the passed-in random_state
        twice (once per mut_pm call). Sequential consumption from one Generator
        is still deterministic, so a given seed must reproduce exactly."""
        problem = _mutation_problem(RULE_LEN, MF_LEN)
        X = _mid_domain_X(problem)

        def _run(seed):
            op = _make_mutation(RULE_LEN, MF_LEN, eta=20.0)
            return op.do(problem, Population.new(X=X.copy()),
                         random_state=np.random.default_rng(seed)).get('X')

        np.testing.assert_array_equal(_run(42), _run(42))
        assert not np.array_equal(_run(42), _run(43))

    def test_at_least_once_false_leaves_some_individuals_untouched_per_block(self):
        """at_least_once=False (PolynomialMutation's default, preserved here) is
        the correct choice for a per-gene-rate model: an individual takes ~1
        mutation per block ON AVERAGE, so ~37% of individuals take none in the
        rule block. at_least_once=True would force every individual to take at
        least one mutation in EVERY block, inflating the realised rate."""
        problem = _mutation_problem(RULE_LEN, MF_LEN)
        X = _mid_domain_X(problem)
        op = _make_mutation(RULE_LEN, MF_LEN, eta=20.0)

        rng = np.random.default_rng(11)
        untouched = total = 0
        for _ in range(100):
            Xp = op.do(problem, Population.new(X=X.copy()), random_state=rng).get('X')
            rule_changed = (~np.isclose(Xp[:, :RULE_LEN], X[:, :RULE_LEN])).any(axis=1)
            untouched += (~rule_changed).sum()
            total += len(X)

        expected = (1 - 1 / RULE_LEN) ** RULE_LEN  # ~0.366
        assert untouched / total == pytest.approx(expected, rel=0.1)


class TestRunGAReproducibility:
    """Same-seed reproducibility is the guarantee that survives the mutation
    fix. Cross-version bit-identity is deliberately given up: the fix changes
    rule-only GA trajectories (mutation was previously ~100x too weak), so
    pre-fix baselines must be re-run rather than compared against."""

    def test_rule_only_same_seed_reproduces_exactly(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)

        def _run():
            return run_ga(
                schema=toy_schema,
                fitness_fn=_n_active_rules_fitness(codec),
                pop_size=8, n_gen=4, seed=42, verbose=False,
                run_dir_base=str(tmp_path), smoke_test=False,
            )

        a, b = _run(), _run()
        np.testing.assert_array_equal(a['best_chromosome'], b['best_chromosome'])
        assert a['best_score'] == b['best_score']

    def test_mf_optimize_same_seed_reproduces_exactly(self, toy_schema, tmp_path):
        codec = RuleChromosomeCodec(toy_schema)
        mf_cls = build_it2_mf_params_class(toy_schema)
        mf_len = len(mf_chromosome_bounds(mf_cls, toy_schema)[0])

        def _run():
            return run_ga(
                schema=toy_schema,
                fitness_fn=_combined_fitness(codec, codec.chrom_len, mf_len),
                mf_params_cls=mf_cls,
                pop_size=8, n_gen=4, seed=42, verbose=False,
                run_dir_base=str(tmp_path), smoke_test=False,
            )

        a, b = _run(), _run()
        np.testing.assert_array_equal(a['best_chromosome'], b['best_chromosome'])
        assert a['best_score'] == b['best_score']
