import dataclasses
import json
import os

import numpy as np
import pytest

from fuzzyschema.chromosome import RuleChromosomeCodec, mf_chromosome_bounds
from fuzzyschema.ga import run_ga, rules_to_readable
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
