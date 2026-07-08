# fuzzyschema

A schema-driven generation and genetic-optimization layer for Type-1 and
Interval Type-2 Fuzzy Logic Systems, built on top of
[`ex_fuzzy`](https://github.com/Fuminides/ex-fuzzy).

`fuzzyschema` is not a replacement for `ex_fuzzy` — it uses `ex_fuzzy`'s
fuzzy-set primitives and rule-firing math directly. What it adds is
everything `ex_fuzzy` doesn't provide: declaring a variable schema once and
deriving membership-function parameter classes, chromosome encodings, and a
GA optimization harness from it automatically, for any application, without
hand-writing dataclasses or wiring a new optimizer per project.

## What it does

Given a declarative description of an FLS's variables (antecedents, output,
their domains, and their linguistic terms), `fuzzyschema`:

- Generates typed, validated parameter dataclasses for the FLS's membership
  functions (T1 trapezoids, or IT2 upper/lower trapezoid pairs), with field
  names and defaults derived from the schema — nothing hand-written.
- Builds valid IT2 membership functions from T1 ones via domain-aware delta
  expansion (`make_it2_from_t1`), or lets you construct IT2 sets directly
  with fully independent upper/lower bounds.
- Encodes/decodes GA chromosomes for rule bases, MF parameters, or both
  jointly — chromosome length, bounds, and layout are all derived from the
  schema, never hardcoded.
- Runs genetic optimization (via [pymoo](https://pymoo.org)) against a
  fitness function you supply, optimizing rules only, MF parameters only,
  or both together, symmetrically for T1 or IT2.
- Runs FLS inference (Mamdani, single shared rule base, Karnik-Mendel
  defuzzification for IT2) via a small, direct engine that works around two
  bugs found in `ex_fuzzy`'s own KM implementation (see [Known ex_fuzzy
  issues](#known-ex_fuzzy-issues-worked-around)).

The library only knows about *structure* — variables, domains, terms,
chromosomes. It has no knowledge of any specific application's rule
content, data, or fitness metric; those are supplied by the consumer at
every entry point (`fitness_fn`, `rules_fn`, etc.).

## Why this exists

`ex_fuzzy` supplies correct low-level fuzzy-set math, but every example
wires up variables, membership functions, and its own genetic tuner by
hand, per script. Its built-in optimizer (`evolutionary_fit.py`) is also
architecturally tied to classification — every rule's consequent is a
class index, and rule bases are partitioned one-per-class — which doesn't
fit a single shared-consequent Mamdani-regression design. `fuzzyschema`
fills the layer above `ex_fuzzy`'s primitives: schema-driven generation, a
chromosome codec that supports rule-only, MF-only, or joint optimization,
and a GA harness built directly on pymoo, application-agnostic throughout.

## Installation

```bash
pip install -e .
```

Dependencies (declared in `pyproject.toml`): `ex-fuzzy`, `numpy`, `pymoo`.

## Core concepts

### Schema

Everything starts with a `Schema` — antecedent variables plus one output
variable, each with a domain and an ordered set of linguistic terms:

```python
from fuzzyschema.variable_config import Schema, VariableSpec, TermSpec

schema = Schema(
    antecedents=(
        VariableSpec(
            name="lidar_conf", domain=(0.0, 1.0),
            terms=(
                TermSpec(label="LOW",  field="lidar_conf_low",  default=(-0.01, 0.0, 0.2, 0.4)),
                TermSpec(label="HIGH", field="lidar_conf_high", default=(0.6, 0.8, 1.0, 1.0)),
            ),
        ),
        # ... more antecedents
    ),
    output=VariableSpec(
        name="calibrated_conf", domain=(0.0, 1.0),
        terms=(
            TermSpec(label="LOW",  field="out_low",  default=(-0.01, 0.0, 0.3, 0.5)),
            TermSpec(label="HIGH", field="out_high", default=(0.5, 0.7, 1.0, 1.0)),
        ),
    ),
)
```

`field` is the attribute name that will appear on the generated MF
parameter dataclass — one field per (variable, term) pair. `default` is the
expert-anchored T1 trapezoid used as that field's default value; every term
used for MF-class generation must set it.

`Schema.field_domains()` returns `{field_name: (min, max)}` for every field
across the whole schema — the single source of truth used internally by
`make_it2_from_t1`'s domain clamping and by `mf_chromosome_bounds`, so a
variable's domain is declared exactly once.

### MF parameter classes

```python
from fuzzyschema.mf_params import build_mf_params_class

MFParams = build_mf_params_class(schema)
params = MFParams()                       # uses every term's `default`
params = MFParams(lidar_conf_low=(0, 0, 0.15, 0.35))  # override specific fields
```

Every generated class validates its own trapezoids on construction
(`a <= b <= c <= d` per field) and supports:

- `from_vector(v)` — decode a flat float array (e.g. a GA chromosome) into
  an instance, repairing any ordering violation by sorting each field's 4
  values. Always produces a valid instance regardless of input.
- `to_vector()` — the inverse: flatten an instance back to a chromosome.
  Exact round trip (`from_vector(to_vector(p)) == p`) since there's no
  reordering to invert for T1.

### IT2 parameter classes

```python
from fuzzyschema.mf_params_t2 import build_it2_mf_params_class, make_it2_from_t1

IT2MFParams = build_it2_mf_params_class(schema)

# Build IT2 sets from an existing T1 instance via domain-aware delta expansion:
it2_params = make_it2_from_t1(schema, params, delta=0.1, it2_cls=IT2MFParams)

# Or construct any valid IT2 set directly -- UMF and LMF are fully
# independent as long as UMF contains LMF (a_u<=a_l, b_u<=b_l, c_u>=c_l, d_u>=d_l):
it2_params = IT2MFParams(
    lidar_conf_low_umf=(-0.05, 0.0, 0.20, 0.40),
    lidar_conf_low_lmf=( 0.02, 0.05, 0.15, 0.35),
    # ...
)
```

`make_it2_from_t1` widens/narrows each T1 trapezoid by `delta` (a single
float, or a `{variable_name: delta}` dict for per-variable control),
clamped so the result never exceeds the variable's declared domain on
either side.

Every generated IT2 class validates, on construction, that every trap is
individually ordered *and* that UMF contains LMF at every breakpoint.
`from_vector`/`to_vector` work here too: `from_vector` decodes any raw
floats into a valid instance via a UMF-anchored clamp chain (UMF's own 4
genes sort independently; each LMF point is then clamped into the range
implied by containment and ordering relative to the fixed UMF) — this
repairs arbitrary GA-mutated input while reaching the *entire* space of
valid IT2 configurations, not a restricted subset. `to_vector` is a
correspondingly exact inverse for any valid instance.

### Rule authoring

```python
from fuzzyschema.rules import RuleFactory

rf = RuleFactory(schema)
rules = [
    rf.rule(lidar_conf=0, camera_conf=0, out=0),   # unspecified antecedents default to DONT_CARE
    rf.rule(lidar_conf=1, out=1),
]
```

`validate_rules(rules, schema)` checks antecedent length, index ranges, and
duplicate antecedent combinations.

### Chromosome encoding

```python
from fuzzyschema.chromosome import RuleChromosomeCodec

codec = RuleChromosomeCodec(schema)
chrom = codec.expert_chromosome(lambda: rules)   # seed from an expert rule base
decoded_rules = codec.decode(chrom)
lower, upper = codec.bounds()
```

`chrom_len` is the product of each antecedent's term count — one gene per
fully-specified antecedent combination. Gene value `0` means "rule
disabled"; `1..N` means "active, consequent = value - 1".

For MF-optimization mode:

```python
from fuzzyschema.chromosome import (
    mf_chromosome_bounds, build_combined_chromosome, split_combined_chromosome,
)

mf_lower, mf_upper = mf_chromosome_bounds(MFParams, schema)   # or IT2MFParams
combined = build_combined_chromosome(rule_chrom, mf_chrom)
rule_part, mf_part = split_combined_chromosome(combined, rule_len, mf_len)
```

### Inference engine

```python
from fuzzyschema.mf_params import get_antecedents, get_output_var
from fuzzyschema.engine import T1FLSEngine, T2FLSEngine

# T1:
engine = T1FLSEngine(get_antecedents(schema, params), rules, get_output_var(schema, params))
scores = engine.run_inference(X)   # X: (n_samples, n_antecedents)

# IT2:
from fuzzyschema.mf_params_t2 import get_antecedents_t2, get_output_var_t2
engine = T2FLSEngine(get_antecedents_t2(schema, it2_params), rules, get_output_var_t2(schema, it2_params))
scores = engine.run_inference(X)   # NaN where no rule fired
```

### GA optimization

```python
from fuzzyschema.ga import run_ga

result = run_ga(
    schema=schema,
    fitness_fn=my_fitness_fn,      # Callable(chromosome) -> float, to MAXIMISE
    rules_fn=lambda: rules,        # optional: seed rule block from an expert rule base
    pop_size=30,
    n_gen=50,
)
# result: best_chromosome, best_score, best_rules, history, run_dir
```

`fitness_fn` receives the raw chromosome array and is entirely responsible
for decoding it (via `codec.decode`, `split_combined_chromosome`, and/or
`mf_params_cls.from_vector` as needed) and building whatever it needs to
score it — `run_ga` never touches `ex_fuzzy` or builds an inference engine
itself.

**MF-optimization mode:** pass `mf_params_cls` (a class from
`build_mf_params_class` or `build_it2_mf_params_class`) to switch the
chromosome from rule-only to a combined rule+MF chromosome. There is no
separate boolean flag — `mf_params_cls` being set *is* the switch:

```python
result = run_ga(
    schema=schema,
    fitness_fn=my_combined_fitness_fn,
    rules_fn=lambda: rules,
    mf_params_cls=IT2MFParams,
    mf_seed_fn=lambda: expert_it2_params,   # optional: seed the MF block too
    pop_size=30,
    n_gen=50,
)
# result also has: best_mf_params
```

MF-block genes are bounded by `mf_chromosome_bounds` — each gene's search
range is exactly its owning variable's declared domain, derived from the
schema, never hardcoded. `mf_params_cls=None` (the default) reproduces
rule-only behavior exactly, with no `best_mf_params` key in the result and
no `ga_best_mf_params.json` artefact written.

Artefacts saved per run, under `run_dir_base/run_name/`:
`ga_config.json`, `ga_best_chromosome.npy`, `ga_best_rules.json`,
`ga_history.json`, and — MF-optimization mode only — `ga_best_mf_params.json`.

## Known ex_fuzzy issues worked around

- **KM type-reduction crash**: `ex_fuzzy`'s `centroid.py` calls
  `np.argwhere` on an array that can be empty (the all-secondary-firing-zero
  case), raising an `IndexError`. `engine.py`'s `_km_endpoint` reimplements
  Karnik-Mendel type reduction using `np.searchsorted` instead, avoiding the
  crash entirely. A fix upstream is planned.
- **Unordered `consequent_centroids`**: not guaranteed sorted; `engine.py`
  normalizes with `np.minimum`/`np.maximum` before use.
- **`IVFS` constructor argument order** is LMF first, UMF second — the
  reverse of what its docstring states, verified empirically. Every
  `IVFS(...)` call site in this library accounts for this explicitly.

## Testing

```bash
python -m pytest tests/
```

Tests are organized one file per module (`test_variable_config.py`,
`test_mf_params.py`, `test_mf_params_t2.py`, `test_rules.py`,
`test_chromosome.py`, `test_engine.py`, `test_ga.py`), using a shared
`conftest.py` fixture schema (`toy_schema`, plus a minimal
`single_antecedent_schema` for edge cases).

## Design principles

- **Schema-driven, not hardcoded.** Field names, defaults, domains,
  chromosome lengths, and bounds are all derived from a `Schema` instance.
  A new application defines a schema; it does not hand-write dataclasses,
  chromosome layouts, or bounds.
- **Application-agnostic.** No module in this library knows about any
  concrete application's variable names, rule content, dataset, or fitness
  metric. Those are always supplied by the consumer.
- **Interpretability-first.** Rule-only optimization (MF parameters held
  fixed) is fully supported and is the default — MF optimization is opt-in,
  not required, so a consumer can choose how much of the FLS's structure to
  keep fixed and expert-defined versus GA-searched.
- **Correctness over convenience.** Chromosome decoding always produces a
  valid instance regardless of input (repair, not rejection); encoding is
  an exact inverse wherever mathematically possible, and raises clearly
  where it isn't, rather than silently producing a mismatched result.
