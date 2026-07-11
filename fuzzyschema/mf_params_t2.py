"""
fuzzyschema/mf_params_t2.py
---------------------------
Schema-driven IT2 membership-function parameter class + T1->IT2 conversion
+ ex_fuzzy variable builders. Application-agnostic: takes a Schema and
(optionally) an already-built T1 MFParams instance.

IVFS argument order (verified empirically against ex_fuzzy):
  IVFS(name, LMF_params, UMF_params, domain) -- LMF first, UMF second.

FOU design (symmetric delta expansion from T1 breakpoints), left-boundary
and right-boundary handling, and flat-top clamping are schema-independent --
only the field list and variable/domain lookups vary between schemas.
"""

from __future__ import annotations

import dataclasses
from typing import Type, Union

import numpy as np
from ex_fuzzy.fuzzy_sets import IVFS, fuzzyVariable

from fuzzyschema.variable_config import Schema, Trap, check_trap


# ── Minimum UMF width (GA-mutation repair only, see _make_from_vector_it2) ────

# Fraction of a field's own domain width used as the minimum UMF trapezoid
# width floor. Domain-relative, not an absolute constant: variable domains
# vary wildly in scale across a schema (e.g. lidar_density's (0.0, 12.0) vs
# lidar_conf's (0.0, 1.0)) -- a fixed absolute epsilon would be negligible
# for the former and oversized for the latter. 0.1% is small enough to be a
# no-op on any trapezoid with a normal, non-degenerate spread (real UMF
# widths are expected to be orders of magnitude larger relative to the
# domain), but large enough to pull a GA-mutated near-zero-width UMF (all 4
# raw genes landing within noise of each other) away from true degeneracy.
_MIN_WIDTH_FRACTION = 1e-3


def _widen_if_degenerate(a: float, b: float, c: float, d: float,
                          epsilon: float, dom_max: float):
    """
    If the trapezoid's overall width (d - a) is below epsilon, push d out to
    a + epsilon, clamped to the variable's domain max, so the UMF -- and
    everything derived from it downstream (the LMF via containment
    clamping, check_trap's a<=b<=c<=d) -- never collapses to a degenerate
    zero/near-zero-width shape. Only d moves; a/b/c are left untouched, so
    ordering (a<=b<=c<=new_d) is preserved for free, since new_d >= old
    d >= c by construction (widening only ever increases d).
    """
    if d - a < epsilon:
        d = min(a + epsilon, dom_max)
    return a, b, c, d


# ── Containment validation ───────────────────────────────────────────────────

def _check_containment(term: str, umf: Trap, lmf: Trap) -> None:
    # IT2 requirement: the "upper" MF must bound the "lower" MF at every
    # breakpoint -- UMF's rising edge (a, b) must sit at or left of LMF's,
    # and UMF's falling edge (c, d) must sit at or right of LMF's.
    a_u, b_u, c_u, d_u = umf
    a_l, b_l, c_l, d_l = lmf
    if not (a_u <= a_l and b_u <= b_l and c_u >= c_l and d_u >= d_l):
        raise ValueError(
            f"IT2MFParams term '{term}': UMF {umf} does not contain LMF {lmf}.\n"
            f"  Requires: a_u\u2264a_l ({a_u}\u2264{a_l}), b_u\u2264b_l ({b_u}\u2264{b_l}), "
            f"c_u\u2265c_l ({c_u}\u2265{c_l}), d_u\u2265d_l ({d_u}\u2265{d_l})"
        )


def _post_init_it2(self) -> None:
    """Validate ordering and UMF/LMF containment on construction. Fully
    generic: discovers pairing from field-name suffixes, needs no schema."""
    field_names = {f.name for f in dataclasses.fields(self)}

    # Pass 1: every individual trapezoid (UMF and LMF fields alike) must be
    # internally ordered (a <= b <= c <= d) on its own, regardless of how it
    # relates to its pair.
    for f in dataclasses.fields(self):
        check_trap(f"{type(self).__name__}.{f.name}", getattr(self, f.name))

    # Pass 2: only once every trap is individually valid, check that each
    # UMF/LMF pair satisfies containment. Only scan the "_umf" fields and
    # derive the matching "_lmf" name, so each pair is checked once, not twice.
    for f in dataclasses.fields(self):
        if f.name.endswith('_umf'):
            lmf_name = f.name[:-4] + '_lmf'
            if lmf_name in field_names:  # skip if this class has no matching LMF field
                _check_containment(
                    f.name[:-4], getattr(self, f.name), getattr(self, lmf_name),
                )


def _make_from_vector_it2(pairs: dict, field_domain: dict):
    """Build a from_vector classmethod body closing over this schema's
    umf_field -> lmf_field pairing and per-field domains (both computed
    once at class-build time).

    `pairs`/`field_domain` are baked in via closure so the returned function
    can have the plain (cls, v) signature every from_vector needs, while
    still knowing which fields pair up and each field's domain (for the
    minimum-width epsilon, itself domain-relative -- see
    _MIN_WIDTH_FRACTION) for *this* schema. field_domain is keyed by the
    bare term field name (schema.field_domains()'s own convention), so a
    field's own '_umf'/'_lmf' suffix is stripped before lookup, same
    pattern chromosome.py's mf_chromosome_bounds uses.
    """

    def _from_vector(cls, v: np.ndarray):
        field_names = [f.name for f in dataclasses.fields(cls)]
        n_fields = len(field_names)
        if len(v) != n_fields * 4:
            raise ValueError(f"Expected {n_fields * 4} floats, got {len(v)}")

        field_idx = {name: i for i, name in enumerate(field_names)}
        trap_dict: dict = {}
        processed: set = set()  # fields already decoded via a umf/lmf pair

        for umf_name, lmf_name in pairs.items():
            if umf_name not in field_idx or lmf_name not in field_idx:
                continue
            i_u, i_l = field_idx[umf_name], field_idx[lmf_name]

            # UMF's own 4 genes sort independently -- guarantees a valid
            # trapezoid on its own, regardless of what the raw floats were.
            a_u, b_u, c_u, d_u = sorted(v[i_u * 4:(i_u + 1) * 4])

            # GA mutation can legally land all 4 raw genes within noise of
            # each other, sorting into a valid-but-degenerate (zero/near-
            # zero-width) UMF. Widen it here, before the LMF is derived from
            # it below, so the LMF's containment clamp (and check_trap's
            # ordering check, which every field goes through in
            # __post_init__) never has to cope with a degenerate UMF.
            dom_min, dom_max = field_domain[umf_name[:-4]]
            epsilon = _MIN_WIDTH_FRACTION * (dom_max - dom_min)
            a_u, b_u, c_u, d_u = _widen_if_degenerate(a_u, b_u, c_u, d_u, epsilon, dom_max)

            # LMF is built as a chain anchored to the now-fixed UMF: each
            # point is clamped into the range implied by containment (must
            # stay within the matching UMF bound) and ordering (must stay
            # at or above the previous LMF point). Every clip range is
            # provably non-empty given a valid UMF, so this always produces
            # a valid, contained LMF for *any* raw h1..h4 -- the same repair
            # guarantee the old joint-sort scheme had. Unlike joint-sort,
            # this reaches the *entire* space of valid IT2 configurations
            # (including "wide UMF / narrow LMF" shapes where the old
            # scheme's positional sort-assignment couldn't reconstruct the
            # original pair) -- and for an already-valid LMF, every clip is
            # the identity, since a valid LMF's points already sit inside
            # their own range by definition. That's what makes to_vector an
            # exact inverse for *any* valid instance, not just non-crossing
            # ones.
            h1, h2, h3, h4 = v[i_l * 4:(i_l + 1) * 4]
            a_l = min(max(h1, a_u), c_u)
            b_l = min(max(h2, b_u, a_l), c_u)
            c_l = min(max(h3, b_l), c_u)
            d_l = min(max(h4, c_l), d_u)

            trap_dict[umf_name] = (float(a_u), float(b_u), float(c_u), float(d_u))
            trap_dict[lmf_name] = (float(a_l), float(b_l), float(c_l), float(d_l))
            processed.add(umf_name)
            processed.add(lmf_name)

        # Any field with no pairing partner (shouldn't normally happen, but
        # handled defensively) just gets its own 4 floats sorted independently
        # -- no containment relationship to preserve. Same degenerate-width
        # guard as the paired UMF branch above -- a lone field is just as
        # capable of landing all 4 genes within noise of each other.
        for fn in field_names:
            if fn not in processed:
                i = field_idx[fn]
                a, b, c, d = sorted(v[i * 4:(i + 1) * 4])
                base = fn[:-4] if fn.endswith(('_umf', '_lmf')) else fn
                dom_min, dom_max = field_domain[base]
                epsilon = _MIN_WIDTH_FRACTION * (dom_max - dom_min)
                a, b, c, d = _widen_if_degenerate(a, b, c, d, epsilon, dom_max)
                trap_dict[fn] = (float(a), float(b), float(c), float(d))

        return cls(**trap_dict)

    return _from_vector


def _make_to_vector_it2(pairs: dict):
    """Build a to_vector method body closing over this schema's
    umf_field -> lmf_field pairing, mirroring _make_from_vector_it2.

    Concatenates each pair as (umf's 4 floats, lmf's 4 floats) directly --
    no sorting, no precondition check. __post_init__ already guarantees
    each trap is individually ordered and UMF contains LMF, and
    from_vector's clamp-chain decode (anchored to UMF, each LMF point
    clamped into its containment/ordering range) is the identity on
    already-valid points -- so this round-trips exactly for *any* valid
    IT2MFParams instance, not just non-crossing ones.
    """

    def _to_vector(self) -> np.ndarray:
        field_names = [f.name for f in dataclasses.fields(self)]
        return np.concatenate(
            [np.asarray(getattr(self, f), dtype=float) for f in field_names]
        )

    return _to_vector


def build_it2_mf_params_class(schema: Schema, class_name: str = "IT2MFParams") -> Type:
    """
    Generate an IT2MFParams-equivalent dataclass with a _umf/_lmf field pair
    per (variable, term) in `schema`. No defaults -- always construct via
    make_it2_from_t1() (preferred), cls.from_vector(), or explicit kwargs.
    """
    fields = []
    pairs = {}  # umf_field_name -> lmf_field_name, passed to from_vector's closure
    for var in schema.all_vars:
        for term in var.terms:
            umf_name, lmf_name = term.field + '_umf', term.field + '_lmf'
            fields.append((umf_name, Trap))
            fields.append((lmf_name, Trap))
            pairs[umf_name] = lmf_name

    cls = dataclasses.make_dataclass(
        class_name,
        fields,
        namespace={
            "__post_init__": _post_init_it2,
            "from_vector": classmethod(_make_from_vector_it2(pairs, schema.field_domains())),
            "to_vector": _make_to_vector_it2(pairs),
        },
    )
    return cls


# ── Factory: T1 -> IT2 ────────────────────────────────────────────────────────

def make_it2_from_t1(
    schema: Schema,
    t1,
    delta: Union[float, dict],
    it2_cls: Type,
):
    """
    Construct an instance of `it2_cls` from a T1 params instance by applying
    a symmetric FOU, per-field domain and delta lookups derived from `schema`.

    delta: float (applied to all variables) or dict[var_name, float]
           (variables absent from the dict use 0.05).
    """
    # field name -> owning variable name (needed by get_delta to key into
    # the user-supplied delta dict, which is keyed by variable name).
    field_to_var = {
        term.field: var.name for var in schema.all_vars for term in var.terms
    }
    # field name -> owning variable's domain, via the shared Schema method
    # (single source of truth also used by chromosome.py's mf_chromosome_bounds).
    field_domain = schema.field_domains()

    def get_delta(field_name: str) -> float:
        if isinstance(delta, dict):
            return delta.get(field_to_var.get(field_name, ""), 0.05)
        return float(delta)

    def get_domain(field_name: str):
        return field_domain.get(field_name, (-float('inf'), float('inf')))

    def expand_contract(trap, d, domain):
        """Widen one T1 trapezoid into a UMF/LMF pair by symmetric delta
        expansion, with edge-case guards so the result stays valid."""
        a, b, c, dd = trap
        dom_min, dom_max = domain

        if a <= dom_min:
            # Left edge already sits at the domain minimum (e.g. a "LOW" term
            # starting at 0) -- there's no room to expand further left, so
            # UMF and LMF keep the same a, b. Cap eff_d at (c - b) so whatever
            # delta is applied to the right side still can't push b past c.
            eff_d = min(d, c - b)
            umf_a = lmf_a = a
            umf_b = lmf_b = b
        else:
            # Normal case: UMF spreads a/b outward (wider), LMF pulls them
            # inward (narrower). Cap eff_d at (c-b)/2 so the LMF's b can never
            # cross past c -- i.e. the delta can't overtake the flat top.
            # Also cap at (a - dom_min) so umf_a can't be pushed below the
            # variable's domain floor when `a` is close to but not pinned at
            # dom_min -- symmetric to the dom_max clamps already applied to
            # umf_c/umf_d below.
            eff_d = min(d, (c - b) / 2.0, a - dom_min)
            umf_a, lmf_a = a - eff_d, a + eff_d
            umf_b, lmf_b = b - eff_d, b + eff_d

        # Right side, same for both branches above: UMF's c/d push outward
        # but clamp at the domain max; LMF's c/d pull inward but never
        # collapse below lmf_b (which would invert the trapezoid).
        umf_c = min(c + eff_d, dom_max)
        umf_d = min(dd + eff_d, dom_max)
        lmf_c = max(c - eff_d, lmf_b)
        lmf_d = max(dd - eff_d, lmf_c)

        # If the original T1 trapezoid's flat top was already pinned at the
        # domain max (e.g. a "HIGH" term ending at the ceiling), don't let
        # the contraction above shrink LMF's top away from that ceiling --
        # force it back so the "extends to the domain edge" property holds.
        if c >= dom_max and dd >= dom_max:
            lmf_c = lmf_d = dom_max

        return (umf_a, umf_b, umf_c, umf_d), (lmf_a, lmf_b, lmf_c, lmf_d)

    trap_dict = {}
    for f in dataclasses.fields(t1):
        d = get_delta(f.name)
        domain = get_domain(f.name)
        umf, lmf = expand_contract(getattr(t1, f.name), d, domain)
        trap_dict[f.name + '_umf'] = umf
        trap_dict[f.name + '_lmf'] = lmf

    return it2_cls(**trap_dict)


# ── Generic IT2 variable builders ────────────────────────────────────────────

def get_antecedents_t2(schema: Schema, params) -> list:
    """Build the ordered list of ex_fuzzy antecedent IVFS-based fuzzyVariables
    from an IT2MFParams instance. Note IVFS's (LMF, UMF) argument order."""
    result = []
    for var in schema.antecedents:
        fsets = [
            IVFS(
                term.label,
                list(getattr(params, term.field + '_lmf')),  # LMF first
                list(getattr(params, term.field + '_umf')),  # UMF second
                domain=list(var.domain),
            )
            for term in var.terms
        ]
        result.append(fuzzyVariable(var.name, fsets))
    return result


def get_output_var_t2(schema: Schema, params) -> fuzzyVariable:
    """Build the ex_fuzzy output IVFS-based fuzzyVariable from an
    IT2MFParams instance."""
    fsets = [
        IVFS(
            term.label,
            list(getattr(params, term.field + '_lmf')),
            list(getattr(params, term.field + '_umf')),
            domain=list(schema.output.domain),
        )
        for term in schema.output.terms
    ]
    return fuzzyVariable(schema.output.name, fsets)
