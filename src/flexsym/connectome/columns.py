"""Derive the columnar partition -- the candidate equitable partition to test.

This is the part that matters scientifically, so it is kept separate from the
downloading and made auditable: every routine here reports how many neurons it
could actually assign, because a columnar partition that silently covers 60% of
the graph would quietly wreck the equitability statistics.

Two routes to a per-neuron column label:

  `from_instance`  -- parse the hemibrain `instance` string, which encodes PB
                      glomerulus / EB sector for columnar types (e.g. the
                      "(PB06a)" in "PEN_a(PB06a)_L"). Self-contained, no extra
                      download, but it is string parsing and coverage must be
                      checked.
  `from_table`     -- join an external curated assignment keyed by bodyId. Use
                      this if you obtain a per-neuron columnar table; the eLife
                      Hulse et al. supplementary tables are *type*-level
                      (Tables 2-4 list cell types, not body IDs), so they do not
                      substitute for a bodyId->column mapping on their own.

Both return an integer label vector aligned to a given neuron ordering, with -1
for unassigned, plus a coverage report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: PB glomeruli: L9..L1, R1..R9 appear as e.g. PB01, PB06a, PB09b.
_PB = re.compile(r"\bPB(\d{2})([ab])?\b", re.IGNORECASE)
#: EB sectors/wedges: e.g. EB05, and the R1..R4 ring subdivisions.
_EB = re.compile(r"\bEB(\d{2})\b", re.IGNORECASE)
#: Anything inside parentheses, as a last resort for inspection.
_PAREN = re.compile(r"\(([^)]*)\)")


@dataclass
class Partition:
    """An integer partition over neurons, with provenance and coverage."""

    labels: np.ndarray  #: int array, -1 = unassigned
    names: list[str]  #: names[i] is the label for integer i
    source: str
    n_total: int
    coverage: float
    per_cell_size: dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        sizes = sorted(self.per_cell_size.items(), key=lambda kv: -kv[1])
        head = ", ".join(f"{k}:{v}" for k, v in sizes[:8])
        return (
            f"[{self.source}] {len(self.names)} cells over "
            f"{int(self.coverage * self.n_total)}/{self.n_total} neurons "
            f"({self.coverage:.1%} coverage)\n  largest: {head}"
        )

    def assigned_mask(self) -> np.ndarray:
        return self.labels >= 0


def _key_from_instance(instance: str) -> str | None:
    if not isinstance(instance, str):
        return None
    m = _PB.search(instance)
    if m:
        return f"PB{int(m.group(1)):02d}{(m.group(2) or '').lower()}"
    m = _EB.search(instance)
    if m:
        return f"EB{int(m.group(1)):02d}"
    return None


def from_instance(neurons: pd.DataFrame, instance_col: str = "instance") -> Partition:
    """Parse columnar identity out of hemibrain instance strings."""
    if instance_col not in neurons.columns:
        raise KeyError(f"{instance_col!r} not in neurons; got {list(neurons.columns)}")

    keys = [_key_from_instance(s) for s in neurons[instance_col]]
    return _to_partition(keys, source=f"instance:{instance_col}")


def from_table(
    neurons: pd.DataFrame,
    table: pd.DataFrame,
    *,
    id_col: str = "bodyId",
    table_id_col: str = "bodyId",
    column_col: str = "column",
) -> Partition:
    """Join an external per-neuron columnar assignment onto `neurons`."""
    for col, df, what in ((id_col, neurons, "neurons"), (table_id_col, table, "table")):
        if col not in df.columns:
            raise KeyError(f"{col!r} not in {what}; got {list(df.columns)}")
    if column_col not in table.columns:
        raise KeyError(f"{column_col!r} not in table; got {list(table.columns)}")

    mapping = dict(zip(table[table_id_col], table[column_col]))
    keys = [mapping.get(i) for i in neurons[id_col]]
    keys = [k if isinstance(k, str) and k else None for k in keys]
    return _to_partition(keys, source=f"table:{column_col}")


def by_type(neurons: pd.DataFrame, type_col: str = "type") -> Partition:
    """Partition by cell type -- the natural comparison partition.

    Cell type is the coarse grouping ("all EPGs together"); the column is the
    fine one ("EPGs in wedge 5"). Testing both separates "does anatomy give an
    equitable partition at all" from "is it the *columnar* structure that does
    the work", which is the claim worth making.
    """
    if type_col not in neurons.columns:
        raise KeyError(f"{type_col!r} not in neurons; got {list(neurons.columns)}")
    keys = [s if isinstance(s, str) and s else None for s in neurons[type_col]]
    return _to_partition(keys, source=f"type:{type_col}")


def cross(a: Partition, b: Partition) -> Partition:
    """The common refinement of two partitions (e.g. type x column).

    A neuron unassigned in either input is unassigned in the result.
    """
    if len(a.labels) != len(b.labels):
        raise ValueError("partitions cover different neuron counts")
    keys: list[str | None] = []
    for la, lb in zip(a.labels, b.labels):
        keys.append(None if la < 0 or lb < 0 else f"{a.names[la]}|{b.names[lb]}")
    return _to_partition(keys, source=f"cross({a.source},{b.source})")


def _to_partition(keys: list[str | None], *, source: str) -> Partition:
    names = sorted({k for k in keys if k is not None})
    index = {name: i for i, name in enumerate(names)}
    labels = np.array([index[k] if k is not None else -1 for k in keys], dtype=int)
    n_total = len(keys)
    sizes = {name: int((labels == i).sum()) for name, i in index.items()}
    return Partition(
        labels=labels,
        names=names,
        source=source,
        n_total=n_total,
        coverage=float((labels >= 0).mean()) if n_total else 0.0,
        per_cell_size=sizes,
    )


def restrict(
    labels: np.ndarray, W: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unassigned neurons from both the label vector and the matrix.

    Equitability is only defined on a full partition, so unassigned neurons must
    be removed rather than lumped into a junk cell -- a junk cell would be
    heterogeneous by construction and drag every statistic toward "not
    equitable".
    """
    keep = labels >= 0
    sub = labels[keep]
    _, sub = np.unique(sub, return_inverse=True)  # re-canonicalise to 0..k-1
    return sub, W[np.ix_(keep, keep)]
