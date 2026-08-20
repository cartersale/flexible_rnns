"""Shared, partition-agnostic metric infrastructure for the restart ensemble.

Reviewer concern #5: the signed-WL partition is non-unique. To settle whether the
downstream conclusions are properties of the *network* or of the *specific
partition*, we must run every reported metric on arbitrary partitions (restart
ensemble, consensus, size-matched random). This module provides:

  * ``ActivityCache`` -- computes ``compute_hidden_by_epoch`` once per (network,
    batch) and memoises it in-process AND on disk, so the 20-restart ensemble does
    not recompute activity 20x.
  * ``analyze(...)`` -- ONE function that takes (W_sub, labels, k, hidden,
    ind_active, [deficit_fn, activity_by_task]) and returns every metric:
    Gamma_signed, equitability z, eigenvector leakage (+z), subspace invariance
    (+z), linear & delay-Jacobian step-R^2 (+z), reorganization gap, connectivity
    -> function partial r, and (if a lesion ``deficit_fn`` is supplied) lesion
    deficit-activity alignment / max single-task drop / specificity.

The lesion metric is the only one needing a live TF graph; ``analyze`` stays
TF-free by taking a ``deficit_fn(units)->deficit_vector`` callable that the TF
driver supplies (backed by a per-network frozenset-keyed deficit cache).

Every metric reuses the exact production primitives so ensemble numbers are
comparable to the single-partition pipeline.
"""

from __future__ import annotations

from flexsym import paths

import sys
import warnings
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")

import numpy as np


# task.py monkeypatches np.load to force allow_pickle=True; make it kwargs-safe.
_patched_load = np.load
_orig_load = (
    _patched_load.__closure__[0].cell_contents
    if getattr(_patched_load, "__closure__", None)
    else _patched_load
)


def _dedup_load(*a: Any, **k: Any):
    k["allow_pickle"] = True
    return _orig_load(*a, **k)


np.load = _dedup_load

from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    gamma_signed,
    labels_to_H,
    random_labels_same_sizes,
)
from flexsym.run.lumpability_check import (  # noqa: E402
    eig_leakage,
    pw_constant_projector,
    subspace_invariance,
)
from flexsym.run.motif_jacobian_lumpability import best_r2  # noqa: E402
from flexsym.run.connectivity_functional_regression import (  # noqa: E402
    corr_sim,
    partial_corr,
    upper,
    zscore,
)
from flexsym.structural.softplus_memory_full_context_alignment import signed_input_features  # noqa: E402
from flexsym.structural.softplus_memory_subpartition_sweep import unit_features_for_period  # noqa: E402
from flexsym.structural.structural_cell_subpartition_pilot import choose_functional_partition  # noqa: E402

REPO = paths.REPO_ROOT
ACT_ROOT = paths.RESULTS_ROOT / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
# external drive has room; keep big activity tensors off the repo/scratch.
CACHE_ROOT = paths.CACHE_ROOT

# --- task-epoch batteries (verbatim from the production downstream scripts) --- #
REORG_BATTERY = [
    ("delaygo", "delay1"), ("delayanti", "delay1"), ("delaydm1", "delay1"),
    ("delaydm2", "delay1"), ("contextdelaydm1", "delay1"), ("multidelaydm", "delay1"),
    ("fdgo", "stim1"), ("fdanti", "stim1"), ("delaygo", "stim1"), ("delaydm1", "stim1"),
    ("reactgo", "go1"), ("reactanti", "go1"), ("fdgo", "go1"), ("delaygo", "go1"),
]
CONN_BATTERY = [
    ("fdgo", "stim1"), ("fdanti", "stim1"), ("delaydm1", "stim1"), ("dmsgo", "stim1"),
    ("delaygo", "delay1"), ("delayanti", "delay1"), ("delaydm1", "delay1"),
    ("contextdelaydm1", "delay1"), ("multidelaydm", "delay1"), ("dmsgo", "delay1"),
    ("reactgo", "go1"), ("fdgo", "go1"), ("fdanti", "go1"),
]
JAC_EPOCH = ("delaygo", "delay1")
EPOCH_TYPE = {"stim1": "stimulus", "delay1": "memory", "go1": "response"}


def all_needed_epochs() -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], None] = {}
    for te in [*REORG_BATTERY, *CONN_BATTERY, JAC_EPOCH]:
        seen.setdefault(te, None)
    return list(seen)


# --------------------------------------------------------------------------- #
# Activity cache
# --------------------------------------------------------------------------- #
class ActivityCache:
    """Memoised hidden activity per (model, batch). Disk-backed, float32."""

    def __init__(self, model_name: str, model_dir: Path, batch_size: int = 512):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.batch_size = int(batch_size)
        self._mem: dict[tuple[str, str], np.ndarray] = {}
        self.path = CACHE_ROOT / f"{model_name}_b{self.batch_size}.npz"

    @staticmethod
    def _key(rule: str, epoch: str) -> str:
        return f"{rule}|{epoch}"

    def _load_disk(self) -> None:
        if not self.path.exists() or self._mem:
            return
        try:
            z = np.load(self.path, allow_pickle=True)
            for kk in z.files:
                rule, epoch = kk.split("|")
                self._mem[(rule, epoch)] = np.asarray(z[kk], dtype=np.float32)
        except Exception:
            self._mem = {}

    def get(self, keys: list[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray]:
        """Return {(rule,epoch): hidden[T,B,U]} for keys, computing/caching misses.

        Keys are filtered to those valid for this network (invalid epochs simply
        do not appear in the returned dict, mirroring ``filter_valid_task_epochs``).
        """
        self._load_disk()
        missing = [k for k in keys if k not in self._mem]
        if missing:
            from flexsym.structural.structural_cell_cross_task_fractionation import filter_valid_task_epochs
            from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch

            valid = filter_valid_task_epochs(self.model_dir, missing, self.batch_size)
            global np  # np.load may have been re-wrapped by task.py import above
            np.load = _dedup_load
            if valid:
                fresh = compute_hidden_by_epoch(
                    self.model_dir, valid, batch_size=self.batch_size
                )
                np.load = _dedup_load
                for kk, v in fresh.items():
                    self._mem[kk] = np.asarray(v, dtype=np.float32)
            # mark invalid keys so we don't retry them every call
            for k in missing:
                self._mem.setdefault(k, None)  # type: ignore[assignment]
            self._flush()
        return {k: self._mem[k] for k in keys if self._mem.get(k) is not None}

    def _flush(self) -> None:
        try:
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            payload = {
                self._key(*k): v
                for k, v in self._mem.items()
                if v is not None
            }
            np.savez(self.path, **payload)
        except Exception:
            pass  # in-memory cache still serves this run


# --------------------------------------------------------------------------- #
# Tier A: structural metrics (W_sub + labels only, no activity, no TF)
# --------------------------------------------------------------------------- #
def _z(value: float, null: np.ndarray) -> float:
    sd = null.std(ddof=1)
    return float((value - null.mean()) / sd) if sd > 0 else float("nan")


def structural_metrics(
    W_sub: np.ndarray, labels: np.ndarray, k: int, *, n_null: int = 200, seed: int = 0
) -> dict[str, float]:
    W_pos = np.maximum(W_sub, 0.0)
    W_neg = np.maximum(-W_sub, 0.0)
    H = labels_to_H(labels, k)
    P = pw_constant_projector(labels, k)

    gamma = gamma_signed(W_pos, W_neg, H)
    leak = eig_leakage(W_sub, P, k)
    inv = subspace_invariance(W_sub, P)
    r2_lin = best_r2(W_sub, labels, k)

    g_null = np.empty(n_null)
    leak_null = np.empty(n_null)
    inv_null = np.empty(n_null)
    r2_null = np.empty(n_null)
    for b in range(n_null):
        nl = random_labels_same_sizes(labels, seed=seed * 100003 + b)
        Hn = labels_to_H(nl, k)
        Pn = pw_constant_projector(nl, k)
        g_null[b] = gamma_signed(W_pos, W_neg, Hn)
        leak_null[b] = eig_leakage(W_sub, Pn, k)
        inv_null[b] = subspace_invariance(W_sub, Pn)
        r2_null[b] = best_r2(W_sub, nl, k)

    return {
        "gamma_signed": float(gamma),
        "gamma_z": _z(gamma, g_null),          # negative = more equitable than chance
        "leakage": float(leak),
        "leakage_z": _z(leak, leak_null),      # negative = tighter lumping than chance
        "invariance": float(inv),
        "invariance_z": _z(inv, inv_null),
        "step_r2_linear": float(r2_lin),
        "step_r2_linear_z": _z(r2_lin, r2_null),  # positive = eigenvectors step on cells
    }


# --------------------------------------------------------------------------- #
# Tier B: activity metrics (need cached hidden activity; no TF graph)
# --------------------------------------------------------------------------- #
def reorganization_gap(
    labels: np.ndarray,
    ind_active: np.ndarray,
    hidden: dict[tuple[str, str], np.ndarray],
    *,
    min_cell: int = 5,
    max_k: int = 6,
) -> dict[str, float]:
    """Shared- minus non-shared-component sub-partition ARI (Fig 3 reorg gap)."""
    te = [k for k in REORG_BATTERY if k in hidden]
    cells = [c for c in np.unique(labels) if int((labels == c).sum()) >= min_cell]
    shared_pc, nonshared_pc = [], []
    for c in cells:
        pos = np.where(labels == c)[0]
        parts = {}
        for rule, ep in te:
            X = unit_features_for_period(hidden, rule, ep, ind_active, pos)
            r = choose_functional_partition(X, max_k=max_k)
            parts[(rule, ep)] = (np.asarray(r["labels"], dtype=np.int64), int(r["chosen_k"]))
        from sklearn.metrics import adjusted_rand_score

        keys = list(parts)
        sh, no = [], []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                (a, ka), (b, kb) = parts[keys[i]], parts[keys[j]]
                if ka < 2 or kb < 2:
                    continue
                ari = adjusted_rand_score(a, b)
                (sh if keys[i][1] == keys[j][1] else no).append(ari)
        if sh and no:
            shared_pc.append(float(np.mean(sh)))
            nonshared_pc.append(float(np.mean(no)))
    if not shared_pc:
        return {"reorg_shared": float("nan"), "reorg_nonshared": float("nan"),
                "reorg_gap": float("nan"), "reorg_ncell": 0}
    sh_m, no_m = float(np.mean(shared_pc)), float(np.mean(nonshared_pc))
    return {
        "reorg_shared": sh_m,
        "reorg_nonshared": no_m,
        "reorg_gap": sh_m - no_m,
        "reorg_ncell": len(shared_pc),
    }


def connectivity_function_partial_r(
    labels: np.ndarray,
    ind_active: np.ndarray,
    hidden: dict[tuple[str, str], np.ndarray],
    W_rec_full: np.ndarray,
    *,
    min_cell: int = 6,
) -> dict[str, float]:
    """Mean partial r(functional, full-context connectivity | internal wiring)."""
    te = [k for k in CONN_BATTERY if k in hidden]
    all_units = np.arange(W_rec_full.shape[1], dtype=np.int64)
    cells = [c for c in np.unique(labels) if int((labels == c).sum()) >= min_cell]
    r_full_g_int, r_full_raw = [], []
    for c in cells:
        pos = np.where(labels == c)[0]
        uids = ind_active[pos]
        if uids.size < 4:
            continue
        Sfull = corr_sim(signed_input_features(W_rec_full, uids, all_units))
        Sint = corr_sim(signed_input_features(W_rec_full, uids, uids))
        yfull = zscore(upper(Sfull))
        yint = zscore(upper(Sint))
        for rule, epoch in te:
            X = unit_features_for_period(hidden, rule, epoch, ind_active, pos)
            F = corr_sim(X)
            yF = zscore(upper(F))
            r_full_g_int.append(partial_corr(yF, yfull, [yint]))
            r_full_raw.append(partial_corr(yF, yfull, []))
    r_full_g_int = np.asarray(r_full_g_int, dtype=float)
    r_full_raw = np.asarray(r_full_raw, dtype=float)
    ok = np.isfinite(r_full_g_int)
    return {
        "partial_r": float(np.nanmean(r_full_g_int)) if ok.any() else float("nan"),
        "partial_r_raw_full": float(np.nanmean(r_full_raw)) if np.isfinite(r_full_raw).any() else float("nan"),
        "partial_r_ncell_period": int(ok.sum()),
    }


def jacobian_step_r2(
    W_sub: np.ndarray,
    labels: np.ndarray,
    k: int,
    ind_active: np.ndarray,
    hidden: dict[tuple[str, str], np.ndarray],
    *,
    n_null: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Step-R^2 of the delay-period Jacobian eigenvectors on the partition."""
    if JAC_EPOCH not in hidden:
        return {"step_r2_jacobian": float("nan"), "step_r2_jacobian_z": float("nan")}
    h = hidden[JAC_EPOCH]  # [T, B, U_full]
    hstar = h[-5:].mean(axis=(0, 1))[ind_active]  # active-unit fixed-point activation
    g = 1.0 - np.exp(-np.maximum(hstar, 0.0))  # softplus gain at h*
    M = g[:, None] * W_sub.T  # delay-Jacobian on active submatrix (transpose convention)
    r2 = best_r2(M, labels, k)
    null = np.empty(n_null)
    for b in range(n_null):
        null[b] = best_r2(M, random_labels_same_sizes(labels, seed=seed * 7919 + b), k)
    return {"step_r2_jacobian": float(r2), "step_r2_jacobian_z": _z(r2, null)}


# --------------------------------------------------------------------------- #
# Tier C hook: lesion metrics from an injected deficit callable
# --------------------------------------------------------------------------- #
def lesion_metrics(
    labels: np.ndarray,
    ind_active: np.ndarray,
    deficit_fn: Callable[[np.ndarray], np.ndarray],
    activity_by_task: np.ndarray,
    *,
    min_cell: int = 1,
) -> dict[str, float]:
    """Aggregate lesion deficit-activity alignment / maxdrop / specificity.

    ``deficit_fn(original_unit_ids) -> deficit_vector`` runs (or caches) the TF
    lesion. ``activity_by_task`` is A[n_rnn, n_rules] intact mean activation.
    """
    aligns, maxdrops, specs = [], [], []
    for c in np.unique(labels):
        pos = np.where(labels == c)[0]
        units = ind_active[pos]
        if units.size < min_cell:
            continue
        d = np.asarray(deficit_fn(units), dtype=float)
        a_g = activity_by_task[units, :].mean(axis=0)
        if np.std(a_g) > 0 and np.std(d) > 0:
            aligns.append(float(np.corrcoef(a_g, d)[0, 1]))
        maxdrops.append(float(d.max()))
        if d.sum() > 1e-9:
            specs.append(float(d.max() / d.sum()))
    return {
        "lesion_align": float(np.nanmean(aligns)) if aligns else float("nan"),
        "lesion_maxdrop": float(np.nanmean(maxdrops)) if maxdrops else float("nan"),
        "lesion_specificity": float(np.nanmean(specs)) if specs else float("nan"),
        "lesion_ncell": len(maxdrops),
    }


# --------------------------------------------------------------------------- #
# Umbrella
# --------------------------------------------------------------------------- #
def analyze(
    W_sub: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    ind_active: np.ndarray,
    W_rec_full: np.ndarray,
    hidden: dict[tuple[str, str], np.ndarray] | None = None,
    deficit_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    activity_by_task: np.ndarray | None = None,
    n_null: int = 200,
    seed: int = 0,
    tiers: tuple[str, ...] = ("A", "B", "C"),
) -> dict[str, float]:
    """Compute every reported metric for an arbitrary partition ``labels``."""
    out: dict[str, float] = {}
    if "A" in tiers:
        out.update(structural_metrics(W_sub, labels, k, n_null=n_null, seed=seed))
    if "B" in tiers and hidden is not None:
        out.update(reorganization_gap(labels, ind_active, hidden))
        out.update(connectivity_function_partial_r(labels, ind_active, hidden, W_rec_full))
        out.update(jacobian_step_r2(W_sub, labels, k, ind_active, hidden, n_null=n_null, seed=seed))
    if "C" in tiers and deficit_fn is not None and activity_by_task is not None:
        out.update(lesion_metrics(labels, ind_active, deficit_fn, activity_by_task))
    return out
