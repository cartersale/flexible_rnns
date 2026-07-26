"""Signed input-side approximate WL structural-role sweep.

The activity-role input is the output from ``cluster_epoch_variance.py``. This
script compares activity partitions against hard signed WL partitions on
``W_rec`` using gamma_EP and shuffled-label nulls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)


STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))


def labels_to_H(labels: np.ndarray, k: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    H = np.zeros((n, k), dtype=float)
    valid = (labels >= 0) & (labels < k)
    H[np.arange(n)[valid], labels[valid]] = 1.0
    return H


def checked_matmul(A: np.ndarray, B: np.ndarray, name: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
        out = A @ B
    if not np.isfinite(out).all():
        n_bad = int(np.size(out) - np.isfinite(out).sum())
        raise FloatingPointError(f"{name} produced {n_bad} non-finite values.")
    return out


def random_labels_same_sizes(labels: np.ndarray, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.array(labels, copy=True)
    rng.shuffle(shuffled)
    return shuffled


def _canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster ids to 0..n_clusters-1 by first occurrence."""
    mapping: dict[int, int] = {}
    out = np.empty_like(labels, dtype=np.int64)
    next_label = 0
    for i, label in enumerate(labels):
        label_int = int(label)
        if label_int not in mapping:
            mapping[label_int] = next_label
            next_label += 1
        out[i] = mapping[label_int]
    return out


def approx_wl_signed(
    W_pos: np.ndarray,
    W_neg: np.ndarray,
    k: int,
    *,
    n_iter: int = 20,
    seed: int = 0,
    n_init: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    n = W_pos.shape[0]
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    H = labels_to_H(labels, k)

    for _ in range(n_iter):
        X = np.concatenate(
            [
                checked_matmul(W_pos, H, "W_pos @ H"),
                checked_matmul(W_neg, H, "W_neg @ H"),
            ],
            axis=1,
        )
        labels_new = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=seed,
        ).fit_predict(X)
        labels_new = _canonicalize_labels(labels_new)

        if np.array_equal(labels_new, labels):
            break

        labels = labels_new
        H = labels_to_H(labels, k)

    return labels, H


def gamma_ep(W: np.ndarray, H: np.ndarray) -> float:
    cell_sizes = H.sum(axis=0)
    D_inv_diag = 1.0 / np.maximum(cell_sizes, 1e-12)
    WH = checked_matmul(W, H, "W @ H")
    cell_means = checked_matmul(H.T, WH, "H.T @ WH") * D_inv_diag[:, None]
    projected = checked_matmul(H, cell_means, "H @ cell_means")
    return float(np.linalg.norm(WH - projected, ord="fro"))


def gamma_signed(W_pos: np.ndarray, W_neg: np.ndarray, H: np.ndarray) -> float:
    return gamma_ep(W_pos, H) + gamma_ep(W_neg, H)


def quotient_matrix(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    cell_sizes = H.sum(axis=0)
    D_inv_diag = 1.0 / np.maximum(cell_sizes, 1e-12)
    return checked_matmul(checked_matmul(H.T, W, "H.T @ W"), H, "H.T @ W @ H") * D_inv_diag[:, None]


def contingency_table(labels_struct: np.ndarray, labels_act: np.ndarray, k: int) -> np.ndarray:
    table = np.zeros((k, k), dtype=np.int64)
    for s, a in zip(labels_struct, labels_act):
        if 0 <= s < k and 0 <= a < k:
            table[int(s), int(a)] += 1
    return table


def majority_map_struct_to_activity(
    labels_struct: np.ndarray, labels_act: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    table = contingency_table(labels_struct, labels_act, k)
    mapping = np.argmax(table, axis=1)
    mapped = mapping[np.asarray(labels_struct, dtype=np.int64)]
    return mapped, mapping


def refinement_metrics(labels_struct: np.ndarray, labels_act: np.ndarray, k: int) -> dict:
    table = contingency_table(labels_struct, labels_act, k)
    purity = float(table.max(axis=1).sum() / max(table.sum(), 1))
    mapped_labels, mapping = majority_map_struct_to_activity(labels_struct, labels_act, k)
    return {
        "purity_struct_to_act": purity,
        "homogeneity_struct_to_act": float(homogeneity_score(labels_act, labels_struct)),
        "completeness_struct_to_act": float(completeness_score(labels_act, labels_struct)),
        "ami_struct_to_act": float(adjusted_mutual_info_score(labels_act, labels_struct)),
        "mapped_ari_struct_to_act": float(adjusted_rand_score(labels_act, mapped_labels)),
        "majority_mapping_struct_to_act": mapping,
        "contingency": table,
    }


def apply_matrix_variant(W_rec: np.ndarray, variant: str) -> np.ndarray:
    if variant == "original":
        return W_rec
    if variant == "transpose":
        return W_rec.T
    if variant == "no_diag":
        W_no_diag = W_rec.copy()
        np.fill_diagonal(W_no_diag, 0.0)
        return W_no_diag
    raise ValueError(f"Unknown matrix variant: {variant}")


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def load_w_rec(model_dir: Path) -> np.ndarray:
    hp_path = model_dir / "hp.json"
    params_path = model_dir / "model_params.npz"
    hp = json.loads(hp_path.read_text())
    params = np.load(params_path)
    n_rec = int(hp["n_rnn"])
    return np.asarray(params["w_in"][-n_rec:, :], dtype=float)


def _activity_labels_for_k(Z: np.ndarray, k: int) -> np.ndarray:
    return fcluster(Z, k, criterion="maxclust").astype(np.int64) - 1


def _plot_gamma(summary_rows: list[dict], output_dir: Path) -> None:
    k = np.array([row["k"] for row in summary_rows])
    gamma_act = np.array([row["gamma_act"] for row in summary_rows])
    gamma_struct = np.array([row["gamma_struct"] for row in summary_rows])
    gamma_null_mean = np.array([row["gamma_null_mean"] for row in summary_rows])
    gamma_null_sd = np.array([row["gamma_null_sd"] for row in summary_rows])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, gamma_struct, marker="o", ms=3, lw=1.5, label="Signed WL structural")
    ax.plot(k, gamma_act, marker="o", ms=3, lw=1.5, label="Activity")
    ax.fill_between(
        k,
        gamma_null_mean - gamma_null_sd,
        gamma_null_mean + gamma_null_sd,
        color="0.7",
        alpha=0.25,
        linewidth=0,
        label="Null mean +/- sd",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("signed gamma_EP")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "gamma_vs_k.png", dpi=200)
    fig.savefig(output_dir / "gamma_vs_k.pdf")
    plt.close(fig)


def _plot_chosen_null(
    chosen_k: int,
    gamma_act: float,
    gamma_struct: float,
    null_gammas: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_gammas, bins=40, color="0.75", edgecolor="white")
    ax.axvline(gamma_act, color="#2457a6", lw=2, label="Activity")
    ax.axvline(gamma_struct, color="#b23a2f", lw=2, label="Signed WL structural")
    ax.set_xlabel("signed gamma_EP")
    ax.set_ylabel("null count")
    ax.set_title(f"k={chosen_k}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"gamma_null_chosen_k_{chosen_k}.png", dpi=200)
    fig.savefig(output_dir / f"gamma_null_chosen_k_{chosen_k}.pdf")
    plt.close(fig)


def _plot_ari(summary_rows: list[dict], output_dir: Path) -> None:
    k = np.array([row["k"] for row in summary_rows])
    ari = np.array([row["ari"] for row in summary_rows])
    ari_null_mean = np.array([row["ari_null_mean"] for row in summary_rows])
    ari_null_sd = np.array([row["ari_null_sd"] for row in summary_rows])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, ari, marker="o", ms=3, lw=1.5, label="ARI")
    ax.fill_between(
        k,
        ari_null_mean - ari_null_sd,
        ari_null_mean + ari_null_sd,
        color="0.7",
        alpha=0.25,
        linewidth=0,
        label="Shuffled activity labels +/- sd",
    )
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlabel("k")
    ax.set_ylabel("ARI(structural, activity)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ari_vs_k.png", dpi=200)
    fig.savefig(output_dir / "ari_vs_k.pdf")
    plt.close(fig)


def _plot_quotient(B_pos: np.ndarray, B_neg: np.ndarray, chosen_k: int, output_dir: Path) -> None:
    vmax = max(float(B_pos.max(initial=0.0)), float(B_neg.max(initial=0.0)))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    for ax, B, title in zip(axes, [B_pos, B_neg], ["Positive", "Negative"]):
        im = ax.imshow(B, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("source structural class")
        ax.set_ylabel("target structural class")
    fig.colorbar(im, ax=axes, shrink=0.85, label="mean recurrent weight")
    fig.suptitle(f"Signed quotient matrices, k={chosen_k}")
    fig.savefig(output_dir / f"quotient_matrix_struct_k_{chosen_k}.png", dpi=200)
    fig.savefig(output_dir / f"quotient_matrix_struct_k_{chosen_k}.pdf")
    plt.close(fig)


def _plot_contingency(table: np.ndarray, chosen_k: int, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(table, cmap="viridis", aspect="auto")
    ax.set_xlabel("activity cluster")
    ax.set_ylabel("structural cell")
    ax.set_title(f"structural vs activity counts, k={chosen_k}")
    fig.colorbar(im, ax=ax, label="unit count")
    fig.tight_layout()
    fig.savefig(output_dir / f"contingency_struct_vs_act_k_{chosen_k}.png", dpi=200)
    fig.savefig(output_dir / f"contingency_struct_vs_act_k_{chosen_k}.pdf")
    plt.close(fig)


def run_signed_input_wl_sweep(
    model_dir: str | os.PathLike[str],
    *,
    activity_clusters: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    k_min: int = 3,
    k_max: int = 40,
    n_iter: int = 20,
    seed: int = 0,
    n_init: int = 20,
    n_null: int = 1000,
    chosen_k: int | None = None,
    matrix_variant: str = "original",
) -> dict:
    model_dir = Path(model_dir)
    if activity_clusters is None:
        activity_clusters = model_dir / "epoch_variance_unit_clusters.npz"
    activity_clusters = Path(activity_clusters)
    output_dir = Path(output_dir) if output_dir is not None else model_dir / "signed_input_wl"
    output_dir.mkdir(parents=True, exist_ok=True)

    activity = np.load(activity_clusters, allow_pickle=True)
    Z_act = np.asarray(activity["Z"], dtype=float)
    ind_active = np.asarray(activity["ind_active"], dtype=np.int64)
    activity_best_k = int(activity["best_k"])
    if chosen_k is None:
        chosen_k = activity_best_k
    if chosen_k < k_min or chosen_k > k_max:
        raise ValueError(
            f"chosen_k={chosen_k} is outside the swept range [{k_min}, {k_max}]. "
            "Pass --chosen-k inside the range or widen --k-min/--k-max."
        )

    W_rec_full_base = load_w_rec(model_dir)
    W_rec_full = apply_matrix_variant(W_rec_full_base, matrix_variant)
    W_rec = W_rec_full[np.ix_(ind_active, ind_active)]
    W_pos = np.maximum(W_rec, 0.0)
    W_neg = np.maximum(-W_rec, 0.0)

    rows = []
    labels_struct_by_k = {}
    labels_act_by_k = {}
    null_gammas_by_k = {}

    for k in range(k_min, k_max + 1):
        labels_struct, H_struct = approx_wl_signed(
            W_pos,
            W_neg,
            k,
            n_iter=n_iter,
            seed=seed,
            n_init=n_init,
        )
        labels_act = _activity_labels_for_k(Z_act, k)
        H_act = labels_to_H(labels_act, k)

        gamma_struct = gamma_signed(W_pos, W_neg, H_struct)
        gamma_act = gamma_signed(W_pos, W_neg, H_act)

        null_gammas = np.empty(n_null, dtype=float)
        ari_null = np.empty(n_null, dtype=float)
        for b in range(n_null):
            labels_null = random_labels_same_sizes(labels_act, seed=b)
            H_null = labels_to_H(labels_null, k)
            null_gammas[b] = gamma_signed(W_pos, W_neg, H_null)
            ari_null[b] = adjusted_rand_score(labels_null, labels_struct)

        ari = adjusted_rand_score(labels_act, labels_struct)
        nmi = normalized_mutual_info_score(labels_act, labels_struct)
        refine = refinement_metrics(labels_struct, labels_act, k)
        p_lower = float(np.mean(null_gammas <= gamma_act))

        row = {
            "k": k,
            "gamma_act": gamma_act,
            "gamma_struct": gamma_struct,
            "gamma_null_mean": float(null_gammas.mean()),
            "gamma_null_sd": float(null_gammas.std(ddof=1)),
            "p_lower": p_lower,
            "ari": float(ari),
            "nmi": float(nmi),
            "purity_struct_to_act": refine["purity_struct_to_act"],
            "homogeneity_struct_to_act": refine["homogeneity_struct_to_act"],
            "completeness_struct_to_act": refine["completeness_struct_to_act"],
            "ami_struct_to_act": refine["ami_struct_to_act"],
            "mapped_ari_struct_to_act": refine["mapped_ari_struct_to_act"],
            "ari_null_mean": float(ari_null.mean()),
            "ari_null_sd": float(ari_null.std(ddof=1)),
            "ari_null_p_upper": float(np.mean(ari_null >= ari)),
            "n_struct_clusters": int(len(np.unique(labels_struct))),
            "n_act_clusters": int(len(np.unique(labels_act))),
        }
        rows.append(row)
        labels_struct_by_k[k] = labels_struct
        labels_act_by_k[k] = labels_act
        null_gammas_by_k[k] = null_gammas

    csv_path = output_dir / "signed_input_wl_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    chosen_labels_struct = labels_struct_by_k[chosen_k]
    chosen_labels_act = labels_act_by_k[chosen_k]
    chosen_H_struct = labels_to_H(chosen_labels_struct, chosen_k)
    chosen_B_pos = quotient_matrix(W_pos, chosen_H_struct)
    chosen_B_neg = quotient_matrix(W_neg, chosen_H_struct)
    chosen_refine = refinement_metrics(chosen_labels_struct, chosen_labels_act, chosen_k)

    npz_path = output_dir / "signed_input_wl_sweep.npz"
    np.savez(
        npz_path,
        W_rec=W_rec,
        W_pos=W_pos,
        W_neg=W_neg,
        ind_active=ind_active,
        rows=np.asarray([[row[key] for key in rows[0].keys()] for row in rows], dtype=object),
        row_columns=np.asarray(list(rows[0].keys()), dtype=object),
        labels_struct_by_k=np.asarray([labels_struct_by_k[k] for k in range(k_min, k_max + 1)], dtype=object),
        labels_act_by_k=np.asarray([labels_act_by_k[k] for k in range(k_min, k_max + 1)], dtype=object),
        contingency_chosen=chosen_refine["contingency"],
        majority_mapping_struct_to_act_chosen=chosen_refine["majority_mapping_struct_to_act"],
        B_pos_chosen=chosen_B_pos,
        B_neg_chosen=chosen_B_neg,
        chosen_k=chosen_k,
        activity_best_k=activity_best_k,
        null_gammas_chosen=null_gammas_by_k[chosen_k],
        activity_clusters_path=str(activity_clusters),
        matrix_variant=matrix_variant,
    )

    contingency_path = output_dir / f"contingency_struct_vs_act_k_{chosen_k}.csv"
    with contingency_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["struct_cell/activity_cluster", *range(chosen_k)])
        for i, row in enumerate(chosen_refine["contingency"]):
            writer.writerow([i, *row.tolist()])

    _plot_gamma(rows, output_dir)
    chosen_row = rows[chosen_k - k_min]
    _plot_chosen_null(
        chosen_k,
        chosen_row["gamma_act"],
        chosen_row["gamma_struct"],
        null_gammas_by_k[chosen_k],
        output_dir,
    )
    _plot_ari(rows, output_dir)
    _plot_quotient(chosen_B_pos, chosen_B_neg, chosen_k, output_dir)
    _plot_contingency(chosen_refine["contingency"], chosen_k, output_dir)

    best_ari_row = max(rows, key=lambda row: row["ari"])
    best_mapped_ari_row = max(rows, key=lambda row: row["mapped_ari_struct_to_act"])
    min_gamma_struct_row = min(rows, key=lambda row: row["gamma_struct"])
    min_gamma_act_row = min(rows, key=lambda row: row["gamma_act"])

    summary = {
        "model_dir": str(model_dir),
        "activity_clusters": str(activity_clusters),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "npz_path": str(npz_path),
        "n_units_full": int(W_rec_full.shape[0]),
        "n_units_active": int(len(ind_active)),
        "matrix_variant": matrix_variant,
        "k_min": k_min,
        "k_max": k_max,
        "n_null": n_null,
        "seed": seed,
        "n_iter": n_iter,
        "activity_best_k": activity_best_k,
        "chosen_k": chosen_k,
        "chosen_row": chosen_row,
        "chosen_refinement": {
            key: value
            for key, value in chosen_refine.items()
            if key not in {"contingency", "majority_mapping_struct_to_act"}
        },
        "chosen_contingency_path": str(contingency_path),
        "chosen_majority_mapping_struct_to_act": chosen_refine[
            "majority_mapping_struct_to_act"
        ].tolist(),
        "best_ari_k": int(best_ari_row["k"]),
        "best_ari_row": best_ari_row,
        "best_mapped_ari_k": int(best_mapped_ari_row["k"]),
        "best_mapped_ari_row": best_mapped_ari_row,
        "min_gamma_struct_k": int(min_gamma_struct_row["k"]),
        "min_gamma_struct_row": min_gamma_struct_row,
        "min_gamma_act_k": int(min_gamma_act_row["k"]),
        "min_gamma_act_row": min_gamma_act_row,
    }
    with (output_dir / "signed_input_wl_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run signed input-side approximate WL on W_rec and compare to activity roles."
    )
    parser.add_argument("model_dir")
    parser.add_argument("--activity-clusters", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=40)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--chosen-k", type=int, default=None)
    parser.add_argument(
        "--matrix-variant",
        choices=("original", "transpose", "no_diag"),
        default="original",
    )
    args = parser.parse_args()

    summary = run_signed_input_wl_sweep(
        args.model_dir,
        activity_clusters=args.activity_clusters,
        output_dir=args.output_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        n_iter=args.n_iter,
        seed=args.seed,
        n_init=args.n_init,
        n_null=args.n_null,
        chosen_k=args.chosen_k,
        matrix_variant=args.matrix_variant,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
