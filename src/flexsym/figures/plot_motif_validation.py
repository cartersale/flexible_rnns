"""Validation-gate plots + taxonomy table for recovered dynamical motifs.

For a network's fixed-point directory (results/motif_fixed_points/<net>/):
  1. q-histogram grid (log10 q per condition) -> confirm the keep-threshold and
     the bimodal fixed-point / not-converged split the finder produced.
  2. Jacobian eigenvalue spectra in the complex plane (unit circle) for a few
     representative conditions -> visually confirm line-attractor (lambda~1),
     decision saddle (one lambda>1), point attractor (all inside).
  3. taxonomy CSV + a printed verdict checking the qualitative expectations
     (memory epochs -> line attractors; DM stimulus epochs -> saddles;
     non-DM sensory epochs -> point attractors) as the seeds 0-2 sanity gate.

Run under conda env `myenv`.
  conda run -n myenv python stepnet/plot_motif_validation.py \
      results/motif_fixed_points/LeakyRNN_softplus_diag_256_all_0
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from motif_classify import classify_net, classify_condition, npload  # noqa: E402

# DM / decision tasks expected to show saddles during stimulus/evidence epochs
DM_RULES = {
    "delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm",
}
# match/category comparison tasks also make a categorical decision at stim2
DECISION_RULES = DM_RULES | {"dmsgo", "dmsnogo", "dmcgo", "dmcnogo"}


def plot_q_histograms(net_dir: Path, out_dir: Path):
    npzs = sorted(net_dir.glob("*__*.npz"))
    n = len(npzs)
    if n == 0:
        return
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, npz in zip(axes, npzs):
        d = npload(npz)
        q = np.asarray(d["qstar"], dtype=float)
        q = q[q > 0]
        ax.hist(np.log10(q), bins=40, color="#4C72B0")
        ax.axvline(-8, color="k", ls="--", lw=0.8)   # Q_FP
        ax.axvline(-6, color="grey", ls=":", lw=0.8)  # Q_SLOW
        ax.set_title(npz.stem, fontsize=7)
        ax.set_xlabel("log10 q", fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "q_histograms.png", dpi=130)
    plt.close(fig)


def plot_spectra(net_dir: Path, out_dir: Path, picks=None):
    """Complex-plane eigenvalue spectra for representative conditions."""
    npzs = {p.stem: p for p in net_dir.glob("*__*.npz")}
    if picks is None:
        # try to pick one of each interesting kind if present
        picks = []
        for stem in npzs:
            if stem.endswith("delay1") and "delaygo" in stem:
                picks.append(stem)
            if "delaydm1__stim1" == stem or "contextdelaydm1__stim1" == stem:
                picks.append(stem)
            if "delaygo__stim1" == stem:
                picks.append(stem)
        picks = picks[:6] or list(npzs)[:6]
    theta = np.linspace(0, 2 * np.pi, 200)
    ncol = min(3, len(picks))
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    for ax, stem in zip(axes, picks):
        d = npload(npzs[stem])
        eig = np.asarray(d["eigvals"])
        q = np.asarray(d["qstar"], dtype=float)
        keep = q < 1e-6
        ev = eig[keep].ravel() if keep.any() else eig.ravel()
        ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=0.8)
        ax.scatter(ev.real, ev.imag, s=4, alpha=0.3, color="#C44E52")
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        ax.axvline(1, color="green", ls="--", lw=0.6)
        ax.set_title(stem, fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.tick_params(labelsize=6)
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle("Jacobian eigenvalue spectra (unit circle; green=+1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "eigenvalue_spectra.png", dpi=130)
    plt.close(fig)


def write_table_and_verdict(net_dir: Path, out_dir: Path):
    res = classify_net(net_dir)
    rows = []
    for key, cm in res.items():
        rows.append({
            "rule": cm.rule, "epoch": cm.epoch, "epoch_type": cm.epoch_type,
            "motif": cm.condition_motif, "line_attractor": cm.line_attractor,
            "align_median": round(cm.line_align_median, 3),
            "local_dim_median": round(cm.local_dim_median, 3),
            "global_PR": round(cm.participation_ratio, 3),
            "n_fp": cm.extra["n_fp"], "n_saddle": cm.extra["n_saddle"],
            "n_rotation": cm.extra["n_rotation"],
            "n_point_attractor": cm.extra["n_point_attractor"],
            "q_min": cm.extra["q_min"], "q_med": cm.extra["q_med"],
        })
    with open(out_dir / "motif_taxonomy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- qualitative verdict (seeds 0-2 sanity gate) ----
    mem = [cm for cm in res.values() if cm.epoch_type == "memory"
           and cm.epoch in ("delay1", "delay2")]
    mem_line = [cm for cm in mem if cm.line_attractor]
    dm_stim = [cm for cm in res.values()
               if cm.rule in DECISION_RULES and cm.epoch_type == "stim"]
    dm_saddle = [cm for cm in dm_stim if cm.extra["n_saddle"] > 0]
    nondm_stim = [cm for cm in res.values()
                  if cm.rule not in DECISION_RULES and cm.epoch_type == "stim"]

    print("\n=== VALIDATION VERDICT ===")
    print(f"memory delay epochs that are line attractors: "
          f"{len(mem_line)}/{len(mem)}")
    print(f"decision-task stim epochs with >=1 saddle:    "
          f"{len(dm_saddle)}/{len(dm_stim)}")
    if nondm_stim:
        pa = sum(cm.condition_motif == "point_attractor" for cm in nondm_stim)
        print(f"non-decision stim epochs = point attractors:  "
              f"{pa}/{len(nondm_stim)}")
    ok = (len(mem) and len(mem_line) >= 0.6 * len(mem)) and \
         (len(dm_stim) == 0 or len(dm_saddle) >= 0.5 * len(dm_stim))
    print("GATE:", "PASS" if ok else "REVIEW — taxonomy not as expected")
    return rows, ok


def main():
    net_dir = Path(sys.argv[1])
    out_dir = net_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_q_histograms(net_dir, out_dir)
    plot_spectra(net_dir, out_dir)
    write_table_and_verdict(net_dir, out_dir)
    print(f"\nfigs + table -> {out_dir}")


if __name__ == "__main__":
    main()
