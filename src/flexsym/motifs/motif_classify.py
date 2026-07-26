"""Classify dynamical motifs from saved fixed/slow-point sets.

Consumes the .npz files written by run_motif_fixed_points.py (one per rule x
epoch) and classifies the recovered points into the motif taxonomy, using the
discrete-map Jacobian J = dF/dh spectrum (|lambda| relative to 1):

  point attractor : all |lambda| < 1 - eps          (stable, no escape direction)
  saddle/decision : exactly one |lambda| > 1 + eps  (real) -> unstable eigvec = decision axis
  rotation        : a complex-conjugate pair with |lambda| ~ 1 and |Im| > im_thresh
  line attractor  : a connected 1-D manifold of slow points, one real lambda ~ 1
                    whose eigenvector is tangent to the manifold. Because memory
                    of a *direction* is a RING (closed, curved 1-D manifold), the
                    tangency test is LOCAL: at each slow point the local manifold
                    tangent (leading PC of its k nearest neighbours) must align
                    with the near-unit real eigenvector (|cos| > cos_thresh), and
                    the neighbourhood must be locally 1-D.

Per-point classification is spectral; line-attractor detection is a property of
the whole slow-point *set* within a condition (rule x epoch), matching the
memory-motif geometry (a continuum of marginally stable points).

A per-unit participation vector p_m (>= 0, unit norm) is built for the dominant
motif, from |loadings| so it is invariant to eigenvector sign / ring phase:
  line attractor : mean over slow points of |near-unit real eigenvector|
  saddle         : mean over saddles of |unstable eigenvector| (decision axis)
  rotation       : mean over rotation points of the eigenplane magnitude
                   sqrt(Re^2 + Im^2) per unit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def npload(path):
    """np.load robust to task.py's global monkeypatch (which force-injects
    allow_pickle=True). Works whether or not task.py has been imported."""
    try:
        return np.load(path, allow_pickle=True)
    except TypeError:
        return np.load(path)   # task.py's patched np.load already sets allow_pickle


# --- thresholds (tunable per network from the q-histogram / spectrum) ---------
EPS_STABLE = 0.02      # |lambda| < 1 - EPS_STABLE  => contracting
EPS_UNSTABLE = 0.02    # |lambda| > 1 + EPS_UNSTABLE => expanding
IM_THRESH = 0.05       # |Im(lambda)| above this => genuine rotation
COS_THRESH = 0.8       # |cos| local-tangent vs eigenvector for tangency
Q_FP = 1e-8            # q below this => treat as a true fixed point
Q_SLOW = 1e-6          # q below this => slow point (line-attractor candidate)
K_LOCAL = 12           # neighbours for local-tangent estimation
LOCAL_DIM_MAX = 1.8    # local participation ratio below this => locally 1-D


@dataclass
class ConditionMotifs:
    rule: str
    epoch: str
    epoch_type: str
    n_points: int
    point_labels: np.ndarray            # per-point spectral label (str array)
    n_unstable: np.ndarray              # per-point count of |lambda|>1+eps
    top_abs: np.ndarray                 # per-point max |lambda|
    condition_motif: str                # dominant motif for the condition
    line_attractor: bool
    line_align_median: float            # median |cos(local tangent, slow eigvec)|
    participation_ratio: float          # global PCA dim of the slow-point cloud
    local_dim_median: float             # median local (kNN) participation ratio
    participation: np.ndarray | None = None  # (N,) >=0 unit-norm, dominant motif
    signed_eigvecs: np.ndarray | None = None  # (m, N) signed motif directions
    extra: dict = field(default_factory=dict)


def _top_eig(eigvals):
    """Index of the eigenvalue with largest magnitude."""
    return int(np.argmax(np.abs(eigvals)))


def classify_point(eigvals, eps_stable=EPS_STABLE, eps_unstable=EPS_UNSTABLE,
                   im_thresh=IM_THRESH):
    """Spectral label for a single point from its Jacobian eigenvalues."""
    mag = np.abs(eigvals)
    n_unstable = int(np.sum(mag > 1.0 + eps_unstable))
    top = _top_eig(eigvals)
    top_mag = float(mag[top])
    top_im = float(abs(eigvals[top].imag))

    # rotation: leading mode (or any near-unit mode) is a genuine complex pair
    near_unit = mag > 1.0 - eps_stable
    has_complex_near_unit = np.any(near_unit & (np.abs(eigvals.imag) > im_thresh))

    if n_unstable == 0 and top_mag < 1.0 - eps_stable:
        label = "point_attractor"
    elif has_complex_near_unit and n_unstable <= 2:
        label = "rotation"
    elif n_unstable == 1:
        label = "saddle"
    elif n_unstable == 0 and top_mag <= 1.0 + eps_unstable:
        label = "marginal"          # slow / line-attractor candidate
    else:
        label = f"unstable_{n_unstable}"
    return label, n_unstable, top_mag, top_im


def _real_near_unit_eigvec(eigvals, eigvecs, im_thresh=IM_THRESH):
    """Right eigenvector of the (near-real) eigenvalue closest to +1."""
    real_mask = np.abs(eigvals.imag) <= im_thresh
    if not np.any(real_mask):
        idx = _top_eig(eigvals)
    else:
        cand = np.where(real_mask)[0]
        idx = cand[np.argmin(np.abs(eigvals[cand].real - 1.0))]
    v = np.real(eigvecs[:, idx])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _participation_ratio(singvals):
    s2 = np.asarray(singvals, dtype=float) ** 2
    if s2.sum() == 0:
        return 0.0
    return float((s2.sum() ** 2) / (np.sum(s2 ** 2)))


def _local_tangents(X, k=K_LOCAL):
    """Local manifold tangent + local dimension at each point via kNN PCA.

    Returns tangents (M, N) unit vectors and local_dims (M,) participation
    ratios of each neighbourhood (1 => locally 1-D). Robust to curved manifolds
    (rings), unlike a single global PC.
    """
    M, N = X.shape
    k = min(k, M - 1)
    # pairwise squared distances via ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
    # (avoids the M x M x N temporary; M can be ~1000)
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0, out=d2)
    tangents = np.zeros((M, N))
    local_dims = np.full(M, np.nan)
    for i in range(M):
        nn = np.argsort(d2[i])[: k + 1]           # includes self
        P = X[nn] - X[nn].mean(axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(P, full_matrices=False)
        tangents[i] = vt[0]
        local_dims[i] = _participation_ratio(s)
    return tangents, local_dims


def detect_line_attractor(xstar, Js, cos_thresh=COS_THRESH, im_thresh=IM_THRESH,
                          local_dim_max=LOCAL_DIM_MAX, max_points=300):
    """Is the slow-point set a (possibly curved/ring) 1-D manifold tangent to a
    near-unit eigenvector?

    Returns (is_line, median_align, global_PR, local_dim_median, participation)
    where participation = mean over points of |near-unit real eigenvector| (>=0).
    For efficiency the per-point eigen-decomposition uses a representative
    subsample of the (densely sampled) manifold.
    """
    M = xstar.shape[0]
    N = xstar.shape[1]
    if M < 8:
        return False, 0.0, 0.0, np.nan, None, None
    if M > max_points:
        idx = np.random.default_rng(0).choice(M, size=max_points, replace=False)
        xstar = xstar[idx]
        Js = Js[idx]
        M = max_points
    X = xstar - xstar.mean(axis=0, keepdims=True)
    global_pr = _participation_ratio(np.linalg.svd(X, compute_uv=False))
    tangents, local_dims = _local_tangents(X)

    aligns = np.empty(M)
    absvecs = np.zeros((M, N))
    signed = np.zeros((M, N))
    for i in range(M):
        w, V = np.linalg.eig(Js[i].astype(np.float64))
        u = _real_near_unit_eigvec(w, V, im_thresh)
        if not np.all(np.isfinite(u)):
            u = np.zeros(N)
        absvecs[i] = np.abs(u)
        signed[i] = u
        t = tangents[i]
        denom = (np.linalg.norm(t) * np.linalg.norm(u)) or 1.0
        aligns[i] = abs(float(np.dot(t, u))) / denom
    med = float(np.median(aligns))
    local_dim_med = float(np.nanmedian(local_dims))
    participation = absvecs.mean(axis=0)
    nrm = np.linalg.norm(participation)
    participation = participation / nrm if nrm > 0 else participation
    is_line = (med > cos_thresh) and (local_dim_med < local_dim_max)
    return is_line, med, global_pr, local_dim_med, participation, signed


def classify_condition(npz_path, q_fp=Q_FP, q_slow=Q_SLOW, **kw) -> ConditionMotifs:
    d = npload(npz_path)
    xstar = d["xstar"]
    q = d["qstar"]
    J = d["J_xstar"]
    eigvals = d["eigvals"]
    rule = str(d["rule"])
    epoch = str(d["epoch"])
    etype = str(d["epoch_type"])

    M = xstar.shape[0]
    labels = np.empty(M, dtype=object)
    n_unst = np.zeros(M, dtype=int)
    top_abs = np.zeros(M)
    for i in range(M):
        lab, nu, tm, _ = classify_point(eigvals[i], **kw)
        labels[i], n_unst[i], top_abs[i] = lab, nu, tm

    # true fixed points vs slow points
    fp_mask = q < q_fp
    slow_mask = q < q_slow

    # line-attractor detection on the slow set (curved/ring-aware)
    is_line, med_align, pr, local_dim_med, line_part, line_signed = \
        detect_line_attractor(xstar[slow_mask], J[slow_mask])

    # dominant condition motif (among true fixed points, unless a line attractor)
    fp_labels = labels[fp_mask]
    n_fp = int(fp_mask.sum())
    n_saddle = int(np.sum(fp_labels == "saddle"))
    n_rot = int(np.sum(fp_labels == "rotation"))
    n_point = int(np.sum(fp_labels == "point_attractor"))
    frac = lambda c: c / max(n_fp, 1)
    if is_line:
        cond = "line_attractor"
    elif frac(n_saddle) >= 0.1:
        cond = "saddle_decision"
    elif frac(n_rot) >= 0.1:
        cond = "rotation"
    elif n_point > 0:
        cond = "point_attractor"
    else:
        cond = "mixed"

    # participation vector p_m (>=0, unit norm) + signed motif directions for the
    # subspace/lumpability test. signed vectors are unit-norm per point/instance.
    N = xstar.shape[1]
    participation = None
    signed = None
    if cond == "line_attractor":
        participation = line_part
        signed = line_signed
    elif cond in ("saddle_decision", "point_attractor"):
        want = "saddle" if cond == "saddle_decision" else "point_attractor"
        idx = np.where(fp_mask & (labels == want))[0]
        acc = np.zeros(N)
        vecs = []
        for i in idx:
            w, V = np.linalg.eig(J[i].astype(np.float64))
            v = np.real(V[:, _top_eig(w)])       # expanding axis / slowest mode
            v = v / (np.linalg.norm(v) or 1.0)
            acc += np.abs(v)
            vecs.append(v)
        participation = acc
        signed = np.array(vecs) if vecs else None
    elif cond == "rotation":
        rot_idx = np.where(fp_mask & (labels == "rotation"))[0]
        acc = np.zeros(N)
        vecs = []
        for i in rot_idx:
            w, V = np.linalg.eig(J[i].astype(np.float64))
            v = V[:, _top_eig(w)]
            re = np.real(v); im = np.imag(v)
            acc += np.sqrt(re ** 2 + im ** 2)     # eigenplane magnitude
            for comp in (re, im):                 # both plane axes are directions
                nc = np.linalg.norm(comp)
                if nc > 0:
                    vecs.append(comp / nc)
        participation = acc
        signed = np.array(vecs) if vecs else None
    if participation is not None:
        nrm = np.linalg.norm(participation)
        participation = participation / nrm if nrm > 0 else participation

    return ConditionMotifs(
        rule=rule, epoch=epoch, epoch_type=etype, n_points=M,
        point_labels=labels, n_unstable=n_unst, top_abs=top_abs,
        condition_motif=cond, line_attractor=is_line,
        line_align_median=med_align, participation_ratio=pr,
        local_dim_median=local_dim_med, participation=participation,
        signed_eigvecs=signed,
        extra={"n_fp": n_fp, "n_slow": int(slow_mask.sum()),
               "n_saddle": n_saddle, "n_rotation": n_rot,
               "n_point_attractor": n_point,
               "q_min": float(q.min()), "q_med": float(np.median(q))},
    )


def classify_net(net_dir, **kw):
    """Classify every rule x epoch condition for a network's FP directory."""
    net_dir = Path(net_dir)
    out = {}
    for npz in sorted(net_dir.glob("*__*.npz")):
        try:
            cm = classify_condition(npz, **kw)
        except Exception as e:            # e.g. truncated npz from an interrupted run
            print(f"  [skip] {npz.name}: {type(e).__name__}: {e}")
            continue
        out[f"{cm.rule}__{cm.epoch}"] = cm
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("net_dir")
    args = ap.parse_args()
    res = classify_net(args.net_dir)
    print(f"{'condition':32s} {'etype':7s} {'motif':16s} "
          f"{'align':>6s} {'ldim':>5s} {'gPR':>5s} {'#fp':>5s} {'#sad':>5s} {'#rot':>5s}")
    for k, cm in res.items():
        print(f"{k:32s} {cm.epoch_type:7s} {cm.condition_motif:16s} "
              f"{cm.line_align_median:6.2f} {cm.local_dim_median:5.2f} "
              f"{cm.participation_ratio:5.2f} "
              f"{cm.extra['n_fp']:5d} {cm.extra['n_saddle']:5d} "
              f"{cm.extra['n_rotation']:5d}")
