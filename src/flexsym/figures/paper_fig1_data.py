"""Cache the real data behind Figure 1, panel A.

Pulls, for one trained network:
  * the trained recurrent weights (active units), ordered by structural role
  * real trial inputs and epoch boundaries for the example conditions
  * the recovered fixed/slow points, their local Jacobians, and real trial
    trajectories, projected into a 2-D plane chosen per motif

Conditions were picked by measuring the geometry of the recovered point sets
(see ROWS): a clean open 1-D manifold for the line attractor, a closed one for
the ring, an isolated point for the point attractor, and the unstable-direction
plane for the saddle.

Everything lands in figures/fig1/data/fig1_panelA.npz so the plotting script
never needs the data volume.

    python -m flexsym.figures.paper_fig1_data
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flexsym import paths

#: Example network for panel A. Must be a network in the current RESULTS_ROOT
#: cohort; override with --net. The former default (256 seed 0) is a released
#: Driscoll network and lives in the `released` cohort, not the main paper's grid.
NET = "LeakyRNN_softplus_diag_256_all_5"
OUT = paths.REPO_ROOT / "figures" / "fig1" / "data"
FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
FP_RESP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points_response"


def wl_root(net=None):
    """The net's selected-k WL directory, whatever operating k it was cut at."""
    net = net or NET
    base = paths.RESULTS_ROOT / "signed_input_wl_paper" / net
    hits = sorted(base.glob("transpose_selected_k_*"))
    if not hits:
        raise FileNotFoundError(
            f"no transpose_selected_k_* directory for {net} under {base}")
    return hits[0]


def model_dir(net=None):
    """Resolve the checkpoint across both stores (released vs our own grid)."""
    net = net or NET
    for store in (paths.ORIGINAL_STORE, paths.STAGED_STORE):
        if (store / net).exists():
            return store / net
    raise FileNotFoundError(f"model dir for {net} in neither store")


def cache_path(net=None):
    return paths.CACHE_ROOT / f"{net or NET}_b512.npz"

# (key, rule, epoch, kind, in response dir?, q threshold)
ROWS = [
    ("point",  "reactgo",   "go1",    "point",  True,  1e-8),
    # The q threshold is net-dependent: on seed 5 only 4 points reach q<1e-8 in
    # this condition, while q<1e-6 gives 417 points forming a clean closed ring
    # (radial CV 0.11, max angular gap 18 deg) by the same geometric criterion
    # used to pick these conditions originally.
    ("ring",   "delayanti", "delay1", "ring",   False, 1e-6),
    ("line",   "dmcgo",     "delay1", "line",   False, 1e-6),
    ("saddle", "delaydm1",  "stim1",  "saddle", False, 1e-6),
    # q<1e-6 matches the slow-point threshold used by the rotation analysis in
    # results/motif_rotation, so the angle here is the published one
    ("rot",    "delaygo",   "delay1", "rot",    False, 1e-6),   # paired with go1
]

MAX_TRAJ = 20          # trial trajectories drawn per portrait


def npload(path):
    try:
        return np.load(path, allow_pickle=True)
    except TypeError:      # task.py monkeypatches np.load to force allow_pickle
        return np.load(path)


# ------------------------------------------------------------------- weights

def structural_weights():
    """Trained W_rec over active units, reordered by structural role."""
    d = npload(wl_root() / "signed_input_wl_sweep.npz")
    W = np.asarray(d["W_rec"], dtype=float)
    labels = np.asarray(d["labels_struct_by_k"][0], dtype=int)
    k = int(d["chosen_k"])
    sizes = np.bincount(labels, minlength=k)
    rank = np.empty(k, dtype=int)
    rank[np.argsort(-sizes)] = np.arange(k)
    order = np.argsort(rank[labels], kind="stable")
    return W[np.ix_(order, order)], rank[labels][order], k


# --------------------------------------------------------------------- trials

def trial_inputs(rules):
    """Real trial inputs and epoch boundaries, one example trial per rule."""
    import flexsym.core.task as task_mod

    md = model_dir()
    hp = json.loads((md / "hp.json").read_text())
    hp["rng"] = np.random.RandomState(0)

    out = {}
    for rule in rules:
        trial = task_mod.generate_trials(rule, hp, mode="test")
        x = np.asarray(trial.x)                    # (T, batch, n_input)
        b = x.shape[1] // 2
        epochs = {k: (0 if v[0] is None else int(v[0]),
                      x.shape[0] if v[1] is None else int(v[1]))
                  for k, v in trial.epochs.items()}
        out[rule] = dict(x=x[:, b, :].astype(np.float32), epochs=epochs)
    return out


# ------------------------------------------------------------- trajectories

class Traj:
    """Epoch-resolved hidden activity, [time, batch, unit] -> [batch, time, unit]."""

    def __init__(self):
        self._disk = npload(paths.require(cache_path(), "activity cache"))
        self._fresh = {}

    def get(self, rule, epoch):
        key = f"{rule}|{epoch}"
        if key in self._disk.files:
            H = np.asarray(self._disk[key], dtype=float)
        elif key in self._fresh:
            H = self._fresh[key]
        else:
            H = self._compute(rule, epoch)
            if H is None:
                return None
            self._fresh[key] = H
        return H.transpose(1, 0, 2)                # -> (batch, time, unit)

    def _compute(self, rule, epoch):
        from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch
        md = model_dir()
        print(f"  computing activity for {rule}|{epoch} (not cached)")
        got = compute_hidden_by_epoch(md, [(rule, epoch)], batch_size=256)
        return np.asarray(got[(rule, epoch)], dtype=float) if got else None


def _subsample(H, n=MAX_TRAJ):
    if H is None:
        return None
    idx = np.linspace(0, H.shape[0] - 1, min(n, H.shape[0])).astype(int)
    return H[idx]


# ---------------------------------------------------- plane construction

def _orth2(a, b):
    a = a / (np.linalg.norm(a) or 1.0)
    b = b - a * (a @ b)
    return np.c_[a, b / (np.linalg.norm(b) or 1.0)]


def _pcs(X, n=2):
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return Vt[:n].T


def _flow(J, B, p0, extent, n=22):
    """In-plane linearised velocity v = (J - I)(h - h*), restricted to B."""
    A = B.T @ (J - np.eye(J.shape[0])) @ B
    g = np.linspace(-extent, extent, n)
    P, Q = np.meshgrid(g, g)
    V = A @ np.stack([P.ravel() - p0[0], Q.ravel() - p0[1]])
    return g, V[0].reshape(P.shape), V[1].reshape(P.shape)


# ------------------------------------------------------------ per-motif build

def build_row(key, rule, epoch, kind, response, qthr, traj):
    root = FP_RESP_ROOT if response else FP_ROOT
    d = npload(root / NET / f"{rule}__{epoch}.npz")
    x = np.asarray(d["xstar"], dtype=float)
    q = np.asarray(d["qstar"], dtype=float)
    J = np.asarray(d["J_xstar"], dtype=float)
    eig = np.asarray(d["eigvals"])

    keep = q < qthr
    X, Jk, Ek = x[keep], J[keep], eig[keep]
    H = _subsample(traj.get(rule, epoch))

    # ---- rotation: two epochs, one shared plane ---------------------------
    if kind == "rot":
        d2 = npload(FP_RESP_ROOT / NET / f"{rule}__go1.npz")
        X2 = np.asarray(d2["xstar"], dtype=float)[np.asarray(d2["qstar"]) < qthr]
        mu = np.r_[X, X2].mean(0)
        a, b = _pcs(X, 1)[:, 0], _pcs(X2, 1)[:, 0]
        B = _orth2(a, b)
        H2 = _subsample(traj.get(rule, "go1"))
        return dict(
            pts=(X - mu) @ B, pts_b=(X2 - mu) @ B,
            stable=np.ones(len(X), bool),
            traj=((H - mu) @ B) if H is not None else None,
            traj_b=((H2 - mu) @ B) if H2 is not None else None,
            flow=None, angle=float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))),
            n_fp=len(X), n_fp_b=len(X2),
        )

    # ---- point attractor: one input condition, not the set across them ----
    if kind == "point":
        # the fp closest to where trajectories actually settle
        ends = traj.get(rule, epoch)[:, -1, :]
        i = int(np.argmin(np.linalg.norm(X - ends.mean(0), axis=1)))
        h = X[i]
        # trials that converge to this fp (nearest half of the endpoints)
        dist = np.linalg.norm(ends - h, axis=1)
        sel = np.argsort(dist)[: max(8, len(dist) // 6)]
        H = _subsample(traj.get(rule, epoch)[sel])
        w, V = np.linalg.eig(Jk[i])
        slowest = np.argsort(-np.abs(w))[:2]       # the two least-contracting modes
        B = _orth2(np.real(V[:, slowest[0]]), np.real(V[:, slowest[1]]))
        mu = h
        P = np.zeros((1, 2))
        span = float(np.abs((H - mu) @ B).max()) * 1.1
        g, U, Vf = _flow(Jk[i], B, np.zeros(2), span)
        return dict(pts=P, pts_b=None, stable=np.ones(1, bool),
                    traj=(H - mu) @ B, traj_b=None, flow=(g, U, Vf),
                    angle=np.nan, n_fp=1, n_fp_b=0)

    # ---- saddle: plane = unstable direction x spread of the point set -----
    if kind == "saddle":
        top = np.abs(Ek).max(1)
        i = int(np.argmax(top))
        w, V = np.linalg.eig(Jk[i])
        u = np.real(V[:, int(np.argmax(np.abs(w)))])
        mu = X.mean(0)
        B = _orth2(u, _pcs(X, 1)[:, 0])
        ref = i
    else:                                          # ring / line
        mu = X.mean(0)
        B = _pcs(X, 2)
        P = (X - mu) @ B
        ref = int(np.argmin(np.linalg.norm(P, axis=1)))

    P = (X - mu) @ B
    T = ((H - mu) @ B) if H is not None else None
    stable = np.abs(Ek).max(1) < 1.02
    span = float(np.abs(P).max()) * 1.15
    g, U, Vf = _flow(Jk[ref], B, (X[ref] - mu) @ B, span)
    return dict(pts=P, pts_b=None, stable=stable, traj=T, traj_b=None,
                flow=(g, U, Vf), angle=np.nan, n_fp=len(X), n_fp_b=0)


# ------------------------------------------------------------------------ main

def main():
    global NET
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default=NET,
                    help="example network for panel A (must be in the current cohort)")
    args = ap.parse_args()
    NET = args.net
    print(f"=== panel A data: {NET} ===")
    OUT.mkdir(parents=True, exist_ok=True)
    blob = {}

    W, roles, k = structural_weights()
    blob["W"], blob["roles"], blob["k"] = W.astype(np.float32), roles.astype(np.int32), np.int64(k)
    print(f"W_rec {W.shape}, {k} structural roles")

    for rule, rec in trial_inputs(sorted({r for _, r, _, _, _, _ in ROWS})).items():
        blob[f"x__{rule}"] = rec["x"]
        blob[f"ep__{rule}"] = np.array([(n, a, b) for n, (a, b) in rec["epochs"].items()],
                                       dtype=[("name", "U8"), ("a", "i4"), ("b", "i4")])
        print(f"trial {rule}: x {rec['x'].shape}, epochs {list(rec['epochs'])}")

    traj = Traj()
    for key, rule, epoch, kind, resp, qthr in ROWS:
        m = build_row(key, rule, epoch, kind, resp, qthr, traj)
        for f in ("pts", "pts_b", "traj", "traj_b", "stable"):
            if m[f] is not None:
                blob[f"{key}__{f}"] = np.asarray(m[f], dtype=np.float32)
        if m["flow"] is not None:
            g, U, V = m["flow"]
            blob[f"{key}__flow_g"] = g.astype(np.float32)
            blob[f"{key}__flow_U"] = U.astype(np.float32)
            blob[f"{key}__flow_V"] = V.astype(np.float32)
        blob[f"{key}__meta"] = np.array(
            [(rule, epoch, kind, float(m["angle"]), m["n_fp"], m["n_fp_b"])],
            dtype=[("rule", "U20"), ("epoch", "U12"), ("kind", "U8"),
                   ("angle", "f8"), ("n_fp", "i4"), ("n_fp_b", "i4")])
        nt = 0 if m["traj"] is None else len(m["traj"])
        print(f"motif {key:7s} {rule}|{epoch}: {m['n_fp']} points, {nt} trajectories"
              + (f", rotation {m['angle']:.1f} deg" if kind == "rot" else ""))

    np.savez_compressed(OUT / "fig1_panelA.npz", **blob)
    print(f"wrote {OUT / 'fig1_panelA.npz'}")


if __name__ == "__main__":
    main()
