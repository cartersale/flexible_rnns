"""Per motif-type aggregate mode-energy profiles, in a defensible orthonormal basis, compared
WITHIN each network (never by matching mode index across networks).

Motivation and the corrections this addresses:
  * The quotient Q = (H'H)^-1 H' W H is directed and NON-NORMAL: its eigenvectors are not
    orthogonal, so energy / participation ratio computed in raw eigenvector coordinates can be
    distorted. We therefore decompose motif energy in an ORTHONORMAL basis. Two are built and
    cross-checked: (i) an eigen-ordered orthonormal basis that keeps BOTH real and imaginary
    parts of complex eigenvectors (so a conjugate pair contributes its full 2-D real invariant
    plane) then Gram-Schmidt (QR) IN UNIT SPACE, after lifting through H (since W H = H Q,
    an eigenvector z of Q lifts as H z); (ii) the REAL SCHUR vectors of the same quotient
    written in orthonormal role coordinates, Qtilde = D^-1/2 H' W H D^-1/2, which satisfies
    W S = S Qtilde for S = H D^-1/2 and is therefore the operator that may be lifted through
    S. Both are orthonormal in unit space and their leading columns span the dominant (by
    |lambda|) invariant subspace. We report the
    participation ratio in the orthonormal basis and confirm it is not an artifact of the earlier
    real-parts-only construction.
  * NO cross-network mode correspondence is claimed: mode j of seed 0 is not mode j of seed 1.
    Every profile comparison is done WITHIN a network on that network's own basis; only the
    resulting scalar (cosine, overlap, angle) is aggregated across seeds, with the seed as the
    unit of replication.
  * Repeated samples: many motifs come from the same ring / task family, so a raw pool is
    dominated by the numerous line attractors. Each motif's energy profile is normalised to sum
    one BEFORE pooling (equal weight per motif); a per-task-family average is also reported.

Per network at fixed k we form three aggregate mode-energy vectors (line / point / saddle), each
an equal-weight mean of the per-motif normalised profiles on that network's orthonormal basis, and
compare the three profiles by cosine, correlation, top-3 index overlap, and principal angles
between the top-3 mode spans. Size-matched random quotients give the null for every comparison.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from scipy.linalg import schur, subspace_angles
from scipy.stats import wilcoxon

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians, load_conditions, indicator, quotient, quotient_ortho)

FP = paths.RESULTS_ROOT / "motif_fixed_points"
NET = lambda s: f"LeakyRNN_softplus_diag_256_all_{s}"
TYPES = ["line_attractor", "point_attractor", "saddle_decision"]
SHORT = {"line_attractor": "line", "point_attractor": "point", "saddle_decision": "saddle"}
PAIRS = [("line_attractor", "point_attractor"), ("line_attractor", "saddle_decision"),
         ("point_attractor", "saddle_decision")]


def eig_real_blocks(Q):
    """Real invariant directions of Q in RAW role coordinates, ordered by |lambda|.

    Each real eigenvalue contributes one column; each complex-conjugate pair
    contributes the real and imaginary parts of one member, i.e. its full 2-D
    real invariant plane. The columns are NOT orthonormal (Q is non-normal);
    orthonormalisation happens after the lift to unit space, where it belongs.
    """
    lam, V = np.linalg.eig(Q)
    order = np.argsort(-np.abs(lam))
    cols, used = [], set()
    for i in order:
        if i in used:
            continue
        if abs(lam[i].imag) < 1e-9:
            cols.append(np.real(V[:, i]))
        else:
            cols.append(np.real(V[:, i])); cols.append(np.imag(V[:, i]))
            j = np.argmin(np.abs(lam - np.conj(lam[i])))     # its conjugate partner
            used.add(j)
        used.add(i)
    return np.column_stack(cols)


def lifted_eigbasis(Q, H):
    """Orthonormal unit-space basis from the eigen-directions of the RAW quotient Q.

    W H = H Q, so an eigenvector z of Q lifts as H z. The lift is therefore
    through H, and orthonormality is restored by QR in unit space. (Lifting z
    through S = H D^-1/2 instead would mix coordinate systems: S is the correct
    lift only for eigenvectors of Qtilde = D^1/2 Q D^-1/2.)
    """
    B, _ = np.linalg.qr(H @ eig_real_blocks(Q))
    return B[:, :Q.shape[0]]


def schur_basis(Q):
    """Real Schur vectors, reordered so leading columns carry the largest |lambda|."""
    T, Z = schur(Q, output="real")
    # eigenvalues from 1x1 / 2x2 diagonal blocks
    n = Q.shape[0]; i = 0; blocks = []
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 1e-9:
            ev = np.linalg.eigvals(T[i:i + 2, i:i + 2]); blocks.append((i, 2, max(np.abs(ev)))); i += 2
        else:
            blocks.append((i, 1, abs(T[i, i]))); i += 1
    cols = []
    for start, size, _ in sorted(blocks, key=lambda b: -b[2]):
        cols.extend(range(start, start + size))
    return Z[:, cols]


def lifted(B, S):
    """Lift a k-dim orthonormal basis to units via S (n_active x k, orthonormal cols)."""
    return S @ B                                            # (n_active, k), orthonormal columns


def energy_profiles(P, M):
    """Per-motif normalised energy over the orthonormal modes M (n_active x k).
    Returns (profiles[m, k] summing to 1, participation ratios[m])."""
    C = P @ M                                               # (m, k) coordinates (orthonormal)
    E = C ** 2
    tot = E.sum(1, keepdims=True)
    keep = (tot[:, 0] > 1e-12)
    prof = np.zeros_like(E); prof[keep] = E[keep] / tot[keep]
    pr = np.full(len(P), np.nan)
    pr[keep] = 1.0 / (prof[keep] ** 2).sum(1)              # PR of a sum-1 vector = 1/sum p^2
    return prof, pr, keep


def aggregate(prof, types, rules, target, family_avg=False):
    """Equal-weight mean profile of motifs of `target` type (optionally averaging within task
    family first)."""
    idx = [i for i, t in enumerate(types) if t == target]
    if not idx:
        return None
    if not family_avg:
        return prof[idx].mean(0)
    fam = {}
    for i in idx:
        fam.setdefault(rules[i], []).append(prof[i])
    return np.mean([np.mean(v, 0) for v in fam.values()], 0)


def top3_overlap(a, b):
    return len(set(np.argsort(-a)[:3]) & set(np.argsort(-b)[:3])) / 3.0


def principal_angle_deg(M, a, b):
    """Mean principal angle (deg) between the top-3 mode spans selected by profiles a and b."""
    Sa = M[:, np.argsort(-a)[:3]]; Sb = M[:, np.argsort(-b)[:3]]
    ang = subspace_angles(Sa, Sb)
    return float(np.degrees(ang).mean())


def compare(prof, types, rules, M, family_avg=False):
    g = {t: aggregate(prof, types, rules, t, family_avg) for t in TYPES}
    out = {}
    for a, b in PAIRS:
        if g[a] is None or g[b] is None:
            continue
        key = f"{SHORT[a]}-{SHORT[b]}"
        cos = float(g[a] @ g[b] / (np.linalg.norm(g[a]) * np.linalg.norm(g[b]) + 1e-12))
        out[key] = {"cosine": cos, "corr": float(np.corrcoef(g[a], g[b])[0, 1]),
                    "top3_overlap": top3_overlap(g[a], g[b]),
                    "angle_deg": principal_angle_deg(M, g[a], g[b])}
    # within-type concentration (equal-weighted aggregate), orthonormal basis
    conc = {}
    for t in TYPES:
        if g[t] is not None:
            conc[SHORT[t]] = {"top3": float(np.sort(g[t])[::-1][:3].sum()),
                              "pr": float(1.0 / (g[t] ** 2).sum())}
    return out, conc


def run_net(net, k, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    P = np.array([c["v"] for c in conds])
    types = [c["motif"] for c in conds]; rules = [c["rule"] for c in conds]
    ls = wl["labels_struct_by_k"]
    role = np.asarray(ls[_row_for_k(ls, k)], int)
    H, sizes = indicator(role)
    S = H / np.sqrt(sizes)[None, :]
    Q = quotient(W, H, sizes)                              # raw role coordinates: W H = H Q
    Qt = quotient_ortho(W, H, sizes)                       # orthonormal role coordinates: W S = S Qt

    # defensible bases
    M_eig = lifted_eigbasis(Q, H)                          # eigen-ordered, lifted through H
    M_schur = lifted(schur_basis(Qt), S)                   # Schur of Qtilde, lifted through S
    # artifact check: earlier real-parts-only construction
    lam, V = np.linalg.eig(Q); order = np.argsort(-np.abs(lam))
    M_realonly, _ = np.linalg.qr(H @ np.real(V[:, order]))

    prof_eig, pr_eig, keep = energy_profiles(P, M_eig)
    prof_sch, pr_sch, _ = energy_profiles(P, M_schur)
    _, pr_realonly, _ = energy_profiles(P, M_realonly)

    obs, conc = compare(prof_eig, types, rules, M_eig)
    obs_fam, _ = compare(prof_eig, types, rules, M_eig, family_avg=True)

    # null: random size-matched partitions, own quotient + basis
    null = {key: {m: [] for m in ("cosine", "top3_overlap", "angle_deg")} for key in obs}
    _, szs = np.unique(role, return_counts=True)
    for _ in range(n_perm):
        perm = np.repeat(np.arange(len(szs)), szs); rng.shuffle(perm)
        Hr, sr = indicator(perm)
        Qr = quotient(W, Hr, sr)
        Mr = lifted_eigbasis(Qr, Hr)
        pr_, _, _ = energy_profiles(P, Mr)
        cr, _ = compare(pr_, types, rules, Mr)
        for key in obs:
            if key in cr:
                for m in ("cosine", "top3_overlap", "angle_deg"):
                    null[key][m].append(cr[key][m])

    return {
        "k": int(role.max() + 1),
        "n_by_type": {SHORT[t]: int(sum(x == t for x in types)) for t in TYPES},
        "observed": obs, "observed_familyavg": obs_fam, "concentration": conc,
        "null_mean": {key: {m: float(np.mean(v)) for m, v in null[key].items()} for key in null},
        "pr_check": {"eig_ordered": float(np.nanmedian(pr_eig)),
                     "schur": float(np.nanmedian(pr_sch)),
                     "real_parts_only": float(np.nanmedian(pr_realonly))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9"); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-"); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    rng = np.random.default_rng(0)

    per = {}
    for s in seeds:
        net = NET(s)
        if not (FP / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}"); continue
        per[net] = run_net(net, args.k, args.n_perm, rng)
        r = per[net]
        print(f"[done] {net} k={r['k']} n_by_type={r['n_by_type']}", flush=True)

    print(f"\n=== PR artifact check (median PR over motifs, k={args.k}) ===")
    for basis in ("eig_ordered", "schur", "real_parts_only"):
        print(f"  {basis:16s}: {np.median([per[n]['pr_check'][basis] for n in per]):.2f}")

    print(f"\n=== within-type concentration (equal-weighted, orthonormal eig basis) ===")
    for t in ("line", "point", "saddle"):
        tops = [per[n]['concentration'][t]['top3'] for n in per if t in per[n]['concentration']]
        prs = [per[n]['concentration'][t]['pr'] for n in per if t in per[n]['concentration']]
        print(f"  {t:7s}: top-3 energy {np.median(tops):.3f}  PR {np.median(prs):.2f}  (n_seeds={len(tops)})")

    print(f"\n=== between-type profile similarity (WITHIN network; median over seeds) vs random-quotient null ===")
    print(f"  {'pair':14s} | {'cosine obs/null':>18} {'top3-ovlp obs/null':>20} {'angle(deg) obs/null':>20} {'cos p':>7}")
    for key in PAIRS:
        k2 = f"{SHORT[key[0]]}-{SHORT[key[1]]}"
        rows = [per[n] for n in per if k2 in per[n]["observed"]]
        if not rows: continue
        co = [r["observed"][k2]["cosine"] for r in rows]; cn = [r["null_mean"][k2]["cosine"] for r in rows]
        oo = [r["observed"][k2]["top3_overlap"] for r in rows]; on = [r["null_mean"][k2]["top3_overlap"] for r in rows]
        ao = [r["observed"][k2]["angle_deg"] for r in rows]; an = [r["null_mean"][k2]["angle_deg"] for r in rows]
        p = wilcoxon(co, cn).pvalue if len(co) > 1 else np.nan
        print(f"  {k2:14s} | {np.median(co):>8.3f}/{np.median(cn):<8.3f} {np.median(oo):>9.2f}/{np.median(on):<9.2f} "
              f"{np.median(ao):>9.1f}/{np.median(an):<9.1f} {p:>7.3f}")

    (FP / "motif_type_mode_profiles.json").write_text(json.dumps(per, indent=2, default=float))
    print(f"\nwrote {FP/'motif_type_mode_profiles.json'}")


if __name__ == "__main__":
    main()
