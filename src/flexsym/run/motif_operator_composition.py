"""Does compositionality in function induce compositionality in connectivity?

Driscoll's observation is functional: tasks are built from recurring dynamical
motifs (memory, decision, rotation). This asks the structural question. The role
partition H is derived from W_rec alone and is therefore TASK-FREE, so every
task-conditioned quotient

    Q_c = (H'H)^-1 H' Jbar_c H            c = (rule, epoch)

lives in the SAME k x k space for every condition in a network. Quotient modes
are comparable across tasks without any matching procedure -- which is exactly
what units are not. That is what makes "operator reuse" separable from "node
reuse" here.

Three tests, each with the seed as the unit of replication (10 nets).

  A. OPERATOR REUSE vs NODE REUSE.
     Every cross-rule pair of conditions gets two similarities: a NODE profile
     similarity (squared unit loadings of the motif direction) and an OPERATOR
     profile similarity (mode energy in the orthonormal quotient basis). Ask
     which one better separates same-motif-type pairs from different-type pairs,
     scored by AUC -- rank-based, so the differing dimensions (n_active vs k) do
     not bias the comparison. The user-facing claim "same motif, different
     neurons, same quotient eigenstructure" is the LOW-NODE-OVERLAP cell:
     restricted to pairs in the bottom tertile of node similarity, is same-type
     operator similarity still elevated?

  B. OPERATOR TRANSPORT (the parallelogram form of composition).
     If a functional primitive is a reusable operator, adding it to task X and to
     task Y should be the SAME displacement in quotient space. Each family below
     is one primitive with >= 2 independent instantiations, e.g. anti = {fdgo ->
     fdanti, delaygo -> delayanti}. Statistic: cosine between instantiation
     contrasts, computed on dQ (the quotient displacement) and on dgbar (the
     group-mean gain displacement -- the only task-dependent term in the honest
     predictor Q_pred = (1-a) I + a diag(gbar) Q_struct, so gain additivity IS
     coarse-grained compositionality for that predictor).

  C. DIRECT SUM. For triples where C is functionally A + B -- multidelaydm is
     literally delaydm1 and delaydm2 evidence integrated together -- ask whether
     the motif direction of C lies in span(top modes of Q_A) + span(top modes of
     Q_B), conditional on containment, against mismatched-pair nulls. Reported
     with the contribution balance, so "C ~ A alone" does not count as composed.

Nulls throughout: size-matched random partitions with the quotient RECOMPUTED
(so the null has its own coarse-grained dynamics), and the activity partition at
matched k (is this connectivity-specific, or would any functional grouping do?).
Both are mandatory here -- the cross-task motif reuse result in claim 2 died
against exactly the first of these.

    conda run -n myenv python -m flexsym.run.motif_operator_composition --seeds 0-9
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import wilcoxon

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    indicator,
    load_conditions,
    quotient,
)
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.motif_type_mode_profiles import lifted_eigbasis
from flexsym.structural.operating_k import OPK, model_name

FP = paths.RESULTS_ROOT / "motif_fixed_points"
TOP_M = 3           # modes spanned per condition in test C
N_PERM = 100


# --------------------------------------------------------------------------- #
# families and triples
#
# Only fix1 / stim1 / delay1 / stim2 / delay2 conditions exist in the fixed-point
# cache -- go1 carries no fixed points -- so every contrast below is built from
# those. A family is (primitive, epoch, [(base_rule, plus_rule), ...]); both
# instantiations of a family differ only in the task context the primitive is
# added to, so their contrasts should coincide if the primitive is an operator.
# --------------------------------------------------------------------------- #
FAMILIES = [
    ("anti", "stim1", [("fdgo", "fdanti"), ("delaygo", "delayanti")]),
    ("anti", "fix1", [("fdgo", "fdanti"), ("reactgo", "reactanti")]),
    ("nogo", "stim1", [("dmsgo", "dmsnogo"), ("dmcgo", "dmcnogo")]),
    ("nogo", "delay1", [("dmsgo", "dmsnogo"), ("dmcgo", "dmcnogo")]),
    ("nogo", "fix1", [("dmsgo", "dmsnogo"), ("dmcgo", "dmcnogo")]),
    ("category", "stim1", [("dmsgo", "dmcgo"), ("dmsnogo", "dmcnogo")]),
    ("category", "delay1", [("dmsgo", "dmcgo"), ("dmsnogo", "dmcnogo")]),
    ("context", "stim1", [("delaydm1", "contextdelaydm1"),
                          ("delaydm2", "contextdelaydm2")]),
    ("context", "delay1", [("delaydm1", "contextdelaydm1"),
                           ("delaydm2", "contextdelaydm2")]),
    ("context", "stim2", [("delaydm1", "contextdelaydm1"),
                          ("delaydm2", "contextdelaydm2")]),
    ("modality", "stim1", [("delaydm1", "delaydm2"),
                           ("contextdelaydm1", "contextdelaydm2")]),
    ("modality", "delay1", [("delaydm1", "delaydm2"),
                            ("contextdelaydm1", "contextdelaydm2")]),
]

# (C, A, B): C is functionally A composed with B.
TRIPLES = [
    ("multidelaydm__stim1", "delaydm1__stim1", "delaydm2__stim1"),
    ("multidelaydm__stim2", "delaydm1__stim2", "delaydm2__stim2"),
    ("delaydm1__delay1", "delaygo__delay1", "delaydm1__stim1"),
    ("delaydm2__delay1", "delaygo__delay1", "delaydm2__stim1"),
    ("delayanti__delay1", "delaygo__delay1", "fdanti__stim1"),
]


def key(rule, epoch):
    return f"{rule}__{epoch}"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a.ravel() @ b.ravel() / (na * nb)) if na > 0 and nb > 0 else np.nan


def auc(pos, neg):
    """Mann-Whitney AUC: P(a random positive scores above a random negative)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1.0
    # average ranks for ties
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def profiles_for(P, M):
    """Normalised energy of each motif direction over the orthonormal modes M."""
    C = P @ M
    E = C ** 2
    tot = E.sum(1, keepdims=True)
    out = np.zeros_like(E)
    ok = tot[:, 0] > 1e-12
    out[ok] = E[ok] / tot[ok]
    return out, ok


def pair_cosines(prof, ok):
    """Full symmetric cosine matrix of the (already non-negative) profiles."""
    n = len(prof)
    S = np.full((n, n), np.nan)
    nrm = np.linalg.norm(prof, axis=1)
    for i in range(n):
        if not ok[i] or nrm[i] == 0:
            continue
        for j in range(i + 1, n):
            if not ok[j] or nrm[j] == 0:
                continue
            S[i, j] = S[j, i] = prof[i] @ prof[j] / (nrm[i] * nrm[j])
    return S


def type_auc(S, types, rules, mask=None):
    """AUC separating same-motif-type from different-type CROSS-RULE pairs."""
    pos, neg = [], []
    n = len(types)
    for i in range(n):
        for j in range(i + 1, n):
            if rules[i] == rules[j] or not np.isfinite(S[i, j]):
                continue
            if mask is not None and not mask[i, j]:
                continue
            (pos if types[i] == types[j] else neg).append(S[i, j])
    return auc(pos, neg), len(pos), len(neg)


def basis_and_quotients(labels, W, conds):
    """Orthonormal lifted quotient basis + per-condition Q, gbar for a partition."""
    H, sizes = indicator(labels)
    S = H / np.sqrt(sizes)[None, :]
    Qs = quotient(W, H, sizes)
    M = lifted_eigbasis(Qs, H)          # W H = H Qs, so eigen-directions lift through H
    per = {}
    for c in conds:
        per[c["key"]] = {
            "Q": quotient(c["Jb"], H, sizes),
            "gbar": (H.T @ c["g"]) / sizes,
        }
    return M, per, H, sizes


# --------------------------------------------------------------------------- #
# test B: operator transport
# --------------------------------------------------------------------------- #
def transport(per, present):
    """Per family: cosine between the contrasts of its instantiations."""
    out = {}
    for prim, epoch, pairs in FAMILIES:
        have = [(key(a, epoch), key(b, epoch)) for a, b in pairs
                if key(a, epoch) in present and key(b, epoch) in present]
        if len(have) < 2:
            continue
        dQ = [per[b]["Q"] - per[a]["Q"] for a, b in have]
        dg = [per[b]["gbar"] - per[a]["gbar"] for a, b in have]
        cq = [cos(dQ[i], dQ[j]) for i in range(len(have))
              for j in range(i + 1, len(have))]
        cg = [cos(dg[i], dg[j]) for i in range(len(have))
              for j in range(i + 1, len(have))]
        out[f"{prim}@{epoch}"] = {"dQ": float(np.nanmean(cq)),
                                  "dgbar": float(np.nanmean(cg)),
                                  "n_inst": len(have)}
    return out


def transport_unit_level(conds, present):
    """The same contrasts at the unit level: cosine between dJbar and dg."""
    byk = {c["key"]: c for c in conds}
    out = {}
    for prim, epoch, pairs in FAMILIES:
        have = [(key(a, epoch), key(b, epoch)) for a, b in pairs
                if key(a, epoch) in present and key(b, epoch) in present]
        if len(have) < 2:
            continue
        dJ = [byk[b]["Jb"] - byk[a]["Jb"] for a, b in have]
        dg = [byk[b]["g"] - byk[a]["g"] for a, b in have]
        cj = [cos(dJ[i], dJ[j]) for i in range(len(have))
              for j in range(i + 1, len(have))]
        cg = [cos(dg[i], dg[j]) for i in range(len(have))
              for j in range(i + 1, len(have))]
        out[f"{prim}@{epoch}"] = {"dJ": float(np.nanmean(cj)),
                                  "dg": float(np.nanmean(cg))}
    return out


def transport_mismatched(per, present, rng, n=200):
    """Null for B: cosines between contrasts drawn from DIFFERENT families."""
    items = []
    for prim, epoch, pairs in FAMILIES:
        for a, b in pairs:
            if key(a, epoch) in present and key(b, epoch) in present:
                items.append((f"{prim}@{epoch}", key(a, epoch), key(b, epoch)))
    if len(items) < 2:
        return np.nan, np.nan
    cq, cg = [], []
    for _ in range(n):
        i, j = rng.choice(len(items), 2, replace=False)
        if items[i][0] == items[j][0]:
            continue
        _, a1, b1 = items[i]
        _, a2, b2 = items[j]
        cq.append(cos(per[b1]["Q"] - per[a1]["Q"], per[b2]["Q"] - per[a2]["Q"]))
        cg.append(cos(per[b1]["gbar"] - per[a1]["gbar"],
                      per[b2]["gbar"] - per[a2]["gbar"]))
    return float(np.nanmean(cq)), float(np.nanmean(cg))


# --------------------------------------------------------------------------- #
# test C: direct sum
# --------------------------------------------------------------------------- #
def top_span(Q, H, sizes, m):
    """Lifted top-m invariant subspace of Q, orthonormalised (unit coordinates)."""
    lam, V = np.linalg.eig(Q)
    order = np.argsort(-np.abs(lam))[:m]
    U = H @ np.real(V[:, order])
    if np.any(np.abs(np.imag(V[:, order])) > 1e-12):
        U = np.hstack([U, H @ np.imag(V[:, order])])
    B, R = np.linalg.qr(U)
    return B[:, np.abs(np.diag(R)) > 1e-10]


def contained(v, B, denom):
    if B.shape[1] == 0 or denom <= 0:
        return np.nan
    return float(np.linalg.norm(B.T @ v) / denom)


def direct_sum(conds, per, H, sizes, rng, n_null=100):
    byk = {c["key"]: c for c in conds}
    spans = {k: top_span(per[k]["Q"], H, sizes, TOP_M) for k in per}
    out = {}
    for kc, ka, kb in TRIPLES:
        if not all(x in byk for x in (kc, ka, kb)):
            continue
        v = byk[kc]["v"]
        gs = H.T @ v
        ph = float(np.sqrt(np.sum(gs ** 2 / sizes)))          # ||P_H v||
        Ba, Bb = spans[ka], spans[kb]
        Bab = np.linalg.qr(np.hstack([Ba, Bb]))[0][:, :Ba.shape[1] + Bb.shape[1]]
        rho = contained(v, Bab, ph)
        ra, rb = contained(v, Ba, ph), contained(v, Bb, ph)
        # null: mismatched operand pairs of the same total dimension
        pool = [k for k in spans if k not in (kc, ka, kb)]
        nulls = []
        for _ in range(n_null):
            i, j = rng.choice(len(pool), 2, replace=False)
            Bn = np.linalg.qr(np.hstack([spans[pool[i]], spans[pool[j]]]))[0]
            nulls.append(contained(v, Bn[:, :Bab.shape[1]], ph))
        mu, sd = np.nanmean(nulls), np.nanstd(nulls)
        out[kc] = {"A": ka, "B": kb, "rho_AB": rho, "rho_A": ra, "rho_B": rb,
                   "balance": float(min(ra, rb) / max(ra, rb)) if max(ra, rb) > 0 else np.nan,
                   "gain_over_best_single": float(rho - max(ra, rb)),
                   "null": float(mu),
                   "z": float((rho - mu) / sd) if sd > 0 else np.nan,
                   "angle_AB_deg": float(np.degrees(subspace_angles(Ba, Bb)).mean())}
    return out


# --------------------------------------------------------------------------- #
# per network
# --------------------------------------------------------------------------- #
def run_net(net, k, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    if len(conds) < 8:
        return None
    P = np.array([c["v"] for c in conds])
    types = [c["motif"] for c in conds]
    rules = [c["rule"] for c in conds]
    present = {c["key"] for c in conds}

    ls, la = wl["labels_struct_by_k"], wl["labels_act_by_k"]
    r = _row_for_k(ls, k)
    role = np.asarray(ls[r], int)
    actl = np.asarray(la[r], int)

    M, per, H, sizes = basis_and_quotients(role, W, conds)
    M_act, per_act, _, _ = basis_and_quotients(actl, W, conds)

    # ---- test A ----------------------------------------------------------- #
    op_prof, ok = profiles_for(P, M)
    op_act, ok_a = profiles_for(P, M_act)
    node_prof = P ** 2
    node_ok = np.ones(len(P), bool)

    S_op = pair_cosines(op_prof, ok)
    S_act = pair_cosines(op_act, ok_a)
    S_node = pair_cosines(node_prof, node_ok)

    a_op, n_pos, n_neg = type_auc(S_op, types, rules)
    a_node, _, _ = type_auc(S_node, types, rules)
    a_actp, _, _ = type_auc(S_act, types, rules)

    # low-node-overlap cell: bottom tertile of node similarity
    finite = S_node[np.isfinite(S_node)]
    thr = np.quantile(finite, 1 / 3) if len(finite) else np.nan
    low = np.isfinite(S_node) & (S_node <= thr)
    a_op_low, npos_low, nneg_low = type_auc(S_op, types, rules, mask=low)
    a_node_low, _, _ = type_auc(S_node, types, rules, mask=low)

    # random size-matched partitions, quotient + basis recomputed
    _, szs = np.unique(role, return_counts=True)
    null_auc = []
    for _ in range(n_perm):
        perm = np.repeat(np.arange(len(szs)), szs)
        rng.shuffle(perm)
        Mr, _, _, _ = basis_and_quotients(perm, W, [])
        pr, okr = profiles_for(P, Mr)
        null_auc.append(type_auc(pair_cosines(pr, okr), types, rules)[0])
    mu, sd = np.nanmean(null_auc), np.nanstd(null_auc)

    # eigenvalue signature: |lambda| of the quotient mode carrying most energy
    lam_abs = np.sort(np.abs(np.linalg.eigvals(quotient(W, H, sizes))))[::-1]
    lam_mode = [lam_abs[int(np.argmax(op_prof[i]))] if ok[i] else np.nan
                for i in range(len(P))]
    lam_line = [lam_mode[i] for i in range(len(P)) if types[i] == "line_attractor"]
    lam_sad = [lam_mode[i] for i in range(len(P)) if types[i] == "saddle_decision"]

    # ---- tests B and C ---------------------------------------------------- #
    tb = transport(per, present)
    tb_act = transport(per_act, present)
    tb_unit = transport_unit_level(conds, present)
    mis_q, mis_g = transport_mismatched(per, present, rng)
    tc = direct_sum(conds, per, H, sizes, rng)

    return {
        "k": int(np.unique(role).size),
        "n_cond": len(conds),
        "A": {"auc_operator": a_op, "auc_node": a_node, "auc_activity": a_actp,
              "auc_null_mean": float(mu),
              "z_vs_random_partition": float((a_op - mu) / sd) if sd > 0 else np.nan,
              "n_pos": n_pos, "n_neg": n_neg,
              "low_node": {"auc_operator": a_op_low, "auc_node": a_node_low,
                           "n_pos": npos_low, "n_neg": nneg_low},
              "lambda_line_median": float(np.nanmedian(lam_line)) if lam_line else np.nan,
              "lambda_saddle_median": float(np.nanmedian(lam_sad)) if lam_sad else np.nan,
              "lambda_auc_line_vs_saddle": auc(lam_line, lam_sad)},
        "B": {"quotient": tb, "activity": tb_act, "unit": tb_unit,
              "mismatched_dQ": mis_q, "mismatched_dgbar": mis_g},
        "C": tc,
    }


# --------------------------------------------------------------------------- #
def med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else np.nan


def wilc(a, b):
    pairs = [(x, y) for x, y in zip(a, b)
             if np.isfinite(x) and np.isfinite(y) and x != y]
    if len(pairs) < 5:
        return np.nan
    return float(wilcoxon([p[0] for p in pairs], [p[1] for p in pairs]).pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--out", default="operator_composition.json")
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    rng = np.random.default_rng(0)

    per_net = {}
    for s in seeds:
        net = model_name(s)
        if not (FP / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}")
            continue
        r = run_net(net, OPK[s], args.n_perm, rng)
        if r:
            per_net[net] = r
            print(f"[done] seed {s}  k={r['k']}  n_cond={r['n_cond']}  "
                  f"AUC op={r['A']['auc_operator']:.3f} node={r['A']['auc_node']:.3f}",
                  flush=True)
    if not per_net:
        return
    V = list(per_net.values())

    print("\n" + "=" * 78)
    print("A. OPERATOR REUSE vs NODE REUSE")
    print("   AUC separating same-motif-type from different-type CROSS-RULE pairs")
    print("=" * 78)
    op = [v["A"]["auc_operator"] for v in V]
    nd = [v["A"]["auc_node"] for v in V]
    ac = [v["A"]["auc_activity"] for v in V]
    nl = [v["A"]["auc_null_mean"] for v in V]
    print(f"  {'representation':28s} {'median AUC':>11}  {'nets > node':>12}")
    print(f"  {'quotient modes (role)':28s} {med(op):>11.3f}  "
          f"{sum(o > n for o, n in zip(op, nd)):>7}/{len(V)}")
    print(f"  {'unit loadings (node)':28s} {med(nd):>11.3f}  {'--':>12}")
    print(f"  {'quotient modes (activity k)':28s} {med(ac):>11.3f}  "
          f"{sum(a > n for a, n in zip(ac, nd)):>7}/{len(V)}")
    print(f"  {'random partition null':28s} {med(nl):>11.3f}  "
          f"{sum(x > n for x, n in zip(nl, nd)):>7}/{len(V)}")
    print(f"\n  operator - node : median {med([o - n for o, n in zip(op, nd)]):+.3f}"
          f"   p={wilc(op, nd):.4f}   positive in "
          f"{sum(o > n for o, n in zip(op, nd))}/{len(V)}")
    print(f"  operator - random-partition quotient : median "
          f"{med([o - n for o, n in zip(op, nl)]):+.3f}   p={wilc(op, nl):.4f}"
          f"   positive in {sum(o > n for o, n in zip(op, nl))}/{len(V)}")
    print(f"  operator - activity quotient : median "
          f"{med([o - a for o, a in zip(op, ac)]):+.3f}   p={wilc(op, ac):.4f}"
          f"   positive in {sum(o > a for o, a in zip(op, ac))}/{len(V)}")

    lo = [v["A"]["low_node"]["auc_operator"] for v in V]
    ln = [v["A"]["low_node"]["auc_node"] for v in V]
    print(f"\n  --- LOW node-overlap pairs only (bottom tertile of node similarity) ---")
    print(f"  operator AUC {med(lo):.3f}   node AUC {med(ln):.3f}   "
          f"diff {med([a - b for a, b in zip(lo, ln)]):+.3f}  p={wilc(lo, ln):.4f}"
          f"   op>0.5 in {sum(x > 0.5 for x in lo)}/{len(V)}")

    print(f"\n  eigenstructure: |lambda| of the dominant quotient mode")
    print(f"    line attractors  median {med([v['A']['lambda_line_median'] for v in V]):.3f}")
    print(f"    saddle/decision  median {med([v['A']['lambda_saddle_median'] for v in V]):.3f}")
    la = [v["A"]["lambda_auc_line_vs_saddle"] for v in V]
    print(f"    AUC(line vs saddle) median {med(la):.3f}   >0.5 in "
          f"{sum(x > 0.5 for x in la if np.isfinite(x))}/{len(V)}")

    print("\n" + "=" * 78)
    print("B. OPERATOR TRANSPORT -- is a primitive the same displacement")
    print("   in quotient space wherever it is added?")
    print("=" * 78)
    fams = sorted({f for v in V for f in v["B"]["quotient"]})
    print(f"  {'family':18s} {'nets':>5} {'cos dQ':>8} {'cos dgbar':>10} "
          f"{'cos dJ (unit)':>14} {'cos dg (unit)':>14}")
    for f in fams:
        q = [v["B"]["quotient"][f]["dQ"] for v in V if f in v["B"]["quotient"]]
        g = [v["B"]["quotient"][f]["dgbar"] for v in V if f in v["B"]["quotient"]]
        uj = [v["B"]["unit"][f]["dJ"] for v in V if f in v["B"]["unit"]]
        ug = [v["B"]["unit"][f]["dg"] for v in V if f in v["B"]["unit"]]
        print(f"  {f:18s} {len(q):>5} {med(q):>8.3f} {med(g):>10.3f} "
              f"{med(uj):>14.3f} {med(ug):>14.3f}")
    allq = [med([v["B"]["quotient"][f]["dQ"] for f in v["B"]["quotient"]]) for v in V]
    allg = [med([v["B"]["quotient"][f]["dgbar"] for f in v["B"]["quotient"]]) for v in V]
    alla = [med([v["B"]["activity"][f]["dQ"] for f in v["B"]["activity"]]) for v in V]
    mq = [v["B"]["mismatched_dQ"] for v in V]
    mg = [v["B"]["mismatched_dgbar"] for v in V]
    print(f"\n  pooled over families (per-net median, then median over nets):")
    print(f"    same-family  cos dQ {med(allq):.3f}   cos dgbar {med(allg):.3f}")
    print(f"    MISMATCHED   cos dQ {med(mq):.3f}   cos dgbar {med(mg):.3f}")
    print(f"    same - mismatched (dQ)    {med([a - b for a, b in zip(allq, mq)]):+.3f}"
          f"  p={wilc(allq, mq):.4f}  positive in "
          f"{sum(a > b for a, b in zip(allq, mq))}/{len(V)}")
    print(f"    same - mismatched (dgbar) {med([a - b for a, b in zip(allg, mg)]):+.3f}"
          f"  p={wilc(allg, mg):.4f}  positive in "
          f"{sum(a > b for a, b in zip(allg, mg))}/{len(V)}")
    print(f"    role - activity partition (dQ) {med([a - b for a, b in zip(allq, alla)]):+.3f}"
          f"  p={wilc(allq, alla):.4f}  positive in "
          f"{sum(a > b for a, b in zip(allq, alla))}/{len(V)}")

    print("\n" + "=" * 78)
    print("C. DIRECT SUM -- does the composed condition's motif direction live in")
    print("   span(top modes of A) + span(top modes of B)?")
    print("=" * 78)
    tri = sorted({t for v in V for t in v["C"]})
    print(f"  {'C = A + B':24s} {'nets':>5} {'rho_AB':>7} {'rho_A':>7} {'rho_B':>7} "
          f"{'null':>7} {'z':>6} {'bal':>6} {'gain':>7}")
    for t in tri:
        rows = [v["C"][t] for v in V if t in v["C"]]
        print(f"  {t:24s} {len(rows):>5} "
              f"{med([r['rho_AB'] for r in rows]):>7.3f} "
              f"{med([r['rho_A'] for r in rows]):>7.3f} "
              f"{med([r['rho_B'] for r in rows]):>7.3f} "
              f"{med([r['null'] for r in rows]):>7.3f} "
              f"{med([r['z'] for r in rows]):>6.2f} "
              f"{med([r['balance'] for r in rows]):>6.2f} "
              f"{med([r['gain_over_best_single'] for r in rows]):>+7.3f}")
    print("    rho = ||P_span v_C|| / ||P_H v_C||   (conditional on containment)")
    print("    bal = min(rho_A,rho_B)/max(...)      gain = rho_AB - max(rho_A,rho_B)")

    out = FP / args.out
    out.write_text(json.dumps(per_net, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
