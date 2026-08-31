"""NumPy forward simulation of a trained LeakyRNN, with per-unit gain modulation.

The TF graph in `core.network` bakes its nonlinearity into the cell, so a gain
change on a subset of units cannot be imposed by assigning weights -- it needs
the cell itself. Rather than rebuild the graph, the trained cell is six lines of
numpy:

    u_t = W x_{t}   + M h_{t-1} + b + noise        (M = w_in[n_input:].T)
    h_t = (1-alpha) h_{t-1} + alpha * gain * f(u_t)

with f = softplus and noise ~ N(0, sqrt(2/alpha) * sigma_rec), matching
`LeakyRNNCell.call` exactly (the noise is pre-activation there too).

`gain` is the intervention. With gain_i = 1+gamma on a set of units, the
Jacobian of the map becomes

    J = (1-alpha) I + alpha diag(gain * f'(u)) M

i.e. exactly the perturbation d g_i that the eigenvalue susceptibility S_ev is
about, imposed on the running network rather than on a stored Jacobian. `bias`
is the alternative, physiological intervention: a constant injected current,
whose induced gain change is state-dependent.

Validated against the network's own training log (see `selftest`): intact
per-rule performance reproduces `log.json` to within trial noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flexsym import paths


def net_dir(net: str) -> Path:
    """Locate a trained network in either store."""
    for store in (paths.STAGED_STORE, paths.ORIGINAL_STORE):
        d = Path(store) / net
        if (d / "model_params.npz").exists():
            return d
    raise FileNotFoundError(f"no model_params.npz for {net} under "
                            f"{paths.STAGED_STORE} or {paths.ORIGINAL_STORE}")


ACT = {
    "softplus": lambda u: np.logaddexp(0.0, u),
    "relu": lambda u: np.maximum(u, 0.0),
    "tanh": np.tanh,
    "retanh": lambda u: np.tanh(np.maximum(u, 0.0)),
    "power": lambda u: np.maximum(u, 0.0) ** 2,
}
DACT = {
    "softplus": lambda u: 1.0 / (1.0 + np.exp(-u)),
    "relu": lambda u: (u > 0).astype(float),
    "tanh": lambda u: 1.0 - np.tanh(u) ** 2,
    "retanh": lambda u: (u > 0) * (1.0 - np.tanh(np.maximum(u, 0.0)) ** 2),
    "power": lambda u: 2.0 * np.maximum(u, 0.0),
}


class NpNet:
    """A trained LeakyRNN, ready to run forward under an arbitrary gain vector."""

    def __init__(self, net: str, model_dir: Path | None = None, dtype=np.float64):
        d = Path(model_dir) if model_dir is not None else net_dir(net)
        self.dir = d
        self.name = net
        self.dtype = np.dtype(dtype)
        hp = json.loads((d / "hp.json").read_text())
        p = np.load(d / "model_params.npz")
        n_in, n_rnn = int(hp["n_input"]), int(hp["n_rnn"])
        kern = np.asarray(p["w_in"], float)              # (n_in + n_rnn, n_rnn)
        self.W_in = kern[:n_in].T                        # (n_rnn, n_in)
        self.M = kern[n_in:].T                           # (n_rnn, n_rnn), row = postsyn
        self.b = np.asarray(p["b_in"], float)
        self.W_out = np.asarray(p["w_out"], float)       # (n_rnn, n_out)
        self.b_out = np.asarray(p["b_out"], float)
        self.alpha = float(hp["dt"]) / float(hp["tau"])
        self.sigma = np.sqrt(2.0 / self.alpha) * float(hp["sigma_rec"])
        self.f = ACT[hp["activation"]]
        self.df = DACT[hp["activation"]]
        for a in ("W_in", "M", "b", "W_out", "b_out"):
            setattr(self, a, np.asarray(getattr(self, a), self.dtype))
        self.n_rnn = n_rnn
        self.hp = hp
        self.rules = list(hp.get("rule_trains") or hp["rules"])

    # -- the intervention ---------------------------------------------------
    def gain_vector(self, units=None, gamma: float = 0.0) -> np.ndarray:
        g = np.ones(self.n_rnn, self.dtype)
        if units is not None and gamma != 0.0:
            g[np.asarray(units, int)] = 1.0 + gamma
        return g

    def bias_vector(self, units=None, current: float = 0.0) -> np.ndarray:
        b = np.zeros(self.n_rnn, self.dtype)
        if units is not None and current != 0.0:
            b[np.asarray(units, int)] = current
        return b

    # -- forward ------------------------------------------------------------
    def run(self, x, gain=None, bias=None, rng=None, noise=True,
            return_u=False, track_gain=False, keep_h=True):
        """x: (T, B, n_in) -> y_hat (T, B, n_out), h (T, B, n_rnn).

        `track_gain` additionally returns the trial-averaged EFFECTIVE gain per
        unit, gain_i * f'(u_i), which is the quantity S_ev is a derivative in.
        For the direct intervention it is (1+gamma) f'(u) by construction; for
        current injection it is whatever the shifted operating point produces.
        """
        T, B, _ = x.shape
        x = np.asarray(x, self.dtype)
        gain = np.ones(self.n_rnn, self.dtype) if gain is None else np.asarray(gain, self.dtype)
        bias = self.b if bias is None else self.b + bias
        a = self.alpha
        h = np.zeros((B, self.n_rnn), self.dtype)
        H = np.empty((T, B, self.n_rnn), self.dtype) if keep_h else None
        Y = np.empty((T, B, self.W_out.shape[1]), self.dtype)
        U = np.empty((T, B, self.n_rnn)) if return_u else None
        xin = x @ self.W_in.T                            # (T, B, n_rnn)
        gsum = np.zeros(self.n_rnn, self.dtype) if track_gain else None
        if noise and self.sigma > 0:
            rng = np.random.default_rng(0) if rng is None else rng
            eta = (rng.standard_normal((T, B, self.n_rnn)) * self.sigma).astype(self.dtype)
        else:
            eta = np.zeros((1, 1, 1))
        for t in range(T):
            u = xin[t] + h @ self.M.T + bias + (eta[t] if eta.shape[0] > 1 else 0.0)
            h = (1 - a) * h + a * gain * self.f(u)
            if keep_h:
                H[t] = h
            Y[t] = h @ self.W_out + self.b_out
            if return_u:
                U[t] = u
            if track_gain:
                gsum += (gain * self.df(u)).mean(0)
        out = (Y, H)
        if return_u:
            out = out + (U,)
        if track_gain:
            out = out + (gsum / T,)
        return out


def perf_by_rule(net: NpNet, trials, gain=None, bias=None, seed=0,
                 track_gain=False):
    """Per-rule performance under one intervention, on a FIXED trial set.

    `trials` is {rule: Trial}; the same trials and the same recurrent-noise
    stream are used for every condition, so deficits are paired differences and
    carry no between-condition trial variance.
    """
    from flexsym.core.network import get_perf
    out, gains = {}, {}
    for rule, tr in trials.items():
        rng = np.random.default_rng(seed)
        r = net.run(tr.x, gain=gain, bias=bias, rng=rng, track_gain=track_gain,
                    keep_h=False)
        out[rule] = float(np.mean(get_perf(r[0], tr.y_loc)))
        if track_gain:
            gains[rule] = r[-1]
    return (out, gains) if track_gain else out


def make_trials(net: NpNet, rules=None, batch_size=200, seed=0):
    """One fixed batch of test trials per rule."""
    from flexsym.core.task import generate_trials
    hp = dict(net.hp)
    hp["rng"] = np.random.RandomState(seed)
    np.random.seed(seed)   # some task builders draw from the global stream too
    return {r: generate_trials(r, hp, "random", batch_size=batch_size)
            for r in (rules or net.rules)}


def selftest(net="LeakyRNN_softplus_diag_256_all_3", batch_size=200):
    """Intact numpy performance vs the network's own training log."""
    n = NpNet(net)
    log = json.loads((n.dir / "log.json").read_text())
    tr = make_trials(n, batch_size=batch_size)
    got = perf_by_rule(n, tr)
    print(f"{'rule':<18}{'numpy':>8}{'log':>8}{'diff':>8}")
    d = []
    for r in n.rules:
        lg = log[f"perf_{r}"][-1]
        d.append(got[r] - lg)
        print(f"{r:<18}{got[r]:8.3f}{lg:8.3f}{got[r]-lg:+8.3f}")
    print(f"mean abs diff {np.mean(np.abs(d)):.4f}   max {np.max(np.abs(d)):.4f}")
    return got


if __name__ == "__main__":
    import sys
    selftest(*(sys.argv[1:] or []))
