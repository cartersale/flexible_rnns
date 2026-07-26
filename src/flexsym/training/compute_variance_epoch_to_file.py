"""Compute Driscoll-style epoch variance and write it to an arbitrary file."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np




def compute_variance_epoch(model_dir: str, batch_size: int = 1024) -> dict:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    from flexsym.core.network import Model
    from flexsym.core.task import generate_trials
    from flexsym.core import tools

    model = Model(model_dir, sigma_rec=0)
    h_all_byepoch = OrderedDict()
    with tf.Session() as sess:
        model.restore()
        hp = model.hp
        rules = hp["rules"]
        n_hidden = hp["n_rnn"]
        for rule in rules:
            trial = generate_trials(rule, hp, "test", noise_on=False, batch_size=batch_size)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)
            for e_name, e_time in trial.epochs.items():
                if "fix" in e_name:
                    continue
                start = 0 if e_time[0] is None else e_time[0]
                stop = h.shape[0] if e_time[1] is None else e_time[1]
                h_all_byepoch[(rule, e_name)] = h[start:stop, :, :]

    keys = list(h_all_byepoch.keys())
    ind_key_sort = np.argsort(list(zip(*keys))[1], kind="mergesort")
    h_all_byepoch = OrderedDict([(keys[i], h_all_byepoch[keys[i]]) for i in ind_key_sort])

    h_var_all = np.zeros((n_hidden, len(h_all_byepoch)), dtype=float)
    h_var_end = np.zeros((n_hidden, len(h_all_byepoch)), dtype=float)
    for i, val in enumerate(h_all_byepoch.values()):
        val_reshape = np.reshape(val, (-1, n_hidden))
        h_var_all[:, i] = val_reshape.var(axis=0)
        h_var_end[:, i] = val[-1, :, :].var(axis=0)

    return {"h_var_all": h_var_all, "h_var_end": h_var_end, "keys": list(h_all_byepoch.keys())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    result = compute_variance_epoch(args.model_dir, batch_size=args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(result, f)
    print(
        {
            "output": str(output),
            "n_units": int(result["h_var_all"].shape[0]),
            "n_task_epochs": int(result["h_var_all"].shape[1]),
            "batch_size": args.batch_size,
        }
    )


if __name__ == "__main__":
    main()
