"""Download the CX connectome subgraph and report what we actually got.

    python -m flexsym.run.fetch_connectome --source hemibrain
    python -m flexsym.run.fetch_connectome --source flywire
    python -m flexsym.run.fetch_connectome --source both

This only acquires and describes the data -- no statistics are computed here.
The point of the printout is to make the two things that can silently invalidate
the whole test visible up front:

  1. columnar-partition coverage (how many neurons actually get a column), and
  2. cell sizes (a partition with many singleton cells is trivially equitable
     and tells us nothing).

Credentials:
    NEUPRINT_TOKEN  https://neuprint.janelia.org/ -> Auth Token
    CAVE_TOKEN      https://global.daf-apis.com/auth/api/v1/user/token
"""

from __future__ import annotations

from flexsym import paths

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from flexsym.connectome import columns, graph  # noqa: E402


def do_hemibrain(out_root: Path, *, scaling: str) -> None:
    from flexsym.connectome import hemibrain

    print("=== hemibrain ===")
    bundle = hemibrain.fetch_cx()
    out = hemibrain.save(bundle, out_root / "hemibrain")
    m = bundle["meta"]
    print(f"  {m['n_neurons']} neurons, {m['n_edges']} edges, {m['n_synapses']} synapses")
    print(f"  wrote {out}")

    neurons = bundle["neurons"]
    W = graph.build_matrix(
        neurons["bodyId"],
        bundle["adjacency"],
        pre_col="bodyId_pre",
        post_col="bodyId_post",
        scaling=scaling,
    )
    print("  " + graph.summarise(W, "W_hemibrain"))

    part_col = columns.from_instance(neurons)
    part_type = columns.by_type(neurons)
    for p in (part_type, part_col, columns.cross(part_type, part_col)):
        print("  " + p.report().replace("\n", "\n  "))

    np.savez_compressed(
        out / "matrix.npz",
        W=W,
        body_ids=neurons["bodyId"].to_numpy(),
        column_labels=part_col.labels,
        column_names=np.array(part_col.names, dtype=object),
        type_labels=part_type.labels,
        type_names=np.array(part_type.names, dtype=object),
        scaling=scaling,
    )
    print(f"  wrote {out / 'matrix.npz'}")


def do_flywire(out_root: Path, *, scaling: str, annotation_table: str | None, synapse_table: str | None) -> None:
    from flexsym.connectome import flywire

    print("=== flywire ===")
    bundle = flywire.fetch_cx(annotation_table=annotation_table, synapse_table=synapse_table)
    out = flywire.save(bundle, out_root / "flywire")
    m = bundle["meta"]
    print(f"  {m['n_neurons']} neurons, {m['n_edges']} signed edges, {m['n_synapses']} synapses")
    print(f"  wrote {out}")

    neurons = bundle["neurons"]
    W = graph.build_matrix(
        neurons["pt_root_id"],
        bundle["edges"],
        pre_col="pre",
        post_col="post",
        weight_col="signed_weight",
        scaling=scaling,
    )
    print("  " + graph.summarise(W, "W_flywire(signed)"))
    W_pos, W_neg = graph.split_sign(W)
    print(f"  excitatory edges: {(W_pos > 0).sum()}   inhibitory edges: {(W_neg > 0).sum()}")

    type_col = neurons.attrs.get("type_col", "cell_type")
    part_type = columns.by_type(neurons, type_col=type_col)
    print("  " + part_type.report().replace("\n", "\n  "))

    np.savez_compressed(
        out / "matrix.npz",
        W=W,
        root_ids=neurons["pt_root_id"].to_numpy(),
        type_labels=part_type.labels,
        type_names=np.array(part_type.names, dtype=object),
        scaling=scaling,
    )
    print(f"  wrote {out / 'matrix.npz'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["hemibrain", "flywire", "both"], default="both")
    p.add_argument("--scaling", choices=["raw", "log1p", "infrac"], default="raw")
    p.add_argument("--annotation-table", default=None, help="FlyWire annotation table override")
    p.add_argument("--synapse-table", default=None, help="FlyWire synapse table override")
    p.add_argument("--output-dir", default=str(paths.CONNECTOME_ROOT))
    args = p.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.source in ("hemibrain", "both"):
        do_hemibrain(out_root, scaling=args.scaling)
    if args.source in ("flywire", "both"):
        do_flywire(
            out_root,
            scaling=args.scaling,
            annotation_table=args.annotation_table,
            synapse_table=args.synapse_table,
        )


if __name__ == "__main__":
    main()
