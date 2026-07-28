"""Pull the central-complex subgraph from the hemibrain via neuPrint.

Pinned to `hemibrain:v1.2.1` -- the connectome is frozen at v1.2 (v1.2.1 only
recomputed synapse-to-mito distances), so this is the stable citable snapshot.

Requires a neuPrint auth token. Get one free at
    https://neuprint.janelia.org/  ->  account menu -> Auth Token
and expose it as NEUPRINT_TOKEN (or put it in the repo .env).

Outputs, under paths.CONNECTOME_ROOT / "hemibrain":
    neurons.csv    one row per CX neuron: bodyId, type, instance, status, size
    adjacency.csv  summed synapse counts per (bodyId_pre, bodyId_post)
    roi_edges.csv  the same edges resolved per ROI, kept for provenance
    meta.json      dataset tag, ROI set, criteria, client version, row counts

The adjacency is *unsigned*: hemibrain ships no neurotransmitter calls, so only
`gamma_ep` applies here. Signing comes from the FlyWire side.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from flexsym import paths

#: The connectome is frozen at v1.2; v1.2.1 is the current tag of that freeze.
DATASET = "hemibrain:v1.2.1"
SERVER = "neuprint.janelia.org"

#: Core central-complex neuropils. BU/GA/LAL are CX-associated but are input
#: pathways rather than CX proper -- add them explicitly if you want them.
CX_ROIS = ["EB", "PB", "FB", "NO"]


def _token() -> str:
    tok = os.environ.get("NEUPRINT_TOKEN")
    if not tok:
        raise SystemExit(
            "NEUPRINT_TOKEN is not set.\n"
            "  1. Sign in at https://neuprint.janelia.org/ (Google account, free)\n"
            "  2. Account menu -> Auth Token -> copy\n"
            f"  3. export NEUPRINT_TOKEN=... (or add it to {paths.REPO_ROOT / '.env'})"
        )
    return tok


def connect(dataset: str = DATASET):
    """Return an authenticated neuPrint client pinned to `dataset`."""
    from neuprint import Client

    return Client(SERVER, dataset=dataset, token=_token())


def fetch_cx(
    *,
    dataset: str = DATASET,
    rois: list[str] | None = None,
    traced_only: bool = True,
    min_weight: int = 1,
) -> dict[str, Any]:
    """Fetch CX neurons and their pairwise synapse counts.

    `traced_only` restricts to fully-traced, uncropped bodies -- the standard
    filter for connectivity analysis, since partially-reconstructed neurons have
    systematically deflated degree, and degree is exactly what our null models
    are sensitive to.
    """
    from neuprint import NeuronCriteria as NC, fetch_adjacencies, fetch_neurons

    rois = list(rois or CX_ROIS)
    client = connect(dataset)

    criteria = NC(rois=rois, client=client)
    if traced_only:
        criteria = NC(rois=rois, status="Traced", cropped=False, client=client)

    neurons, _roi_counts = fetch_neurons(criteria, client=client)
    if neurons.empty:
        raise SystemExit(f"no neurons matched rois={rois} in {dataset}")

    body_ids = neurons["bodyId"].tolist()
    _nrn, roi_edges = fetch_adjacencies(
        sources=NC(bodyId=body_ids, client=client),
        targets=NC(bodyId=body_ids, client=client),
        min_total_weight=min_weight,
        client=client,
    )

    adjacency = (
        roi_edges.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
        .sum()
        .sort_values(["bodyId_pre", "bodyId_post"], ignore_index=True)
    )

    meta = {
        "source": "hemibrain",
        "dataset": dataset,
        "server": SERVER,
        "rois": rois,
        "traced_only": traced_only,
        "min_weight": min_weight,
        "n_neurons": int(len(neurons)),
        "n_edges": int(len(adjacency)),
        "n_synapses": int(adjacency["weight"].sum()),
        "signed": False,
        "neuprint_python": _pkg_version("neuprint-python"),
    }
    return {"neurons": neurons, "adjacency": adjacency, "roi_edges": roi_edges, "meta": meta}


def _pkg_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def save(bundle: dict[str, Any], out_dir: Path | None = None) -> Path:
    """Write a `fetch_cx` bundle to disk and return the directory."""
    out_dir = Path(out_dir or (paths.CONNECTOME_ROOT / "hemibrain"))
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle["neurons"].to_csv(out_dir / "neurons.csv", index=False)
    bundle["adjacency"].to_csv(out_dir / "adjacency.csv", index=False)
    bundle["roi_edges"].to_csv(out_dir / "roi_edges.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(bundle["meta"], indent=2) + "\n")
    return out_dir


def load(out_dir: Path | None = None) -> dict[str, Any]:
    """Load a previously saved bundle."""
    out_dir = Path(out_dir or (paths.CONNECTOME_ROOT / "hemibrain"))
    paths.require(out_dir / "adjacency.csv", "hemibrain adjacency")
    return {
        "neurons": pd.read_csv(out_dir / "neurons.csv"),
        "adjacency": pd.read_csv(out_dir / "adjacency.csv"),
        "meta": json.loads((out_dir / "meta.json").read_text()),
    }
