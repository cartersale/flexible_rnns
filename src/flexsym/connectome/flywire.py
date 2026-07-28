"""Pull the central-complex subgraph from FlyWire (FAFB) via CAVE.

Pinned to materialization **783**, the long-term-support release accompanying
Dorkenwald et al. and Schlegel et al. (Nature, 2024). Unlike ordinary
materialization versions, 783 does not expire -- anything else will silently
drift or 404 later, so the version is pinned explicitly rather than defaulted.

Requires a CAVE token (and agreement to the FlyWire principles):
    https://global.daf-apis.com/auth/api/v1/user/token
Store it once with `caveclient.CAVEclient().auth.save_token(token=...)`, or
expose it as CAVE_TOKEN.

Two things FlyWire adds over hemibrain:
  * both hemispheres, so left/right columnar structure is testable rather than
    assumed;
  * per-synapse neurotransmitter predictions (Eckstein et al. 2024), which let
    us build a *signed* adjacency and therefore apply `gamma_signed`.

Table names in the public datastack have changed across releases, so this module
introspects `materialize.get_tables()` and matches by pattern rather than
hardcoding. Pass explicit names if the guess is wrong -- the error lists what is
actually available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flexsym import paths

DATASTACK = "flywire_fafb_public"

#: LTS materialization for the published FlyWire connectome. Do not float this.
MATERIALIZATION = 783

#: Predicted NT -> sign. ACh is excitatory; GABA and glutamate are inhibitory at
#: the great majority of central-brain synapses (Glu via GluCl-alpha). The
#: monoamines are neither cleanly, so they are dropped from the signed graph
#: rather than guessed at -- see `sign_edges(drop_ambiguous=...)`.
NT_SIGN = {"acetylcholine": +1, "gaba": -1, "glutamate": -1}
NT_AMBIGUOUS = ("dopamine", "serotonin", "octopamine", "unknown")

#: Substrings identifying CX cell types in the FlyWire annotation tables.
CX_TYPE_PREFIXES = ("EPG", "PEG", "PEN", "PFN", "PFL", "PFR", "FC", "FS", "FB", "hDelta", "vDelta", "ER", "EL", "ExR", "IbSpsP", "SpsP", "LNO", "LCNO", "GLNO")


def _token() -> str | None:
    return os.environ.get("CAVE_TOKEN")


def connect(datastack: str = DATASTACK, materialization: int = MATERIALIZATION):
    """Return a CAVE client pinned to `materialization`."""
    import caveclient

    client = caveclient.CAVEclient(datastack, auth_token=_token())
    versions = client.materialize.get_versions()
    if materialization not in versions:
        raise SystemExit(
            f"materialization {materialization} not available on {datastack}.\n"
            f"  available: {sorted(versions)}\n"
            "  783 is the LTS public release; if it is gone, pin another "
            "explicitly rather than letting the client default."
        )
    client.materialize.version = materialization
    return client


def _pick_table(client, *patterns: str) -> str:
    """Return the first available table whose name contains any pattern."""
    tables = client.materialize.get_tables()
    for pat in patterns:
        hits = [t for t in tables if pat in t]
        if hits:
            return sorted(hits, key=len)[0]
    raise SystemExit(
        f"no table matching {patterns} in {DATASTACK} v{client.materialize.version}.\n"
        f"  available tables: {sorted(tables)}\n"
        "  pass the right name explicitly (--annotation-table / --synapse-table)."
    )


def fetch_annotations(client, table: str | None = None) -> pd.DataFrame:
    """Fetch the cell-type annotation table (Schlegel et al. hierarchical types)."""
    table = table or _pick_table(client, "cell_info", "hierarchical_annotation", "codex_annotation")
    return client.materialize.query_table(table)


def select_cx(ann: pd.DataFrame, prefixes: tuple[str, ...] = CX_TYPE_PREFIXES) -> pd.DataFrame:
    """Restrict an annotation table to rows whose cell type looks central-complex.

    Prefix matching is a *starting* filter, not the ground truth partition -- it
    decides which neurons enter the graph, not how they are grouped. The grouping
    comes from the columnar assignment, which is a separate join.
    """
    type_col = next(
        (c for c in ("cell_type", "type", "hierarchy_type", "primary_type") if c in ann.columns),
        None,
    )
    if type_col is None:
        raise SystemExit(f"no recognisable type column in annotations; got {list(ann.columns)}")
    is_cx = ann[type_col].fillna("").str.startswith(prefixes)
    out = ann.loc[is_cx].copy()
    out.attrs["type_col"] = type_col
    return out


def fetch_synapses(
    client,
    root_ids: np.ndarray,
    *,
    table: str | None = None,
    batch: int = 200,
) -> pd.DataFrame:
    """Fetch synapses among `root_ids`, batched over presynaptic partners.

    The full FAFB synapse table is ~10^8 rows, so it is queried per-batch of
    presynaptic roots and filtered to the same root set on the postsynaptic side.
    """
    table = table or _pick_table(client, "synapses_nt", "synapse")
    root_ids = np.asarray(root_ids, dtype=np.int64)
    keep = set(root_ids.tolist())

    frames = []
    for i in range(0, len(root_ids), batch):
        chunk = root_ids[i : i + batch]
        df = client.materialize.query_table(
            table,
            filter_in_dict={"pre_pt_root_id": chunk.tolist()},
        )
        if not df.empty:
            frames.append(df[df["post_pt_root_id"].isin(keep)])
        print(f"  synapses: {min(i + batch, len(root_ids))}/{len(root_ids)} roots", flush=True)

    if not frames:
        return pd.DataFrame(columns=["pre_pt_root_id", "post_pt_root_id"])
    return pd.concat(frames, ignore_index=True)


def sign_edges(syn: pd.DataFrame, *, drop_ambiguous: bool = True) -> pd.DataFrame:
    """Collapse synapses to signed per-pair edges using predicted neurotransmitter.

    Each presynaptic neuron gets ONE sign -- the majority NT call over its
    synapses -- rather than signing synapse-by-synapse. Dale's law makes the
    per-neuron call the meaningful one, and per-synapse noise in the predictions
    would otherwise leak into edge signs.

    Returns columns: pre, post, weight (synapse count), nt, sign, signed_weight.
    """
    nt_col = next((c for c in syn.columns if "neuro_transmitter" in c or c == "nt_type"), None)
    counts = (
        syn.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)
        .size()
        .rename(columns={"pre_pt_root_id": "pre", "post_pt_root_id": "post", "size": "weight"})
    )

    if nt_col is None:
        counts["nt"] = "unknown"
        counts["sign"] = 0
        counts["signed_weight"] = 0.0
        return counts

    per_neuron_nt = (
        syn.groupby("pre_pt_root_id")[nt_col]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("nt")
    )
    counts = counts.merge(per_neuron_nt, left_on="pre", right_index=True, how="left")
    counts["nt"] = counts["nt"].fillna("unknown").str.lower()
    counts["sign"] = counts["nt"].map(NT_SIGN).fillna(0).astype(int)
    if drop_ambiguous:
        counts = counts[counts["sign"] != 0].reset_index(drop=True)
    counts["signed_weight"] = counts["sign"] * counts["weight"]
    return counts


def fetch_cx(
    *,
    datastack: str = DATASTACK,
    materialization: int = MATERIALIZATION,
    annotation_table: str | None = None,
    synapse_table: str | None = None,
) -> dict[str, Any]:
    """Fetch CX neurons, their synapses, and the signed edge list."""
    client = connect(datastack, materialization)
    ann = fetch_annotations(client, annotation_table)
    cx = select_cx(ann)
    if cx.empty:
        raise SystemExit("CX type filter matched nothing -- check the type column and prefixes.")

    root_ids = cx["pt_root_id"].to_numpy(dtype=np.int64)
    syn = fetch_synapses(client, root_ids, table=synapse_table)
    edges = sign_edges(syn)

    meta = {
        "source": "flywire",
        "datastack": datastack,
        "materialization": materialization,
        "annotation_table": annotation_table or "<auto>",
        "synapse_table": synapse_table or "<auto>",
        "n_neurons": int(len(cx)),
        "n_edges": int(len(edges)),
        "n_synapses": int(len(syn)),
        "signed": True,
        "nt_sign": NT_SIGN,
        "caveclient": _pkg_version("caveclient"),
    }
    return {"neurons": cx, "edges": edges, "meta": meta}


def _pkg_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def save(bundle: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or (paths.CONNECTOME_ROOT / "flywire"))
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle["neurons"].to_csv(out_dir / "neurons.csv", index=False)
    bundle["edges"].to_csv(out_dir / "edges.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(bundle["meta"], indent=2) + "\n")
    return out_dir


def load(out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir or (paths.CONNECTOME_ROOT / "flywire"))
    paths.require(out_dir / "edges.csv", "flywire edges")
    return {
        "neurons": pd.read_csv(out_dir / "neurons.csv"),
        "edges": pd.read_csv(out_dir / "edges.csv"),
        "meta": json.loads((out_dir / "meta.json").read_text()),
    }
