"""Single source of truth for every filesystem root used by flexsym.

Nothing else in the package should contain an absolute path. Roots are read
from the environment (via a .env at the repo root, see .env.example) and fall
back to the layout on the original development machine.

The bulk data does not live in the repo: trained checkpoints and fixed-point
outputs are tens of GB and sit on an external volume pointed at by MODEL_ROOT.
`results/motif_fixed_points` and `results/motif_fixed_points_response` are
symlinks into it.

Roots are NOT validated at import time -- importing a module must not require
the data volume to be mounted. Call `require()` at the point of use so failures
are loud and say which root is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; env vars still work without it
    load_dotenv = None

# src/flexsym/paths.py -> src/flexsym -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")


def _root(var: str, default: Path | str) -> Path:
    """Resolve a root from the environment.

    A relative value (e.g. RESULTS_ROOT=./results) is resolved against the repo
    root, never the current working directory -- so `python -m flexsym.run.x`
    writes to the same place regardless of where it is invoked from.
    """
    p = Path(os.environ.get(var) or default).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# --- data roots (typically on the external volume) -------------------------

#: Trained checkpoints, fixed-point outputs, and other bulk artefacts.
MODEL_ROOT = _root("MODEL_ROOT", "/Volumes/X9_Pro/flexible_multitask_data")

#: Processed data released with Driscoll et al. 2024.
DATA_ROOT = _root("DATA_ROOT", MODEL_ROOT / "multitask_processed")

#: The released networks analysed alongside our own training grid.
ORIGINAL_STORE = _root("ORIGINAL_STORE", DATA_ROOT / "suppfig4")

#: Our own size x seed training grid.
STAGED_STORE = _root("STAGED_STORE", MODEL_ROOT / "new_runs" / "staged")

#: Cached network activity, keyed by (net, rule).
CACHE_ROOT = _root("CACHE_ROOT", MODEL_ROOT / "activity_cache")

#: Downloaded connectome reconstructions (hemibrain, FlyWire). Small enough to
#: sit in the repo tree by default, but overridable onto the data volume.
CONNECTOME_ROOT = _root("CONNECTOME_ROOT", REPO_ROOT / "data" / "connectome")

# --- output roots (in the repo, gitignored) --------------------------------

RESULTS_ROOT = _root("RESULTS_ROOT", REPO_ROOT / "results")
FIGURES_ROOT = _root("FIGURES_ROOT", RESULTS_ROOT / "figures")

# --- vendored external dependencies ----------------------------------------

EXTERNAL = REPO_ROOT / "external"


def require(path: Path, what: str) -> Path:
    """Return `path`, or raise with an actionable message if it is missing.

    Use at the point of use, not at import time, so that merely importing a
    module does not require the data volume to be mounted.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}\n"
            f"  Is the data volume mounted? Check MODEL_ROOT / DATA_ROOT in "
            f"{REPO_ROOT / '.env'} (see .env.example)."
        )
    return path


def ensure_dir(path: Path, what: str) -> Path:
    """Create `path` (with parents), or raise with an actionable message.

    `Path.mkdir(parents=True)` on a deep path under an unmounted volume fails
    obscurely: pathlib recurses upward creating parents and dies on /Volumes
    itself with EACCES, behind a stack of chained FileNotFoundErrors that name
    every level but the one that matters. Catch that and say what went wrong.

    Use for output roots, the write-side counterpart to `require()`.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # The deepest ancestor that does exist is the real boundary -- if it is
        # /Volumes, the data volume simply is not mounted.
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        raise OSError(
            f"cannot create {what} at {path}\n"
            f"  failed at {existing} ({e.strerror}).\n"
            f"  Is the data volume mounted? Check MODEL_ROOT / DATA_ROOT in "
            f"{REPO_ROOT / '.env'} (see .env.example), or pass --root."
        ) from e
    return path


def add_external_to_syspath() -> None:
    """Put FixedPointFinder and recurrent-whisperer on sys.path.

    These are not pip-installable and are vendored under external/; see README.
    Call this before importing FixedPointFinderTF.
    """
    import sys

    for name in ("recurrent-whisperer", "fixed-point-finder"):
        p = EXTERNAL / name
        if not p.exists():
            raise FileNotFoundError(
                f"external dependency missing: {p}\n"
                f"  git clone https://github.com/mattgolub/{name}.git "
                f"into {EXTERNAL}/ (see README)."
            )
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
