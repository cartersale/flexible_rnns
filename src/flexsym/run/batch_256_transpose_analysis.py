"""Config-driven batch runner for 256-node transpose signed-WL analyses.

The runner is intentionally checkpointed at model and step granularity. Each
model gets a status JSON under ``<struct_root>/batch_256_transpose/checkpoints``,
and completed output files are used for resume/skip decisions.
"""

from __future__ import annotations

from flexsym import paths

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


REPO_DIR = paths.REPO_ROOT
STEP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_DIR / "configs" / "batch_256_transpose_config.json"
# Child steps run under the SAME interpreter as this runner (sys.executable).
# This previously defaulted to a `.venv` sibling of the repo, which after the
# migration resolved to an environment without flexsym installed -- every step
# died with "No module named 'flexsym'". Use --python to override deliberately.


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    expected: tuple[Path, ...]


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_DIR / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path)
    return cfg


def model_sort_key(path: Path) -> tuple[str, int]:
    stem, _, suffix = path.name.rpartition("_")
    return (stem, int(suffix) if suffix.isdigit() else -1)


def discover_models(cfg: dict[str, Any]) -> list[Path]:
    # Defaults to the released-network store resolved from .env; a config may
    # override it, but nothing tracked here hardcodes an absolute path.
    data_root = Path(cfg["data_root"]) if cfg.get("data_root") else paths.ORIGINAL_STORE
    exclude = set(cfg.get("exclude_models", []))
    models: dict[str, Path] = {}
    for pattern in cfg.get("model_globs", ["Leaky*_256_all_*"]):
        for path in data_root.glob(pattern):
            if path.is_dir() and path.name not in exclude:
                models[path.name] = path
    return sorted(models.values(), key=model_sort_key)


def completed(expected: tuple[Path, ...], force: bool) -> bool:
    return bool(expected) and not force and all(path.exists() for path in expected)


def run_command(step: Step, *, force: bool, dry_run: bool, log_file: Path) -> dict[str, Any]:
    if completed(step.expected, force):
        return {"step": step.name, "status": "skipped", "seconds": 0.0}

    if dry_run:
        return {"step": step.name, "status": "dry_run", "seconds": 0.0}

    start = time.time()
    proc = subprocess.run(
        step.command,
        cwd=str(REPO_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.time() - start
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as f:
        f.write(f"\n\n===== {step.name} =====\n")
        f.write(" ".join(step.command) + "\n")
        f.write(proc.stdout)
        if proc.stdout and not proc.stdout.endswith("\n"):
            f.write("\n")
        f.write(f"exit_code={proc.returncode} seconds={elapsed:.2f}\n")

    if proc.returncode != 0:
        tail = proc.stdout[-4000:] if proc.stdout else ""
        raise RuntimeError(f"Step failed: {step.name}\n{tail}\nLog: {log_file}")

    return {"step": step.name, "status": "completed", "seconds": elapsed}


def read_activity_info(activity_npz: Path) -> dict[str, int]:
    activity = np.load(activity_npz, allow_pickle=True)
    return {
        "activity_k": int(activity["best_k"]),
        "n_active": int(np.asarray(activity["ind_active"], dtype=np.int64).size),
    }


def select_structural_k(activity_npz: Path, sweep_csv: Path | None, cfg: dict[str, Any]) -> dict[str, Any]:
    info = read_activity_info(activity_npz)
    selection_cfg = cfg.get("k_selection", {"policy": "activity_or_saturation"})
    policy = selection_cfg.get("policy", "activity_or_saturation")

    if policy == "active_fraction":
        fraction = float(selection_cfg.get("active_fraction", 0.20))
        raw_k = info["n_active"] * fraction
        rounding = selection_cfg.get("rounding", "round")
        if rounding == "floor":
            structural_k = int(np.floor(raw_k))
        elif rounding == "ceil":
            structural_k = int(np.ceil(raw_k))
        else:
            structural_k = int(round(raw_k))
        structural_k = max(int(cfg["k_min"]), min(structural_k, info["n_active"] - 1))
        selected_k = min(info["activity_k"], structural_k)
        selected_k = max(int(cfg["k_min"]), min(selected_k, info["n_active"] - 1))
        return {
            **info,
            "selection_policy": policy,
            "active_fraction": fraction,
            "raw_structural_k": float(raw_k),
            "saturation_k": "",
            "structural_k": structural_k,
            "selected_k": selected_k,
            "most_negative_gamma_struct_z": "",
            "saturation_threshold_z": "",
        }

    if sweep_csv is None:
        raise ValueError("activity_or_saturation k selection requires a signed-WL sweep CSV")
    saturation_k, most_negative_z, threshold_z = read_saturation_k(
        sweep_csv,
        float(cfg["saturation_within_fraction"]),
    )
    structural_k = max(info["activity_k"], saturation_k)
    return {
        **info,
        "selection_policy": policy,
        "active_fraction": "",
        "raw_structural_k": "",
        "saturation_k": saturation_k,
        "structural_k": structural_k,
        "selected_k": structural_k,
        "most_negative_gamma_struct_z": most_negative_z,
        "saturation_threshold_z": threshold_z,
    }


def read_saturation_k(sweep_csv: Path, within_fraction: float) -> tuple[int, float, float]:
    rows: list[dict[str, float]] = []
    with sweep_csv.open() as f:
        for row in csv.DictReader(f):
            null_sd = float(row["gamma_null_sd"])
            z = np.nan
            if null_sd > 0:
                z = (float(row["gamma_struct"]) - float(row["gamma_null_mean"])) / null_sd
            rows.append({"k": int(row["k"]), "gamma_struct_z": float(z)})
    finite = [row for row in rows if np.isfinite(row["gamma_struct_z"])]
    if not finite:
        raise ValueError(f"No finite gamma_struct z-scores in {sweep_csv}")
    most_negative = min(row["gamma_struct_z"] for row in finite)
    if most_negative >= 0:
        saturation = min(
            (row for row in finite if row["gamma_struct_z"] == most_negative),
            key=lambda row: row["k"],
        )["k"]
        return int(saturation), float(most_negative), float(most_negative)
    threshold = most_negative * (1.0 - within_fraction)
    saturation = min(row["k"] for row in finite if row["gamma_struct_z"] <= threshold)
    return int(saturation), float(most_negative), float(threshold)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_preselection_steps(
    model_dir: Path,
    cfg: dict[str, Any],
    python: str,
    activity_npz: Path,
    variance_path: Path,
    sweep_dir: Path,
) -> list[Step]:
    activity_cfg = cfg["activity_clustering"]
    sweep_npz = sweep_dir / "signed_input_wl_sweep.npz"
    steps = [
        Step(
            "compute_variance",
            [
                python,
                "-m",
                "flexsym.training.compute_variance_epoch_to_file",
                str(model_dir),
                "--output",
                str(variance_path),
                "--batch-size",
                str(cfg["batch_size"]),
            ],
            (variance_path,),
        ),
        Step(
            "activity_clustering_driscoll_silhouette",
            [
                python,
                "-m",
                "flexsym.training.cluster_epoch_variance",
                str(model_dir),
                "--variance-path",
                str(variance_path),
                "--output-dir",
                str(activity_npz.parent),
                "--output-name",
                activity_npz.name,
                "--normalization",
                activity_cfg["normalization"],
                "--linkage-method",
                activity_cfg["linkage_method"],
                "--active-threshold",
                str(activity_cfg["active_threshold"]),
                "--k-min",
                str(cfg["k_min"]),
                "--k-max",
                str(cfg["k_max"]),
            ],
            (activity_npz,),
        ),
    ]
    if cfg.get("k_selection", {}).get("policy", "activity_or_saturation") != "active_fraction":
        steps.append(
            Step(
                "signed_input_wl_transpose_sweep_k3_to_k60",
                [
                    python,
                    "-m",
                "flexsym.structural.signed_input_wl_structural_roles",
                    str(model_dir),
                    "--activity-clusters",
                    str(activity_npz),
                    "--output-dir",
                    str(sweep_dir),
                    "--matrix-variant",
                    cfg["matrix_variant"],
                    "--k-min",
                    str(cfg["k_min"]),
                    "--k-max",
                    str(cfg["k_max"]),
                    "--n-null",
                    str(cfg["wl_null"]),
                    "--n-iter",
                    str(cfg["wl_n_iter"]),
                    "--n-init",
                    str(cfg["wl_n_init"]),
                    "--seed",
                    str(cfg["seed"]),
                ],
                (sweep_npz, sweep_dir / "signed_input_wl_sweep.csv"),
            )
        )
    return steps


def build_postselection_steps(
    model_dir: Path,
    cfg: dict[str, Any],
    python: str,
    activity_npz: Path,
    sweep_dir: Path,
    model_root: Path,
    structural_k: int,
) -> list[Step]:
    k_policy = cfg.get("k_selection", {}).get("policy")
    selected_dir = (
        model_root / f"transpose_selected_k_{structural_k}"
        if k_policy == "active_fraction"
        else model_root / "transpose_selected_k"
    )
    selected_npz = selected_dir / "signed_input_wl_sweep.npz"
    selected_k_min = structural_k if cfg.get("k_selection", {}).get("policy") == "active_fraction" else int(cfg["k_min"])
    selected_k_max = structural_k if cfg.get("k_selection", {}).get("policy") == "active_fraction" else int(cfg["k_max"])
    return [
        Step(
            "materialize_selected_transpose_partition",
            [
                python,
                "-m",
                "flexsym.structural.signed_input_wl_structural_roles",
                str(model_dir),
                "--activity-clusters",
                str(activity_npz),
                "--output-dir",
                str(selected_dir),
                "--matrix-variant",
                cfg["matrix_variant"],
                "--k-min",
                str(selected_k_min),
                "--k-max",
                str(selected_k_max),
                "--chosen-k",
                str(structural_k),
                "--n-null",
                str(cfg["wl_null"]),
                "--n-iter",
                str(cfg["wl_n_iter"]),
                "--n-init",
                str(cfg["wl_n_init"]),
                "--seed",
                str(cfg["seed"]),
            ],
            (selected_npz, selected_dir / f"contingency_struct_vs_act_k_{structural_k}.csv"),
        ),
    ]


def append_k_row(csv_path: Path, row: dict[str, Any]) -> None:
    fields = [
        "model",
        "selection_policy",
        "n_active",
        "active_fraction",
        "raw_structural_k",
        "activity_k",
        "saturation_k",
        "structural_k",
        "selected_k",
        "most_negative_gamma_struct_z",
        "saturation_threshold_z",
        "matrix_variant",
        "k_min",
        "k_max",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if csv_path.exists():
        with csv_path.open() as f:
            for old in csv.DictReader(f):
                existing[old["model"]] = old
    existing[row["model"]] = {key: row.get(key, "") for key in fields}
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing[name] for name in sorted(existing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", action="append", default=None, help="Optional model name filter; can be repeated.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    activity_root = resolve_repo_path(cfg["activity_root"])
    struct_root = resolve_repo_path(cfg["struct_root"])
    run_root = struct_root / cfg.get("run_name", "batch_256_transpose")
    checkpoints = run_root / "checkpoints"
    logs = run_root / "logs"
    k_csv = run_root / "k_selection_summary.csv"

    models = discover_models(cfg)
    if args.model:
        wanted = set(args.model)
        models = [model for model in models if model.name in wanted]
    if not models:
        raise SystemExit("No models selected.")

    print(f"Config: {args.config}")
    print(f"Selected {len(models)} models")
    print(f"K summary: {k_csv}")

    all_status: list[dict[str, Any]] = []
    model_bar = tqdm(models, desc="models", unit="model")
    for model_dir in model_bar:
        model_bar.set_postfix_str(model_dir.name)
        model_root = struct_root / model_dir.name
        activity_dir = activity_root / model_dir.name
        variance_path = activity_dir / "variance_epoch.pkl"
        activity_npz = activity_dir / cfg["activity_clustering"]["output_name"]
        sweep_dir = model_root / cfg["matrix_variant"]
        log_file = logs / f"{model_dir.name}.log"
        checkpoint_file = checkpoints / f"{model_dir.name}.json"
        checkpoint: dict[str, Any] = {
            "model": model_dir.name,
            "model_dir": str(model_dir),
            "config": cfg["_config_path"],
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": [],
            "status": "running",
        }
        write_json(checkpoint_file, checkpoint)

        try:
            pre_steps = build_preselection_steps(model_dir, cfg, args.python, activity_npz, variance_path, sweep_dir)
            for step in tqdm(pre_steps, desc=model_dir.name, unit="step", leave=False):
                result = run_command(step, force=args.force, dry_run=args.dry_run, log_file=log_file)
                checkpoint["steps"].append(result)
                write_json(checkpoint_file, checkpoint)

            if args.dry_run:
                checkpoint["status"] = "dry_run"
                write_json(checkpoint_file, checkpoint)
                continue

            sweep_csv = sweep_dir / "signed_input_wl_sweep.csv"
            k_selection = select_structural_k(
                activity_npz,
                sweep_csv if sweep_csv.exists() else None,
                cfg,
            )
            selected_k = int(k_selection["selected_k"])
            k_row = {
                "model": model_dir.name,
                **k_selection,
                "matrix_variant": cfg["matrix_variant"],
                "k_min": cfg["k_min"],
                "k_max": max(int(cfg["k_max"]), selected_k),
            }
            checkpoint["k_selection"] = k_row
            append_k_row(k_csv, k_row)
            write_json(checkpoint_file, checkpoint)

            post_steps = build_postselection_steps(
                model_dir, cfg, args.python, activity_npz, sweep_dir, model_root, selected_k
            )
            for step in tqdm(post_steps, desc=f"{model_dir.name} selected k={selected_k}", unit="step", leave=False):
                result = run_command(step, force=args.force, dry_run=args.dry_run, log_file=log_file)
                checkpoint["steps"].append(result)
                write_json(checkpoint_file, checkpoint)

            checkpoint["status"] = "completed"
            checkpoint["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_json(checkpoint_file, checkpoint)
        except Exception as exc:
            checkpoint["status"] = "failed"
            checkpoint["error"] = str(exc)
            checkpoint["failed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_json(checkpoint_file, checkpoint)
            raise
        finally:
            all_status.append({"model": model_dir.name, "checkpoint": str(checkpoint_file), "status": checkpoint["status"]})

    write_json(run_root / "last_run_summary.json", {"models": all_status, "k_selection_csv": str(k_csv)})
    print(f"Wrote status: {run_root / 'last_run_summary.json'}")


if __name__ == "__main__":
    main()
