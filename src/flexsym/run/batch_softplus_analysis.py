"""Batch runner for the softplus signed-WL/activity analyses.

By default this processes the two matched softplus checkpoints that were not
used as the canonical pilot:

    LeakyRNN_softplus_diag_256_all_1
    LeakyRNN_softplus_diag_256_all_2

Use --include-canonical to include seed 0 as well.
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


REPO_DIR = paths.REPO_ROOT
RESULTS_ROOT = paths.RESULTS_ROOT
STEP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = paths.ORIGINAL_STORE
DEFAULT_ACTIVITY_ROOT = RESULTS_ROOT / "activity_clusters_paper"
DEFAULT_STRUCT_ROOT = RESULTS_ROOT / "signed_input_wl_paper"
# Child steps run under the SAME interpreter as this runner (sys.executable).
# This previously defaulted to a `.venv` sibling of the repo, which after the
# migration resolved to an environment without flexsym installed -- every step
# died with "No module named 'flexsym'". Use --python to override deliberately.
SOFTPLUS_PATTERN = "LeakyRNN_softplus_diag_256_all_*"
VARIANTS = ("original", "transpose", "no_diag")


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    expected: tuple[Path, ...]


def model_sort_key(path: Path) -> tuple[str, int]:
    suffix = path.name.rsplit("_", 1)[-1]
    return (path.name.rsplit("_", 1)[0], int(suffix) if suffix.isdigit() else -1)


def discover_models(data_root: Path, include_canonical: bool) -> list[Path]:
    models = sorted(data_root.glob(SOFTPLUS_PATTERN), key=model_sort_key)
    if not include_canonical:
        models = [m for m in models if not m.name.endswith("_0")]
    return models


def add_python(command: list[str], python: str) -> list[str]:
    return [python, *command]


# Steps are launched as `python -m <module>`, not by file path. Before the
# package migration this was `STEP_DIR / "<script>.py"`, which broke the moment
# the modules moved into subpackages -- the runner would fail at step 1 with a
# bare "No such file or directory".
STEP_MODULES = {
    "compute_variance_epoch_to_file.py": "flexsym.training.compute_variance_epoch_to_file",
    "cluster_epoch_variance.py": "flexsym.training.cluster_epoch_variance",
    "signed_input_wl_structural_roles.py": "flexsym.structural.signed_input_wl_structural_roles",
    "structural_cell_epoch_variance.py": "flexsym.structural.structural_cell_epoch_variance",
    "structural_cell_activity_coherence.py": "flexsym.structural.structural_cell_activity_coherence",
    "random_partition_activity_covariance_null.py": "flexsym.structural.random_partition_activity_covariance_null",
    "softplus_memory_subpartition_sweep.py": "flexsym.structural.softplus_memory_subpartition_sweep",
    "softplus_memory_full_context_alignment.py": "flexsym.structural.softplus_memory_full_context_alignment",
}

# Referenced by this runner but archived during the July 2026 cleanup as
# superseded exploratory work. They were never migrated into the package, so any
# config enabling their steps cannot run.
ARCHIVED_STEPS = {
    "aggregate_structural_cell_epochs.py",
    "plot_structural_cell_epoch_variance_figures.py",
    "compare_computational_components_k33.py",
}


def step_module(script: str) -> str:
    """Map a legacy script filename to its module path."""
    if script in STEP_MODULES:
        return STEP_MODULES[script]
    if script in ARCHIVED_STEPS:
        raise RuntimeError(
            f"Step '{script}' was archived as superseded and is not part of the "
            f"package. Disable the step in your config, or restore the script "
            f"from the pre-migration repo if you still need it."
        )
    raise KeyError(f"No module mapping for step script '{script}'")


def should_skip(expected: tuple[Path, ...], force: bool) -> bool:
    return bool(expected) and not force and all(path.exists() for path in expected)


def run_step(step: Step, force: bool, dry_run: bool, log_file: Path) -> dict[str, object]:
    if should_skip(step.expected, force):
        print(f"[skip] {step.name}")
        return {"step": step.name, "status": "skipped", "seconds": 0.0}

    print(f"[run]  {step.name}")
    print("       " + " ".join(step.command))
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
    with log_file.open("a") as f:
        f.write(f"\n\n===== {step.name} =====\n")
        f.write(" ".join(step.command) + "\n")
        f.write(proc.stdout)
        if proc.stdout and not proc.stdout.endswith("\n"):
            f.write("\n")
        f.write(f"exit_code={proc.returncode} seconds={elapsed:.2f}\n")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        raise RuntimeError(f"Step failed: {step.name}. See {log_file}")
    return {"step": step.name, "status": "completed", "seconds": elapsed}


def build_steps(
    model_dir: Path,
    python: str,
    activity_root: Path,
    struct_root: Path,
    chosen_k: int,
    batch_size: int,
    wl_null: int,
    covariance_null: int,
    subpartition_null: int,
    skip_hidden: bool,
    skip_coherence: bool,
    skip_subpartition: bool,
) -> list[Step]:
    model_name = model_dir.name
    activity_dir = activity_root / model_name
    base_dir = struct_root / model_name
    variance_path = activity_dir / "variance_epoch.pkl"
    activity_clusters = activity_dir / "epoch_variance_unit_clusters_paper_max.npz"

    def py(script: str, *args: object) -> list[str]:
        return add_python(["-m", step_module(script), *map(str, args)], python)

    steps: list[Step] = []
    if not skip_hidden:
        steps.append(
            Step(
                f"{model_name}: compute task-epoch variance",
                py(
                    "compute_variance_epoch_to_file.py",
                    model_dir,
                    "--output",
                    variance_path,
                    "--batch-size",
                    batch_size,
                ),
                (variance_path,),
            )
        )
    steps.append(
        Step(
            f"{model_name}: Driscoll activity clustering",
            py(
                "cluster_epoch_variance.py",
                model_dir,
                "--variance-path",
                variance_path,
                "--output-dir",
                activity_dir,
                "--output-name",
                activity_clusters.name,
                "--normalization",
                "max",
                "--k-min",
                3,
                "--k-max",
                40,
            ),
            (activity_clusters,),
        )
    )

    for variant in VARIANTS:
        variant_dir = base_dir / variant
        sweep = variant_dir / "signed_input_wl_sweep.npz"
        steps.append(
            Step(
                f"{model_name}: signed input WL {variant}",
                py(
                    "signed_input_wl_structural_roles.py",
                    model_dir,
                    "--activity-clusters",
                    activity_clusters,
                    "--output-dir",
                    variant_dir,
                    "--chosen-k",
                    chosen_k,
                    "--matrix-variant",
                    variant,
                    "--k-min",
                    3,
                    "--k-max",
                    40,
                    "--n-null",
                    wl_null,
                ),
                (sweep,),
            )
        )
        cell_var_dir = variant_dir / "cell_epoch_variance"
        cell_var = cell_var_dir / f"structural_cell_epoch_variance_k_{chosen_k}.npz"
        steps.append(
            Step(
                f"{model_name}: structural cell epoch variance {variant}",
                py(
                    "structural_cell_epoch_variance.py",
                    "--activity-clusters",
                    activity_clusters,
                    "--structural-sweep",
                    sweep,
                    "--output-dir",
                    cell_var_dir,
                    "--k",
                    chosen_k,
                ),
                (cell_var,),
            )
        )

    original_cell_var = base_dir / "original" / "cell_epoch_variance" / f"structural_cell_epoch_variance_k_{chosen_k}.npz"
    transpose_cell_var = base_dir / "transpose" / "cell_epoch_variance" / f"structural_cell_epoch_variance_k_{chosen_k}.npz"
    no_diag_cell_var = base_dir / "no_diag" / "cell_epoch_variance" / f"structural_cell_epoch_variance_k_{chosen_k}.npz"

    steps.extend(
        [
            Step(
                f"{model_name}: aggregate epoch types",
                py(
                    "aggregate_structural_cell_epochs.py",
                    "--input-npz",
                    original_cell_var,
                    transpose_cell_var,
                    no_diag_cell_var,
                    "--output-dir",
                    base_dir / "epoch_type_variance",
                ),
                (base_dir / "epoch_type_variance" / f"structural_cell_epoch_type_variance_k_{chosen_k}.npz",),
            ),
            Step(
                f"{model_name}: epoch variance figures",
                py(
                    "plot_structural_cell_epoch_variance_figures.py",
                    "--base-dir",
                    base_dir,
                    "--output-dir",
                    base_dir / "figures",
                    "--ks",
                    chosen_k,
                ),
                (base_dir / "figures" / f"structural_cell_epoch_variance_panel_k_{chosen_k}.png",),
            ),
            Step(
                f"{model_name}: computational component comparisons",
                py(
                    "compare_computational_components_k33.py",
                    "--input-npz",
                    original_cell_var,
                    transpose_cell_var,
                    no_diag_cell_var,
                    "--output-dir",
                    base_dir / "component_comparisons",
                    "--top-n",
                    7,
                ),
                (base_dir / "component_comparisons" / f"computational_component_summary_k{chosen_k}.json",),
            ),
        ]
    )

    if not skip_coherence:
        coherence_root = base_dir / "activity_coherence_absrank"
        for variant in VARIANTS:
            sweep = base_dir / variant / "signed_input_wl_sweep.npz"
            cell_var = base_dir / variant / "cell_epoch_variance" / f"structural_cell_epoch_variance_k_{chosen_k}.npz"
            coherence_dir = coherence_root / variant
            coherence_npz = coherence_dir / f"structural_cell_activity_coherence_{variant}_k{chosen_k}.npz"
            steps.append(
                Step(
                    f"{model_name}: activity coherence {variant}",
                    py(
                        "structural_cell_activity_coherence.py",
                        model_dir,
                        "--activity-clusters",
                        activity_clusters,
                        "--structural-sweep",
                        sweep,
                        "--cell-variance",
                        cell_var,
                        "--output-dir",
                        coherence_dir,
                        "--k",
                        chosen_k,
                        "--batch-size",
                        batch_size,
                    ),
                    (coherence_npz,),
                )
            )
            null_dir = coherence_root / "random_null" / variant
            steps.append(
                Step(
                    f"{model_name}: random covariance null {variant}",
                    py(
                        "random_partition_activity_covariance_null.py",
                        model_dir,
                        "--activity-clusters",
                        activity_clusters,
                        "--structural-sweep",
                        sweep,
                        "--structural-coherence",
                        coherence_npz,
                        "--output-dir",
                        null_dir,
                        "--k",
                        chosen_k,
                        "--n-random",
                        covariance_null,
                        "--batch-size",
                        batch_size,
                    ),
                    (null_dir / f"random_partition_covariance_null_summary_{variant}_k{chosen_k}.json",),
                )
            )

    if not skip_subpartition:
        subpartition_dir = base_dir / "subpartition_sweep"
        steps.append(
            Step(
                f"{model_name}: memory subpartition sweep",
                py(
                    "softplus_memory_subpartition_sweep.py",
                    model_dir,
                    "--structural-sweep",
                    base_dir / "original" / "signed_input_wl_sweep.npz",
                    "--cell-variance",
                    original_cell_var,
                    "--output-dir",
                    subpartition_dir,
                    "--k",
                    chosen_k,
                    "--batch-size",
                    batch_size,
                ),
                (subpartition_dir / "softplus_memory_subpartition_sweep_summary.json",),
            )
        )
        steps.append(
            Step(
                f"{model_name}: memory full-context wiring alignment",
                py(
                    "softplus_memory_full_context_alignment.py",
                    model_dir,
                    "--structural-sweep",
                    base_dir / "original" / "signed_input_wl_sweep.npz",
                    "--cell-variance",
                    original_cell_var,
                    "--output-dir",
                    subpartition_dir,
                    "--k",
                    chosen_k,
                    "--batch-size",
                    batch_size,
                    "--n-null",
                    subpartition_null,
                ),
                (subpartition_dir / "softplus_memory_functional_full_context_wiring_alignment_null_summary.json",),
            )
        )

    return steps


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "step", "status", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    with output_path.with_suffix(".json").open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--activity-root", type=Path, default=DEFAULT_ACTIVITY_ROOT)
    parser.add_argument("--struct-root", type=Path, default=DEFAULT_STRUCT_ROOT)
    parser.add_argument("--model-dir", type=Path, action="append", default=None)
    parser.add_argument("--include-canonical", action="store_true")
    parser.add_argument("--chosen-k", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--wl-null", type=int, default=1000)
    parser.add_argument("--covariance-null", type=int, default=200)
    parser.add_argument("--subpartition-null", type=int, default=5000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-hidden", action="store_true")
    parser.add_argument("--skip-coherence", action="store_true")
    parser.add_argument("--skip-subpartition", action="store_true")
    args = parser.parse_args()

    models = args.model_dir if args.model_dir else discover_models(args.data_root, args.include_canonical)
    if not models:
        raise SystemExit(f"No models found under {args.data_root} matching {SOFTPLUS_PATTERN}")

    logs_dir = args.struct_root / "batch_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_rows: list[dict[str, object]] = []

    print("Models:")
    for model in models:
        print(f"  {model}")

    for model_dir in models:
        log_file = logs_dir / f"{model_dir.name}_{run_stamp}.log"
        steps = build_steps(
            model_dir=model_dir,
            python=args.python,
            activity_root=args.activity_root,
            struct_root=args.struct_root,
            chosen_k=args.chosen_k,
            batch_size=args.batch_size,
            wl_null=args.wl_null,
            covariance_null=args.covariance_null,
            subpartition_null=args.subpartition_null,
            skip_hidden=args.skip_hidden,
            skip_coherence=args.skip_coherence,
            skip_subpartition=args.skip_subpartition,
        )
        for step in steps:
            result = run_step(step, force=args.force, dry_run=args.dry_run, log_file=log_file)
            summary_rows.append({"model": model_dir.name, **result})

    summary_path = logs_dir / f"batch_softplus_analysis_{run_stamp}.csv"
    write_summary(summary_rows, summary_path)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
