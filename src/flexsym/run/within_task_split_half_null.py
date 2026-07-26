"""Wrapper to run the within-task split-half null (#2) vs cross-task fractionation.

Defaults target the canonical 256-unit softplus-diagonal seed-0 network
(LeakyRNN_softplus_diag_256_all_0), transpose variant, k=41 -- matching
run_cross_task_fractionation.py.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import dotenv

dotenv.load_dotenv()
HOME_DIR = Path(os.environ.get("HOME_DIR", "."))


def main():
    parser = argparse.ArgumentParser(description="Run within-task split-half null analysis")
    parser.add_argument("model_dir", help="Path to trained model checkpoint directory")
    parser.add_argument("--model-name", default="LeakyRNN_softplus_diag_256_all_0")
    parser.add_argument("--k", type=int, default=41)
    parser.add_argument(
        "--variant", choices=["original", "transpose", "no_diag"], default="transpose"
    )
    parser.add_argument("--min-cell-size", type=int, default=5)
    parser.add_argument("--max-subclusters", type=int, default=6)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_base = HOME_DIR / "results" / "signed_input_wl_paper" / args.model_name
    structural_labels = (
        results_base / f"{args.variant}_selected_k_{args.k}" / "signed_input_wl_sweep.npz"
    )
    output_dir = results_base / f"within_task_split_half_null_{args.variant}_k_{args.k}"

    if not Path(args.model_dir).exists():
        print(f"ERROR: Model directory not found: {args.model_dir}")
        sys.exit(1)
    if not structural_labels.exists():
        print(f"ERROR: Structural labels not found: {structural_labels}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Running Within-Task Split-Half Null (#2) ===")
    print(f"Model: {args.model_name}")
    print(f"K: {args.k}  Variant: {args.variant}  Splits: {args.n_splits}")
    print(f"Output: {output_dir}\n")

    cmd = [
        sys.executable,
        "stepnet/within_task_split_half_null.py",
        str(args.model_dir),
        "--structural-labels", str(structural_labels),
        "--k", str(args.k),
        "--output-dir", str(output_dir),
        "--min-cell-size", str(args.min_cell_size),
        "--max-subclusters", str(args.max_subclusters),
        "--n-splits", str(args.n_splits),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    result = subprocess.run(cmd, cwd=HOME_DIR)
    if result.returncode != 0:
        print("\nERROR: Analysis failed")
        sys.exit(1)

    print(f"\n=== Done ===\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
