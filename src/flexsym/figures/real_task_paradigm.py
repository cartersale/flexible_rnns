#!/usr/bin/env python3
"""Plot real task paradigms from the multitask trial generator.

This figure is meant as a data-derived replacement for a purely schematic
task panel: it uses generated Trial.x/Trial.y arrays and Trial.epochs from the
canonical model hyperparameters.
"""

from __future__ import annotations

from flexsym import paths

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


from flexsym.core.task import generate_trials, rule_name, rules_dict  # noqa: E402


DEFAULT_MODEL_DIR = paths.ORIGINAL_STORE / "LeakyRNN_softplus_diag_256_all_0"
DEFAULT_OUTPUT_DIR = (
    paths.RESULTS_ROOT
    / "signed_input_wl_paper"
    / "LeakyRNN_softplus_diag_256_all_0"
    / "figures"
)

TASKS_TO_PLOT = ["delaygo", "delayanti", "delaydm1"]
DISPLAY_RULE_ORDER = [
    "fdgo",
    "fdanti",
    "reactgo",
    "reactanti",
    "delaygo",
    "delayanti",
    "delaydm1",
    "delaydm2",
    "contextdelaydm1",
    "contextdelaydm2",
    "multidelaydm",
    "dmsgo",
    "dmsnogo",
    "dmcgo",
    "dmcnogo",
]
TASK_LABELS = {
    "fdgo": "Go",
    "reactgo": "RT Go",
    "delaygo": "MemoryPro",
    "fdanti": "Anti",
    "reactanti": "RT Anti",
    "delayanti": "MemoryAnti",
    "delaydm1": "DelayDM1",
    "delaydm2": "DelayDM2",
    "contextdelaydm1": "Ctx DelayDM1",
    "contextdelaydm2": "Ctx DelayDM2",
    "multidelaydm": "MultSen DelayDM",
    "dmsgo": "DMS",
    "dmsnogo": "DNMS",
    "dmcgo": "DMC",
    "dmcnogo": "DNMC",
}
TASK_COMPUTATIONS = {
    "fdgo": "pro response after cue",
    "reactgo": "immediate pro response",
    "delaygo": "remember direction -> pro",
    "fdanti": "anti response after cue",
    "reactanti": "immediate anti response",
    "delayanti": "remember direction -> anti",
    "delaydm1": "compare two stimuli, modality 1",
    "delaydm2": "compare two stimuli, modality 2",
    "contextdelaydm1": "context selects modality 1",
    "contextdelaydm2": "context selects modality 2",
    "multidelaydm": "combine modalities",
    "dmsgo": "sample-test exact match",
    "dmsnogo": "sample-test non-match",
    "dmcgo": "same category",
    "dmcnogo": "different category",
}
TASK_GROUPS = {
    "fdgo": "Pro/anti",
    "reactgo": "Pro/anti",
    "delaygo": "Pro/anti",
    "fdanti": "Pro/anti",
    "reactanti": "Pro/anti",
    "delayanti": "Pro/anti",
    "delaydm1": "Decision",
    "delaydm2": "Decision",
    "contextdelaydm1": "Decision",
    "contextdelaydm2": "Decision",
    "multidelaydm": "Decision",
    "dmsgo": "Match/category",
    "dmsnogo": "Match/category",
    "dmcgo": "Match/category",
    "dmcnogo": "Match/category",
}
EPOCH_COLORS = {
    "fix1": "#d9d9d9",
    "stim1": "#f4a261",
    "delay1": "#8ecae6",
    "stim2": "#e76f51",
    "delay2": "#90be6d",
    "go1": "#b8a1d9",
}
TRACE_COLORS = {
    "fixation input": "#2b2d42",
    "stimulus drive": "#e68632",
    "rule cue": "#536dfe",
    "fixation target": "#333333",
    "response target": "#009688",
}


def load_hp(model_dir: Path, seed: int) -> dict:
    with (model_dir / "hp.json").open() as f:
        hp = json.load(f)
    hp["rng"] = np.random.RandomState(seed)
    return hp


def epoch_bounds(epoch: tuple[int | None, int | None], tdim: int) -> tuple[int, int]:
    start, stop = epoch
    return (0 if start is None else int(start), tdim if stop is None else int(stop))


def trial_for_rule(rule: str, hp: dict, batch_size: int):
    hp_local = dict(hp)
    hp_local["rng"] = np.random.RandomState(0)
    return generate_trials(rule, hp_local, "test", noise_on=False, batch_size=batch_size)


def representative_trial_index(trial) -> int:
    """Pick a trial with visible stimulus and response structure."""
    response = np.linalg.norm(trial.y[:, :, 1:], axis=2).max(axis=0)
    stim = np.linalg.norm(trial.x[:, :, 1 : 1 + 2 * trial.n_eachring], axis=2).max(axis=0)
    score = response + stim
    return int(np.nanargmax(score))


def summarize_inputs(trial, idx: int, rule: str) -> dict[str, np.ndarray]:
    n = trial.n_eachring
    x = trial.x[:, idx, :]
    y = trial.y[:, idx, :]
    rule_idx = int(np.argmax(np.nanmax(trial.x[:, :, trial.config["rule_start"] :], axis=(0, 1))))
    rule_channel = trial.config["rule_start"] + rule_idx

    return {
        "fixation input": x[:, 0],
        "stimulus drive": np.linalg.norm(x[:, 1 : 1 + 2 * n], axis=1),
        "rule cue": x[:, rule_channel],
        "fixation target": y[:, 0],
        "response target": np.linalg.norm(y[:, 1:], axis=1),
    }


def draw_epoch_bands(ax, epochs: dict[str, tuple[int | None, int | None]], tdim: int) -> None:
    for epoch, bounds in epochs.items():
        start, stop = epoch_bounds(bounds, tdim)
        color = EPOCH_COLORS.get(epoch, "#eeeeee")
        ax.axvspan(start, stop, color=color, alpha=0.24, lw=0)
        ax.text(
            (start + stop) / 2,
            1.04,
            epoch,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
            transform=ax.get_xaxis_transform(),
        )


def draw_task_trace(ax, rule: str, trial) -> None:
    idx = representative_trial_index(trial)
    traces = summarize_inputs(trial, idx, rule)
    t = np.arange(trial.tdim) * trial.dt / 1000

    draw_epoch_bands(ax, trial.epochs, trial.tdim)
    offsets = {
        "fixation input": 4.0,
        "stimulus drive": 3.0,
        "rule cue": 2.0,
        "fixation target": 1.0,
        "response target": 0.0,
    }
    for label, offset in offsets.items():
        values = traces[label]
        scale = np.nanmax(np.abs(values))
        if scale > 0:
            values = values / scale * 0.72
            color = TRACE_COLORS[label]
            alpha = 1.0
        else:
            color = "#cfcfcf"
            alpha = 0.7
        ax.plot(t, offset + values, lw=1.7, color=color, alpha=alpha)
        ax.text(-0.045, offset, label, ha="right", va="center", fontsize=7, color="#333333")

    display = TASK_LABELS.get(rule, rule_name.get(rule, rule))
    ax.set_title(display, loc="left", fontsize=11, weight="bold", pad=10)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.35, 4.95)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="x", labelsize=8, length=3, color="#888888")
    ax.set_xlabel("time (s)", fontsize=8)


def draw_all_task_epoch_strip(ax, trials_by_rule: dict[str, object], annotate: bool = False) -> None:
    all_rules = DISPLAY_RULE_ORDER
    max_tdim = max(trials_by_rule[rule].tdim for rule in all_rules)
    ax.set_xlim(0, max_tdim + (52 if annotate else 0))
    ax.set_ylim(-0.5, len(all_rules) - 0.5)

    for row, rule in enumerate(all_rules):
        trial = trials_by_rule[rule]
        for epoch, bounds in trial.epochs.items():
            start, stop = epoch_bounds(bounds, trial.tdim)
            ax.barh(
                row,
                stop - start,
                left=start,
                height=0.72,
                color=EPOCH_COLORS.get(epoch, "#dddddd"),
                edgecolor="white",
                linewidth=0.6,
            )
        if annotate:
            ax.text(
                max_tdim + 3,
                row,
                TASK_COMPUTATIONS.get(rule, ""),
                ha="left",
                va="center",
                fontsize=7.8,
                color="#333333",
            )

    labels = [TASK_LABELS.get(rule, rule_name.get(rule, rule)) for rule in all_rules]
    ax.set_yticks(np.arange(len(all_rules)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("time bin", fontsize=8)
    title = "All 15 tasks: actual generated epoch windows"
    if annotate:
        title += " and rule differences"
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="x", labelsize=8, length=3, color="#888888")
    ax.tick_params(axis="y", length=0)
    if annotate:
        ax.axvline(max_tdim, color="#bbbbbb", lw=0.8)
        ax.text(max_tdim + 3, -0.92, "rule/computation", ha="left", va="bottom", fontsize=8, weight="bold")
        add_group_brackets(ax, all_rules)


def add_group_brackets(ax, all_rules: list[str]) -> None:
    groups: list[tuple[str, int, int]] = []
    start = 0
    current = TASK_GROUPS[all_rules[0]]
    for i, rule in enumerate(all_rules[1:], start=1):
        group = TASK_GROUPS[rule]
        if group != current:
            groups.append((current, start, i - 1))
            start = i
            current = group
    groups.append((current, start, len(all_rules) - 1))

    x = -19
    for group, y0, y1 in groups:
        ax.plot([x, x], [y0 - 0.35, y1 + 0.35], color="#666666", lw=0.9, clip_on=False)
        ax.plot([x, x + 2.2], [y0 - 0.35, y0 - 0.35], color="#666666", lw=0.9, clip_on=False)
        ax.plot([x, x + 2.2], [y1 + 0.35, y1 + 0.35], color="#666666", lw=0.9, clip_on=False)
        ax.text(
            x - 2.2,
            (y0 + y1) / 2,
            group,
            ha="right",
            va="center",
            rotation=90,
            fontsize=7.4,
            color="#555555",
            clip_on=False,
        )


def add_epoch_legend(fig) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
        for _, color in EPOCH_COLORS.items()
    ]
    fig.legend(
        handles,
        list(EPOCH_COLORS.keys()),
        loc="lower center",
        ncol=len(EPOCH_COLORS),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )


def make_figure(model_dir: Path, output_dir: Path, seed: int, batch_size: int) -> Path:
    hp = load_hp(model_dir, seed)
    plotted_trials = {rule: trial_for_rule(rule, hp, batch_size) for rule in TASKS_TO_PLOT}
    all_trials = {rule: trial_for_rule(rule, hp, batch_size) for rule in rules_dict["all"]}

    plt.rcParams.update(
        {
            "font.family": "Helvetica",
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.8, 7.2), constrained_layout=False)
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.0, 1.0, 1.0, 1.35], hspace=0.54)

    for i, rule in enumerate(TASKS_TO_PLOT):
        ax = fig.add_subplot(gs[i, 0])
        draw_task_trace(ax, rule, plotted_trials[rule])
        if i < len(TASKS_TO_PLOT) - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])

    ax_strip = fig.add_subplot(gs[3, 0])
    draw_all_task_epoch_strip(ax_strip, all_trials)
    add_epoch_legend(fig)

    fig.suptitle(
        "Task paradigm from generated trials and epoch metadata",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.02,
        0.948,
        "Traces are Trial.x and Trial.y from the canonical softplus model; shaded bands are Trial.epochs.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4a4a4a",
    )
    fig.subplots_adjust(top=0.86, bottom=0.085, left=0.18, right=0.985)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig1a_real_task_paradigm.png"
    svg_path = output_dir / "fig1a_real_task_paradigm.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path


def make_strip_figure(model_dir: Path, output_dir: Path, seed: int, batch_size: int) -> Path:
    hp = load_hp(model_dir, seed)
    all_trials = {rule: trial_for_rule(rule, hp, batch_size) for rule in rules_dict["all"]}

    plt.rcParams.update(
        {
            "font.family": "Helvetica",
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    draw_all_task_epoch_strip(ax, all_trials, annotate=True)
    add_epoch_legend(fig)
    fig.text(
        0.02,
        0.965,
        "Task suite from generated trials",
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.02,
        0.915,
        "Bars show real Trial.epochs; right labels mark the computation that differs when timing is reused.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4a4a4a",
    )
    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.22, right=0.985)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig1a_all_tasks_real_paradigm.png"
    svg_path = output_dir / "fig1a_all_tasks_real_paradigm.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=80)
    args = parser.parse_args()

    out = make_figure(args.model_dir, args.output_dir, args.seed, args.batch_size)
    print(out)
    strip_out = make_strip_figure(args.model_dir, args.output_dir, args.seed, args.batch_size)
    print(strip_out)


if __name__ == "__main__":
    main()
