"""Render paired RoboTwin scene outcomes as an r-by-scene success matrix."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def parse_r(path: Path) -> int:
    return int(path.parent.name.split("_r", maxsplit=1)[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--paired-summary",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional paired_summary.json. Repeat to merge multiple node rounds "
            "into one additional r=25 + one-replan row."
        ),
    )
    parser.add_argument("--replan-r", type=int, default=25)
    parser.add_argument("--replan-row-label", default="25 + one replan")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(args.trace_root.glob("H*_r*/scene_*.json")):
        trace = json.loads(path.read_text())
        records.append({
            "r": parse_r(path), "scene_seed": int(trace["scene_seed"]),
            "success": int(trace["episode_metrics"]["episode_success"]),
        })
    if not records:
        raise FileNotFoundError(f"No traces found under {args.trace_root}")
    rs = sorted({record["r"] for record in records})
    seeds = sorted({record["scene_seed"] for record in records})
    positions = {seed: index for index, seed in enumerate(seeds)}
    r_positions = {r: index for index, r in enumerate(rs)}
    # -1 deliberately stays white if an r/scene trace is absent.
    matrix = np.full((len(rs), len(seeds)), -1, dtype=int)
    for record in records:
        matrix[r_positions[record["r"]], positions[record["scene_seed"]]] = record["success"]
        record["scene_index"] = positions[record["scene_seed"]] + 1

    # A scene-level row exposes intrinsic hardness: green means at least one
    # evaluated execution length succeeded for that fixed scene.
    any_r_success = np.max(matrix, axis=0, keepdims=True)
    display_rows = [any_r_success[0]]
    any_r_count = int(np.sum(any_r_success[0] == 1))
    yticklabels: list[str] = [
        f"any grid r success ({any_r_count}/{len(seeds)}, "
        f"{100 * any_r_count / len(seeds):.0f}%)"
    ]
    replan_row_index = None
    rescue_nodes_by_seed: dict[int, set[int]] = defaultdict(set)
    evaluated_replan_seeds: set[int] = set()
    invalid_replan_pairs = 0

    for r in rs:
        grid_row = matrix[r_positions[r]]
        display_rows.append(grid_row)
        success_count = int(np.sum(grid_row == 1))
        yticklabels.append(
            f"{r} ({success_count}/{len(seeds)}, "
            f"{100 * success_count / len(seeds):.0f}%)"
        )
        if r == args.replan_r and args.paired_summary:
            # Start from the complete r=25 grid outcome. The paired experiment
            # reruns only a selected subset of its control failures, so failures
            # outside that subset must remain failures rather than being treated
            # as successful by omission.
            replan_row = grid_row.copy()
            for summary_path in args.paired_summary:
                summary = json.loads(summary_path.read_text())
                for pair in summary["pairs"]:
                    if pair.get("status") != "complete":
                        continue
                    if not pair.get("valid_pair"):
                        invalid_replan_pairs += 1
                        continue
                    seed = int(pair["scene_seed"])
                    if seed not in positions:
                        raise ValueError(
                            f"{summary_path}: paired seed {seed} is absent from grid traces"
                        )
                    evaluated_replan_seeds.add(seed)
                    if pair["transition"] == "rescue":
                        rescue_nodes_by_seed[seed].add(
                            int(pair["replan_after_actions"])
                        )
            for seed in evaluated_replan_seeds:
                replan_row[positions[seed]] = int(bool(rescue_nodes_by_seed[seed]))
            display_rows.append(replan_row)
            replan_success_count = int(np.sum(replan_row == 1))
            yticklabels.append(
                f"{args.replan_row_label} "
                f"({replan_success_count}/{len(seeds)}, "
                f"{100 * replan_success_count / len(seeds):.0f}%)"
            )
            replan_row_index = len(display_rows) - 1

    if args.paired_summary and replan_row_index is None:
        raise ValueError(
            f"Requested one-replan row after r={args.replan_r}, "
            f"but grid r values are {rs}"
        )

    display_matrix = np.stack(display_rows)
    figure_height = 8.0 if args.paired_summary else 4.2
    figure, axis = plt.subplots(
        figsize=(20, figure_height), constrained_layout=True
    )
    axis.imshow(
        display_matrix,
        aspect="auto",
        interpolation="none",
        cmap=ListedColormap(["white", "#fcae91", "#a1d99b"]),
        vmin=-1,
        vmax=1,
    )
    axis.set(
        xlabel="Scene index (fixed manifest order)", ylabel="Execution length r",
        yticks=range(len(display_rows)), yticklabels=yticklabels,
        xticks=np.arange(0, len(seeds), 10), xticklabels=np.arange(1, len(seeds) + 1, 10),
        title=args.title or "Per-scene evaluation outcome",
    )
    for edge in np.arange(-0.5, len(seeds), 1):
        axis.axvline(edge, color="white", linewidth=0.18)
    for edge in np.arange(-0.5, len(display_rows), 1):
        axis.axhline(edge, color="white", linewidth=0.8)
    legend = [
        Patch(color="#a1d99b", label="success"),
        Patch(color="#fcae91", label="failure"),
    ]
    axis.legend(handles=legend, loc="upper right", ncol=len(legend))

    if replan_row_index is not None:
        rescue_lines = [
            f"seed{seed}: rescue=" + ", ".join(map(str, sorted(nodes)))
            for seed, nodes in sorted(rescue_nodes_by_seed.items())
            if nodes
        ]
        column_count = 3
        columns = [
            rescue_lines[index::column_count]
            for index in range(column_count)
        ]
        axis.text(
            0.5,
            -0.12,
            "One-replan Rescue nodes",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        for column_index, lines in enumerate(columns):
            axis.text(
                0.03 + column_index * 0.34,
                -0.17,
                "\n".join(lines),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                linespacing=1.35,
                color="#00441b",
            )
        axis.text(
            0.5,
            -0.40,
            (
                f"One-replan row: {int(np.sum(matrix[r_positions[args.replan_r]] == 1))} "
                f"r={args.replan_r} grid successes retained; "
                f"{len(evaluated_replan_seeds)} control-failure scenes evaluated. "
                f"Rescue nodes merged across "
                f"{len(args.paired_summary)} round(s). "
                f"Invalid pairs excluded: {invalid_replan_pairs}."
            ),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    figure.savefig(args.output_dir / "scene_outcomes_by_r.png", dpi=220)
    figure.savefig(args.output_dir / "scene_outcomes_by_r.pdf")
    plt.close(figure)

    for record in records:
        record["row_label"] = str(record["r"])
        record["evaluated"] = 1
        record["rescue_nodes"] = ""
    if args.paired_summary:
        for seed in seeds:
            evaluated = seed in evaluated_replan_seeds
            records.append(
                {
                    "scene_index": positions[seed] + 1,
                    "scene_seed": seed,
                    "r": args.replan_r,
                    "success": (
                        int(bool(rescue_nodes_by_seed[seed]))
                        if evaluated
                        else int(matrix[r_positions[args.replan_r], positions[seed]])
                    ),
                    "row_label": args.replan_row_label,
                    "evaluated": int(evaluated),
                    "rescue_nodes": ",".join(
                        map(str, sorted(rescue_nodes_by_seed[seed]))
                    ),
                }
            )
    with (args.output_dir / "scene_outcomes_by_r.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene_index",
                "scene_seed",
                "row_label",
                "r",
                "success",
                "evaluated",
                "rescue_nodes",
            ],
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                records,
                key=lambda row: (
                    rs.index(row["r"])
                    + (
                        0.5
                        if row["row_label"] == args.replan_row_label
                        else 0
                    ),
                    row["scene_index"],
                ),
            )
        )


if __name__ == "__main__":
    main()
