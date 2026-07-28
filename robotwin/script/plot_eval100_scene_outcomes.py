"""Plot all complete 100-scene RoboTwin evaluations in one figure.

The rendering follows ``plot_pi05_scene_outcomes.py``: green cells are
successful episodes, salmon cells are failures, and each task also gets an
``any r`` row.  Replanning experiments are deliberately not merged into the
figure.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


RUN_NAME = re.compile(r"H(?P<horizon>\d+)_r(?P<r>\d+)$")
SCENE_SEED = re.compile(r'"scene_seed"\s*:\s*(\d+)')
EPISODE_SUCCESS = re.compile(r'"episode_success"\s*:\s*(true|false|0|1)')


@dataclass(frozen=True)
class TaskOutcomes:
    name: str
    trace_root: Path
    rs: list[int]
    seeds: list[int]
    rows: list[list[int]]


def parse_trace(path: Path) -> tuple[int, int]:
    """Read the two top-level outcome fields without loading a large trace."""
    with path.open() as handle:
        prefix = handle.read(65536)
    seed_match = SCENE_SEED.search(prefix)
    success_match = EPISODE_SUCCESS.search(prefix)
    if seed_match and success_match:
        return (
            int(seed_match.group(1)),
            int(success_match.group(1) in {"true", "1"}),
        )

    # This is only a compatibility fallback if the fields move farther down in
    # a future trace schema.
    trace = json.loads(path.read_text())
    return (
        int(trace["scene_seed"]),
        int(trace["episode_metrics"]["episode_success"]),
    )


def discover_complete_tasks(eval_root: Path, eval_count: int) -> list[TaskOutcomes]:
    tasks = []
    for trace_root in sorted(eval_root.rglob("traces")):
        runs = []
        for run_dir in trace_root.iterdir():
            match = RUN_NAME.fullmatch(run_dir.name) if run_dir.is_dir() else None
            if match:
                runs.append((int(match.group("r")), run_dir))
        if not runs:
            continue

        runs.sort()
        trace_paths = [
            sorted(run_dir.glob("scene_*.json")) for _, run_dir in runs
        ]
        if not all(len(paths) == eval_count for paths in trace_paths):
            continue

        rows_by_r = []
        expected_seeds = None
        for (r, run_dir), paths in zip(runs, trace_paths):
            outcomes = dict(parse_trace(path) for path in paths)
            if len(outcomes) != eval_count:
                raise ValueError(
                    f"{run_dir} has {len(outcomes)} unique scene seeds; "
                    f"expected {eval_count}"
                )
            seeds = sorted(outcomes)
            if expected_seeds is None:
                expected_seeds = seeds
            elif seeds != expected_seeds:
                raise ValueError(
                    f"Scene seeds differ across r rows under {trace_root}"
                )
            rows_by_r.append([outcomes[seed] for seed in seeds])

        relative = trace_root.relative_to(eval_root)
        task_name = relative.parts[0]
        tasks.append(
            TaskOutcomes(
                name=task_name,
                trace_root=trace_root,
                rs=[r for r, _ in runs],
                seeds=expected_seeds or [],
                rows=rows_by_r,
            )
        )

    duplicate_names = {
        task.name for task in tasks if sum(t.name == task.name for t in tasks) > 1
    }
    if duplicate_names:
        names = ", ".join(sorted(duplicate_names))
        raise ValueError(f"Multiple complete eval roots found for task(s): {names}")
    return tasks


def rate_label(name: str, row: list[int]) -> str:
    successes = sum(row)
    return f"{name}  {successes}/{len(row)} ({100 * successes / len(row):.0f}%)"


def render(tasks: list[TaskOutcomes], output_path: Path) -> None:
    if not tasks:
        raise FileNotFoundError("No complete 100-scene evaluations found")

    width = max(len(task.seeds) for task in tasks)
    figure, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(20, max(11, 2.15 * len(tasks))),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    cmap = ListedColormap(["#fcae91", "#a1d99b"])

    for axis, task in zip(axes[:, 0], tasks):
        any_r_row = [
            max(task.rows[row_index][column_index] for row_index in range(len(task.rows)))
            for column_index in range(len(task.seeds))
        ]
        display_rows = [any_r_row, *task.rows]
        labels = [rate_label("any r", any_r_row)]
        labels.extend(
            rate_label(f"r={r}", row) for r, row in zip(task.rs, task.rows)
        )

        axis.imshow(
            display_rows,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        axis.set_title(task.name, loc="left", fontsize=12, fontweight="bold", pad=4)
        axis.set_yticks(range(len(display_rows)), labels=labels)
        axis.tick_params(axis="y", labelsize=9)
        axis.set_xlim(-0.5, width - 0.5)
        for edge in [value - 0.5 for value in range(width + 1)]:
            axis.axvline(edge, color="white", linewidth=0.18)
        for edge in [value - 0.5 for value in range(len(display_rows) + 1)]:
            axis.axhline(edge, color="white", linewidth=0.7)

    tick_positions = list(range(0, width, 10))
    axes[-1, 0].set_xticks(
        tick_positions,
        labels=[str(position + 1) for position in tick_positions],
    )
    axes[-1, 0].set_xlabel("Scene index (fixed manifest order)")
    figure.suptitle(
        "RoboTwin 100-scene evaluation outcomes by task and execution length",
        fontsize=16,
    )
    figure.legend(
        handles=[
            Patch(color="#a1d99b", label="success"),
            Patch(color="#fcae91", label="failure"),
        ],
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.995, 0.997),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-count", type=int, default=100)
    args = parser.parse_args()

    tasks = discover_complete_tasks(args.eval_root, args.eval_count)
    render(tasks, args.output)
    for task in tasks:
        rates = ", ".join(
            f"r={r}: {sum(row)}/{len(row)}"
            for r, row in zip(task.rs, task.rows)
        )
        print(f"{task.name}: {rates}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
