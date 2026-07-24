"""Read-only integrity audit for the deterministic pi0.5 replan pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.passes.append(message)
        else:
            self.failures.append(message)


def one_trace_per_seed(
    audit: Audit,
    trace_dir: Path,
    expected_seeds: list[int],
) -> dict[int, dict]:
    traces: dict[int, dict] = {}
    paths_by_seed: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(trace_dir.glob("scene_*.json")):
        trace = load_json(path)
        paths_by_seed[int(trace["scene_seed"])].append(path)
        traces[int(trace["scene_seed"])] = trace
    duplicates = {
        seed: [str(path) for path in paths]
        for seed, paths in paths_by_seed.items()
        if len(paths) != 1
    }
    audit.require(not duplicates, f"{trace_dir.name}: exactly one trace per seed")
    audit.require(
        set(traces) == set(expected_seeds),
        f"{trace_dir.name}: trace seeds equal the fixed manifest",
    )
    audit.require(
        len(traces) == len(expected_seeds),
        f"{trace_dir.name}: {len(expected_seeds)} traces",
    )
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--r0", type=int, default=25)
    parser.add_argument(
        "--expected-rs",
        type=int,
        nargs="+",
        default=[10, 15, 25, 30, 35, 40],
    )
    parser.add_argument("--eval-num", type=int, default=100)
    parser.add_argument(
        "--paired-summary",
        type=Path,
        action="append",
        default=[],
        help=(
            "Repeat for every paired round. If omitted, summaries under "
            "ROOT/paired_replan_node_eval*/ are discovered."
        ),
    )
    parser.add_argument(
        "--scene-outcomes-csv",
        type=Path,
        default=None,
        help="Defaults to ROOT/scene_outcome_plot/scene_outcomes_by_r.csv.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    audit = Audit()
    expected_rs = sorted(set(args.expected_rs))

    grid = load_json(root / "grid_results.json")
    seed_manifest = load_json(root / f"seed_manifest_seed0_eval{args.eval_num}.json")
    resolved = load_json(root / "resolved_episode_manifest.json")
    seeds = [int(seed) for seed in seed_manifest["scene_seeds"]]
    prompts = list(resolved["episode_instructions"])
    prompt_by_seed = dict(zip(seeds, prompts, strict=True))

    audit.require(grid["eval_num"] == args.eval_num, "grid eval_num matches")
    audit.require(len(seeds) == args.eval_num, f"manifest has {args.eval_num} scenes")
    audit.require(len(set(seeds)) == len(seeds), "manifest scene seeds are unique")
    audit.require(
        [int(seed) for seed in resolved["scene_seeds"]] == seeds,
        "resolved prompts align with the ordered seed manifest",
    )
    audit.require(
        len(prompts) == args.eval_num and all(isinstance(p, str) and p for p in prompts),
        "all resolved prompt plaintext values are non-empty",
    )

    traces_by_r: dict[int, dict[int, dict]] = {}
    success_by_r: dict[int, int] = {}
    prompt_errors: list[str] = []
    for r in expected_rs:
        trace_dir = root / "traces" / f"H{args.horizon}_r{r}"
        traces = one_trace_per_seed(audit, trace_dir, seeds)
        traces_by_r[r] = traces
        success_by_r[r] = sum(
            int(trace["episode_metrics"]["episode_success"])
            for trace in traces.values()
        )
        for seed in seeds:
            trace = traces.get(seed)
            if trace is None:
                continue
            prompt = trace.get("instruction")
            if prompt != prompt_by_seed[seed]:
                prompt_errors.append(f"seed={seed},r={r}: plaintext")
                continue
            expected_fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            fingerprints = {
                chunk.get("prompt_fingerprint") for chunk in trace.get("chunks", [])
            }
            if fingerprints != {expected_fingerprint}:
                prompt_errors.append(f"seed={seed},r={r}: fingerprint")
    audit.require(
        not prompt_errors,
        "prompt plaintext and SHA-256 fingerprint match in every grid trace",
    )
    audit.require(
        sorted(
            int(key.rsplit("_r", maxsplit=1)[1])
            for key in grid["trials"]
            if key.startswith(f"H{args.horizon}_r")
        )
        == expected_rs,
        "grid_results contains exactly the expected H/r configurations",
    )
    audit.require(
        all(
            grid["trials"][f"H{args.horizon}_r{r}"]["status"] == "complete"
            and int(grid["trials"][f"H{args.horizon}_r{r}"]["episodes"])
            == args.eval_num
            and int(grid["trials"][f"H{args.horizon}_r{r}"]["successes"])
            == success_by_r[r]
            for r in expected_rs
        ),
        "grid trial status, episode totals, and trace success totals agree",
    )
    server_logs = sorted((root / "logs").glob("trial_*_server.log"))
    audit.require(
        len(server_logs) == len(expected_rs)
        and all(
            "Enabled deterministic PyTorch inference kernels" in path.read_text()
            for path in server_logs
        ),
        "every discovery server log confirms deterministic PyTorch kernels",
    )

    recovery_path = root / "prompt_recovery_report.json"
    if recovery_path.exists():
        recovery = load_json(recovery_path)
        audit.require(
            recovery.get("all_exactly_recovered") is True
            and int(recovery.get("exactly_recovered", -1)) == args.eval_num,
            "prompt recovery report passed for every scene",
        )
    recovered_manifest_path = root / "recovered_resolved_episode_manifest.json"
    if recovered_manifest_path.exists():
        recovered = load_json(recovered_manifest_path)
        audit.require(
            [int(seed) for seed in recovered["scene_seeds"]] == seeds
            and recovered["episode_instructions"] == prompts,
            "recovered prompt manifest exactly matches the grid resolved manifest",
        )

    subset_path = root / f"seed_manifest_r{args.r0}_fail_other_success.json"
    subset = load_json(subset_path)
    subset_seeds = [int(seed) for seed in subset["scene_seeds"]]
    expected_subset = [
        seed
        for seed in seeds
        if not bool(
            traces_by_r[args.r0][seed]["episode_metrics"]["episode_success"]
        )
        and any(
            bool(traces_by_r[r][seed]["episode_metrics"]["episode_success"])
            for r in expected_rs
            if r != args.r0
        )
    ]
    audit.require(
        subset_seeds == expected_subset,
        "contrast subset is exactly r0-fail/other-r-success in manifest order",
    )

    controls_path = root / f"replan_window_controls_r{args.r0}_fail_other_success.json"
    controls = load_json(controls_path)
    control_scenes = controls["scenes"]
    configuration = controls["configuration"]
    audit.require(
        [int(scene["scene_seed"]) for scene in control_scenes] == subset_seeds,
        "window-control scenes align with the contrast subset",
    )
    audit.require(
        configuration["horizon"] == args.horizon
        and configuration["time_tolerance_actions"] == 2
        and configuration["window_radius_actions"] == 2
        and configuration["state_lookback_actions"] == 5
        and float(configuration["max_state_distance"]) == 0.35
        and configuration["max_windows_per_scene"] == 3,
        "candidate-window configuration matches the frozen protocol",
    )
    audit.require(
        all(len(scene["candidate_windows"]) <= 3 for scene in control_scenes),
        "every scene has at most three selected window families",
    )

    nodes_path = root / f"replan_nodes_r{args.r0}_fail_other_success.json"
    nodes = load_json(nodes_path)
    node_scenes = nodes["scenes"]
    audit.require(
        [int(scene["scene_seed"]) for scene in node_scenes] == subset_seeds,
        "node scenes align with the contrast subset",
    )
    audit.require(
        int(nodes["max_nodes_per_scene"]) == 10
        and all(int(scene["node_count"]) <= 10 for scene in node_scenes),
        "every scene has at most ten concrete nodes",
    )
    primary_node_count = sum(int(scene["node_count"]) for scene in node_scenes)

    summary_paths = list(args.paired_summary)
    if not summary_paths:
        summary_paths = sorted(
            root.glob("paired_replan_node_eval*/paired_summary.json")
        )
    evaluated_seeds: set[int] = set()
    rescue_nodes_by_seed: dict[int, set[int]] = defaultdict(set)
    paired_counts: list[dict[str, int | str]] = []
    for summary_path in summary_paths:
        summary = load_json(summary_path)
        complete = [
            pair for pair in summary["pairs"] if pair.get("status") == "complete"
        ]
        valid = [pair for pair in complete if pair.get("valid_pair")]
        audit.require(
            len(complete) == int(summary["completed_pairs"]),
            f"{summary_path.parent.name}: completed-pair count is internally consistent",
        )
        audit.require(
            len(valid) == int(summary["valid_pairs"])
            and int(summary["invalid_pairs"]) == len(complete) - len(valid),
            f"{summary_path.parent.name}: validity counts are internally consistent",
        )
        for pair in valid:
            seed = int(pair["scene_seed"])
            evaluated_seeds.add(seed)
            if pair["transition"] == "rescue":
                rescue_nodes_by_seed[seed].add(int(pair["replan_after_actions"]))
        paired_counts.append(
            {
                "round": summary_path.parent.name,
                "valid_pairs": len(valid),
                "rescues": sum(pair["transition"] == "rescue" for pair in valid),
            }
        )

    baseline_success_seeds = {
        seed
        for seed, trace in traces_by_r[args.r0].items()
        if bool(trace["episode_metrics"]["episode_success"])
    }
    rescued_seeds = set(rescue_nodes_by_seed)
    audit.require(
        not (baseline_success_seeds & rescued_seeds),
        "all reported rescues originate from r0 grid failures",
    )
    final_success_seeds = baseline_success_seeds | rescued_seeds

    scene_outcomes_csv = (
        args.scene_outcomes_csv
        if args.scene_outcomes_csv is not None
        else root / "scene_outcome_plot" / "scene_outcomes_by_r.csv"
    )
    if scene_outcomes_csv.exists() and summary_paths:
        with scene_outcomes_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        replan_rows = [
            row for row in rows if row["row_label"] == f"{args.r0} + one replan"
        ]
        csv_rescues = {
            int(row["scene_seed"]): {
                int(node) for node in row["rescue_nodes"].split(",") if node
            }
            for row in replan_rows
            if row["rescue_nodes"]
        }
        audit.require(
            len(replan_rows) == args.eval_num
            and sum(int(row["success"]) for row in replan_rows)
            == len(final_success_seeds),
            "scene-outcome one-replan row matches baseline plus distinct rescued scenes",
        )
        audit.require(
            csv_rescues == dict(rescue_nodes_by_seed),
            "scene-outcome Rescue-node labels match all paired summaries",
        )

    for message in audit.passes:
        print(f"PASS  {message}")
    for message in audit.failures:
        print(f"FAIL  {message}")
    print(
        json.dumps(
            {
                "root": str(root),
                "grid_episodes": len(expected_rs) * args.eval_num,
                "success_by_r": success_by_r,
                "contrast_scenes": len(subset_seeds),
                "primary_candidate_nodes": primary_node_count,
                "paired_rounds": paired_counts,
                "paired_evaluated_scenes": len(evaluated_seeds),
                "distinct_rescued_scenes": len(rescued_seeds),
                "final_r0_plus_one_replan_successes": len(final_success_seeds),
                "checks_passed": len(audit.passes),
                "checks_failed": len(audit.failures),
            },
            indent=2,
        )
    )
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
