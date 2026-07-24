"""Select a second, outcome-adaptive node set for no-rescue scenes.

The first-round treatment outcomes are used only to identify scenes with no
rescued node.  Within those scenes, nodes are selected without consulting any
second-round outcomes:

1. prefer previously untested exact mapped times from every accepted window
   family, including families beyond the first-round three-family cap;
2. if fewer than the requested count remain, fill from untested integer times
   inside accepted mapped windows;
3. retain the original r0-boundary exclusion and rank by distinct successful-r
   support, state distance, proximity to an exact mapped time, family rank, and
   earlier action time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def boundary_distance(time: int, boundaries: list[int]) -> int:
    return min(abs(time - boundary) for boundary in boundaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--prior-nodes", type=Path, required=True)
    parser.add_argument("--paired-summary", type=Path, required=True)
    parser.add_argument("--source-seed-manifest", type=Path, required=True)
    parser.add_argument("--source-resolved-manifest", type=Path, required=True)
    parser.add_argument("--nodes-per-scene", type=int, default=5)
    parser.add_argument("--nodes-output", type=Path, required=True)
    parser.add_argument("--seed-output", type=Path, required=True)
    parser.add_argument("--resolved-output", type=Path, required=True)
    args = parser.parse_args()

    controls = load(args.controls)
    prior_nodes = load(args.prior_nodes)
    summary = load(args.paired_summary)
    source_seeds = load(args.source_seed_manifest)
    source_resolved = load(args.source_resolved_manifest)

    if source_seeds["scene_seeds"] != source_resolved["scene_seeds"]:
        raise ValueError("Source seed and resolved manifests are not aligned")
    if controls["r0"] != 25:
        raise ValueError(f"Expected fixed r0=25, got {controls['r0']}")

    prior_by_seed = {
        int(scene["scene_seed"]): {int(node) for node in scene["replan_nodes"]}
        for scene in prior_nodes["scenes"]
    }
    control_by_seed = {
        int(scene["scene_seed"]): scene for scene in controls["scenes"]
    }
    summary_by_seed = {
        int(scene["scene_seed"]): scene for scene in summary["by_scene"]
    }
    prompt_by_seed = dict(
        zip(
            [int(seed) for seed in source_resolved["scene_seeds"]],
            source_resolved["episode_instructions"],
            strict=True,
        )
    )
    no_rescue_seeds = [
        int(seed)
        for seed in source_seeds["scene_seeds"]
        if not summary_by_seed[int(seed)]["rescued_nodes"]
    ]
    if not no_rescue_seeds:
        raise ValueError("No no-rescue scenes were found")

    delta = int(controls["configuration"]["time_tolerance_actions"])
    selected_scenes = []
    for seed in no_rescue_seeds:
        summary_scene = summary_by_seed[seed]
        if (
            summary_scene["valid_pairs"]
            != summary_scene["eligible_control_failure_pairs"]
        ):
            raise ValueError(
                f"seed={seed}: round-one pairs were not all valid control failures"
            )

        scene = control_by_seed[seed]
        prior = prior_by_seed[seed]
        boundaries = [int(value) for value in scene["r0_replan_times"]]
        failure_actions = int(scene["failure_actions"])
        families = [
            *scene["candidate_windows"],
            *scene["unselected_window_families"],
        ]

        rows: dict[int, dict[str, Any]] = {}
        for global_rank, family in enumerate(families, start=1):
            for evidence in family["evidence"]:
                mapped_time = int(evidence["mapped_time"])
                start, end = [int(value) for value in evidence["mapped_window"]]
                for time in range(start, end + 1):
                    if (
                        time <= 0
                        or time >= failure_actions
                        or time in prior
                        or boundary_distance(time, boundaries) <= delta
                    ):
                        continue
                    row = rows.setdefault(
                        time,
                        {
                            "replan_node": time,
                            "supporting_success_rs": set(),
                            "exact_supporting_success_rs": set(),
                            "best_state_distance": float("inf"),
                            "minimum_offset_from_mapped_time": failure_actions,
                            "family_ranks": set(),
                            "source_mapped_times": set(),
                        },
                    )
                    success_r = int(evidence["success_r"])
                    row["supporting_success_rs"].add(success_r)
                    if time == mapped_time:
                        row["exact_supporting_success_rs"].add(success_r)
                    row["best_state_distance"] = min(
                        row["best_state_distance"],
                        float(evidence["state_distance"]),
                    )
                    row["minimum_offset_from_mapped_time"] = min(
                        row["minimum_offset_from_mapped_time"],
                        abs(time - mapped_time),
                    )
                    row["family_ranks"].add(global_rank)
                    row["source_mapped_times"].add(mapped_time)

        exact = [
            row for row in rows.values() if row["exact_supporting_success_rs"]
        ]
        offsets = [
            row for row in rows.values() if not row["exact_supporting_success_rs"]
        ]

        def priority(row: dict[str, Any]) -> tuple:
            return (
                -len(row["supporting_success_rs"]),
                row["best_state_distance"],
                row["minimum_offset_from_mapped_time"],
                min(row["family_ranks"]),
                row["replan_node"],
            )

        chosen = sorted(exact, key=priority)[: args.nodes_per_scene]
        exact_count = len(chosen)
        if len(chosen) < args.nodes_per_scene:
            chosen.extend(
                sorted(offsets, key=priority)[
                    : args.nodes_per_scene - len(chosen)
                ]
            )
        if len(chosen) != args.nodes_per_scene:
            raise ValueError(
                f"seed={seed}: only {len(chosen)} eligible second-round nodes"
            )

        details = []
        for row in sorted(chosen, key=lambda item: item["replan_node"]):
            details.append(
                {
                    "replan_node": row["replan_node"],
                    "selection_tier": (
                        "untested_exact_mapped_time"
                        if row["exact_supporting_success_rs"]
                        else "accepted_window_offset_fill"
                    ),
                    "supporting_success_rs": sorted(
                        row["supporting_success_rs"]
                    ),
                    "exact_supporting_success_rs": sorted(
                        row["exact_supporting_success_rs"]
                    ),
                    "distinct_r_support": len(row["supporting_success_rs"]),
                    "best_state_distance": row["best_state_distance"],
                    "minimum_offset_from_mapped_time": row[
                        "minimum_offset_from_mapped_time"
                    ],
                    "family_ranks": sorted(row["family_ranks"]),
                    "source_mapped_times": sorted(row["source_mapped_times"]),
                }
            )
        selected_nodes = [detail["replan_node"] for detail in details]
        if prior.intersection(selected_nodes):
            raise AssertionError(f"seed={seed}: prior node leaked into round two")

        selected_scenes.append(
            {
                "scene_seed": seed,
                "r0": 25,
                "successful_rs": scene["successful_rs"],
                "prior_tested_nodes": sorted(prior),
                "replan_nodes": selected_nodes,
                "node_count": len(selected_nodes),
                "exact_mapped_nodes_selected": exact_count,
                "window_offset_nodes_selected": len(selected_nodes) - exact_count,
                "node_details": details,
            }
        )

    nodes_payload = {
        "purpose": (
            "Outcome-adaptive second-round single-replan evaluation for scenes "
            "with no first-round rescued node."
        ),
        "selection_population": (
            "Round-one scenes with all valid control-failure pairs and zero rescue"
        ),
        "selection_rule": (
            "Exclude prior nodes; prefer untested exact mapped times across all "
            "accepted window families; fill from accepted-window integer offsets; "
            "retain the r0 boundary delta gate; rank by distinct-r support, state "
            "distance, mapped-time offset, family rank, then earlier time."
        ),
        "source_controls": str(args.controls),
        "source_prior_nodes": str(args.prior_nodes),
        "source_paired_summary": str(args.paired_summary),
        "nodes_per_scene": args.nodes_per_scene,
        "scene_count": len(selected_scenes),
        "total_nodes": sum(scene["node_count"] for scene in selected_scenes),
        "scenes": selected_scenes,
    }
    seed_payload = {
        "task_name": source_seeds["task_name"],
        "task_config": source_seeds["task_config"],
        "base_seed": int(source_seeds["base_seed"]),
        "purpose": (
            "Outcome-adaptive round-two no-rescue scene manifest; immutable order"
        ),
        "source_manifest": str(args.source_seed_manifest),
        "scene_seeds": no_rescue_seeds,
    }
    resolved_payload = {
        "task_name": source_resolved["task_name"],
        "task_config": source_resolved["task_config"],
        "source_seed_manifest": str(args.seed_output),
        "scene_seeds": no_rescue_seeds,
        "episode_instructions": [prompt_by_seed[seed] for seed in no_rescue_seeds],
    }

    write_new(args.nodes_output, nodes_payload)
    write_new(args.seed_output, seed_payload)
    write_new(args.resolved_output, resolved_payload)
    print(
        f"Selected {nodes_payload['total_nodes']} nodes for "
        f"{nodes_payload['scene_count']} scenes"
    )


if __name__ == "__main__":
    main()
