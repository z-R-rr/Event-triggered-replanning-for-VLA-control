"""Extract at most N concrete replan nodes per scene from window controls."""

import argparse
import csv
import json
from pathlib import Path


def choose_nodes(families: list[dict], limit: int) -> tuple[list[int], list[dict]]:
    node_rows: dict[int, dict] = {}
    representatives: set[int] = set()
    for family in families:
        by_time: dict[int, list[dict]] = {}
        for evidence in family["evidence"]:
            by_time.setdefault(int(evidence["mapped_time"]), []).append(evidence)
        if not by_time:
            continue
        representative = min(
            by_time,
            key=lambda time: (
                abs(time - int(family["representative_time"])),
                min(row["state_distance"] for row in by_time[time]),
                time,
            ),
        )
        representatives.add(representative)
        for time, evidence in by_time.items():
            row = node_rows.setdefault(
                time,
                {
                    "replan_node": time,
                    "supporting_success_rs": set(),
                    "best_state_distance": float("inf"),
                    "window_ranks": set(),
                    "window_distinct_r_support_max": 0,
                },
            )
            row["supporting_success_rs"].update(int(item["success_r"]) for item in evidence)
            row["best_state_distance"] = min(
                row["best_state_distance"], *(float(item["state_distance"]) for item in evidence)
            )
            row["window_ranks"].add(int(family["rank"]))
            row["window_distinct_r_support_max"] = max(
                row["window_distinct_r_support_max"], int(family["distinct_r_support"])
            )

    def priority(row: dict) -> tuple:
        return (
            -len(row["supporting_success_rs"]),
            row["best_state_distance"],
            min(row["window_ranks"]),
            row["replan_node"],
        )

    selected = set(representatives)
    for row in sorted(node_rows.values(), key=priority):
        if len(selected) >= limit:
            break
        selected.add(row["replan_node"])
    if len(selected) > limit:
        # This can occur only when the number of window families exceeds the
        # requested limit. Keep the strongest representative nodes.
        selected = {
            row["replan_node"]
            for row in sorted(
                (node_rows[time] for time in selected),
                key=priority,
            )[:limit]
        }

    details = []
    for time in sorted(selected):
        row = node_rows[time]
        details.append(
            {
                "replan_node": time,
                "supporting_success_rs": sorted(row["supporting_success_rs"]),
                "distinct_r_support_at_node": len(row["supporting_success_rs"]),
                "best_state_distance": row["best_state_distance"],
                "window_ranks": sorted(row["window_ranks"]),
                "window_distinct_r_support_max": row["window_distinct_r_support_max"],
                "is_window_representative": time in representatives,
            }
        )
    return sorted(selected), details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--max-nodes-per-scene", type=int, default=10)
    args = parser.parse_args()

    controls = json.loads(args.controls.read_text())
    scenes = []
    for scene in controls["scenes"]:
        all_nodes = sorted(
            {
                int(evidence["mapped_time"])
                for family in scene["candidate_windows"]
                for evidence in family["evidence"]
            }
        )
        selected, details = choose_nodes(
            scene["candidate_windows"], args.max_nodes_per_scene
        )
        scenes.append(
            {
                "scene_seed": int(scene["scene_seed"]),
                "r0": int(scene["r0"]),
                "successful_rs": scene["successful_rs"],
                "replan_nodes": selected,
                "node_count": len(selected),
                "all_mapped_nodes_before_cap": all_nodes,
                "cap_applied": len(all_nodes) > args.max_nodes_per_scene,
                "node_details": details,
            }
        )

    payload = {
        "purpose": "Concrete per-scene replan nodes derived from selected candidate window families.",
        "source_controls": str(args.controls),
        "max_nodes_per_scene": args.max_nodes_per_scene,
        "selection_rule": (
            "Deduplicate mapped_time; preserve one representative per selected window family; "
            "then rank by distinct successful-r support at the exact node, lower state distance, "
            "window rank, and earlier time."
        ),
        "scenes": scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["scene_seed", "r0", "successful_rs", "replan_nodes", "node_count", "cap_applied"],
            )
            writer.writeheader()
            for scene in scenes:
                writer.writerow(
                    {
                        "scene_seed": scene["scene_seed"],
                        "r0": scene["r0"],
                        "successful_rs": ",".join(map(str, scene["successful_rs"])),
                        "replan_nodes": ",".join(map(str, scene["replan_nodes"])),
                        "node_count": scene["node_count"],
                        "cap_applied": scene["cap_applied"],
                    }
                )
    print(f"Wrote {len(scenes)} scenes: {args.output}")


if __name__ == "__main__":
    main()
