"""Map successful-only replans onto a failed r0 trajectory.

The compact pi0.5 traces do not contain measured robot state.  This script
therefore uses the 12 arm-joint targets of executed actions (grippers omitted)
as an explicit state proxy.  A short proxy-state segment immediately before a
successful replan is matched to the most similar segment in the failed r0
trajectory before candidate windows are merged and ranked.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_hr(directory: str) -> tuple[int, int]:
    horizon, execute = directory.split("_r", maxsplit=1)
    return int(horizon.removeprefix("H")), int(execute)


def load_success_map(path: Path) -> tuple[list[int], dict[int, list[int]]]:
    payload = json.loads(path.read_text())
    if "scene_selection" in payload:
        rows = payload["scene_selection"]
        return (
            [int(row["scene_seed"]) for row in rows],
            {int(row["scene_seed"]): [int(r) for r in row["successful_rs"]] for row in rows},
        )
    if "scenes" in payload:
        rows = payload["scenes"]
        return (
            [int(row["scene_seed"]) for row in rows],
            {int(row["scene_seed"]): [int(r) for r in row["successful_rs"]] for row in rows},
        )
    mapping = payload.get("successful_rs_by_seed", payload)
    if not isinstance(mapping, dict):
        raise ValueError("Unsupported success-map format")
    seeds = [int(seed) for seed in mapping]
    return seeds, {int(seed): [int(r) for r in rs] for seed, rs in mapping.items()}


def load_trace(path: Path) -> dict[str, Any]:
    trace = json.loads(path.read_text())
    chunks = {int(chunk["inference_call"]): chunk for chunk in trace["chunks"]}

    # Chunk start time equals the cumulative number of actions consumed from
    # preceding chunks. This remains correct for early terminal episodes.
    boundaries = [0]
    action_clock = 0
    for chunk in trace["chunks"][1:]:
        action_clock += int(chunk["previous_chunk_cursor"])
        boundaries.append(action_clock)

    states = []
    for action in trace["actions"]:
        if not int(action.get("effective_action_executed", 1)):
            continue
        call = int(action["inference_call"])
        cursor = int(action["chunk_action_index"])
        vector = chunks[call]["chunk_actions"][cursor]
        if len(vector) < 13:
            raise ValueError(f"{path}: expected a 14-D action, got {len(vector)}")
        # [left arm 0:6, left gripper 6, right arm 7:13, right gripper 13]
        states.append([*vector[0:6], *vector[7:13]])

    effective_actions = int(trace["episode_metrics"]["effective_policy_actions"])
    if len(states) != effective_actions:
        raise ValueError(
            f"{path}: reconstructed {len(states)} states for {effective_actions} effective actions"
        )
    return {
        "path": str(path),
        "scene_seed": int(trace["scene_seed"]),
        "success": bool(trace["episode_metrics"]["episode_success"]),
        "actions": effective_actions,
        "replan_times": boundaries,
        "states": np.asarray(states, dtype=np.float64),
    }


def trace_path(trace_root: Path, seed: int, execute_steps: int, horizon: int | None) -> Path:
    # Older grids used scene_<seed>.json; current evaluators retain an
    # occurrence suffix so deliberately repeated seeds are not overwritten.
    matches = sorted(
        {
            *trace_root.glob(f"H*_r{execute_steps}/scene_{seed}.json"),
            *trace_root.glob(f"H*_r{execute_steps}/scene_{seed}_episode_*.json"),
        }
    )
    if horizon is not None:
        matches = [path for path in matches if parse_hr(path.parent.name)[0] == horizon]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one trace for seed={seed}, r={execute_steps}, H={horizon}; got {matches}"
        )
    return matches[0]


def segment_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Mean per-action L2 across the two six-joint arms."""
    delta = left - right
    per_arm = np.stack(
        [np.linalg.norm(delta[:, 0:6], axis=1), np.linalg.norm(delta[:, 6:12], axis=1)],
        axis=1,
    )
    return float(np.mean(per_arm))


def map_state(
    success_states: np.ndarray,
    failure_states: np.ndarray,
    success_time: int,
    lookback: int,
    failure_end_margin: int,
) -> tuple[int, float, int] | None:
    used = min(lookback, success_time)
    if used <= 0 or used > len(success_states):
        return None
    query = success_states[success_time - used : success_time]
    latest_center = len(failure_states) - failure_end_margin
    if latest_center < used:
        return None
    best: tuple[int, float, int] | None = None
    for center in range(used, latest_center + 1):
        distance = segment_distance(query, failure_states[center - used : center])
        if best is None or distance < best[1]:
            best = (center, distance, used)
    return best


def min_boundary_distance(time: int, boundaries: list[int]) -> int | None:
    return min((abs(time - other) for other in boundaries), default=None)


def merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: (row["mapped_window"][0], row["mapped_window"][1])):
        start, end = candidate["mapped_window"]
        if not merged or start > merged[-1]["window"][1] + 1:
            merged.append({"window": [start, end], "evidence": [candidate]})
        else:
            merged[-1]["window"][1] = max(merged[-1]["window"][1], end)
            merged[-1]["evidence"].append(candidate)
    for family in merged:
        support_rs = sorted({row["success_r"] for row in family["evidence"]})
        # One r gets one vote even if it contributes multiple boundaries.
        best_by_r = {
            r: min(
                (row for row in family["evidence"] if row["success_r"] == r),
                key=lambda row: (row["state_distance"], row["mapped_time"]),
            )
            for r in support_rs
        }
        family["supporting_success_rs"] = support_rs
        family["distinct_r_support"] = len(support_rs)
        family["best_evidence_by_r"] = [best_by_r[r] for r in support_rs]
        family["mean_best_state_distance"] = float(
            np.mean([best_by_r[r]["state_distance"] for r in support_rs])
        )
        family["representative_time"] = int(
            round(np.median([best_by_r[r]["mapped_time"] for r in support_rs]))
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0", type=int, required=True)
    parser.add_argument("--scene-success-map", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--state-lookback", type=int, default=5)
    parser.add_argument(
        "--max-state-distance",
        type=float,
        default=0.35,
        help="Maximum mean two-arm joint-target L2 for the stage-similarity gate.",
    )
    parser.add_argument("--max-windows-per-scene", type=int, default=3)
    args = parser.parse_args()

    seeds, successful_rs = load_success_map(args.scene_success_map)
    scene_outputs = []
    for seed in seeds:
        failure = load_trace(trace_path(args.trace_root, seed, args.r0, args.horizon))
        if failure["success"]:
            raise ValueError(f"seed={seed}: r0={args.r0} trace is not a failure")
        failure_replans = failure["replan_times"]
        accepted = []
        rejected = []
        source_traces = {str(args.r0): failure["path"]}
        for success_r in sorted(set(successful_rs[seed])):
            if success_r == args.r0:
                continue
            success = load_trace(trace_path(args.trace_root, seed, success_r, args.horizon))
            source_traces[str(success_r)] = success["path"]
            if not success["success"]:
                raise ValueError(f"seed={seed}: declared successful r={success_r} is not successful")
            for success_time in success["replan_times"]:
                if success_time == 0:
                    continue
                raw_gap = min_boundary_distance(success_time, failure_replans)
                if raw_gap is not None and raw_gap <= args.delta:
                    rejected.append(
                        {
                            "success_r": success_r,
                            "success_replan_time": success_time,
                            "reason": "not_added_relative_to_r0",
                            "nearest_r0_replan_distance": raw_gap,
                        }
                    )
                    continue
                mapped = map_state(
                    success["states"],
                    failure["states"],
                    success_time,
                    args.state_lookback,
                    args.window_radius,
                )
                if mapped is None:
                    rejected.append(
                        {
                            "success_r": success_r,
                            "success_replan_time": success_time,
                            "reason": "insufficient_state_history",
                        }
                    )
                    continue
                mapped_time, distance, used = mapped
                mapped_gap = min_boundary_distance(mapped_time, failure_replans)
                record = {
                    "success_r": success_r,
                    "success_replan_time": success_time,
                    "raw_window": [
                        max(0, success_time - args.window_radius),
                        success_time + args.window_radius,
                    ],
                    "mapped_time": mapped_time,
                    "mapped_window": [
                        max(0, mapped_time - args.window_radius),
                        min(failure["actions"] - 1, mapped_time + args.window_radius),
                    ],
                    "state_distance": distance,
                    "state_lookback_actions": used,
                    "nearest_r0_replan_distance_after_mapping": mapped_gap,
                }
                if distance > args.max_state_distance:
                    rejected.append(record | {"reason": "state_stage_distance_too_high"})
                elif mapped_gap is not None and mapped_gap <= args.delta:
                    rejected.append(record | {"reason": "r0_already_replans_near_mapped_time"})
                elif mapped_time + args.window_radius >= failure["actions"]:
                    rejected.append(record | {"reason": "not_before_failure_end"})
                else:
                    accepted.append(record)

        families = merge_candidates(accepted)
        families.sort(
            key=lambda family: (
                -family["distinct_r_support"],
                family["mean_best_state_distance"],
                family["representative_time"],
            )
        )
        selected = families[: args.max_windows_per_scene]
        for rank, family in enumerate(selected, start=1):
            family["rank"] = rank
        scene_outputs.append(
            {
                "scene_seed": seed,
                "r0": args.r0,
                "failure_actions": failure["actions"],
                "r0_replan_times": failure_replans,
                "successful_rs": sorted(set(successful_rs[seed])),
                "candidate_windows": selected,
                "unselected_window_families": families[args.max_windows_per_scene :],
                "rejected_candidates": rejected,
                "source_traces": source_traces,
            }
        )

    output = {
        "purpose": "Per-scene failed-r0 versus successful-r candidate replan-window controls.",
        "r0": args.r0,
        "trace_root": str(args.trace_root),
        "scene_success_map": str(args.scene_success_map),
        "configuration": {
            "horizon": args.horizon,
            "time_tolerance_actions": args.delta,
            "window_radius_actions": args.window_radius,
            "state_proxy": "executed policy joint targets: left arm 6D + right arm 6D; grippers omitted",
            "state_lookback_actions": args.state_lookback,
            "state_distance": "mean over lookback of per-arm 6D L2, then mean over arms",
            "max_state_distance": args.max_state_distance,
            "distinct_r_voting": True,
            "max_windows_per_scene": args.max_windows_per_scene,
        },
        "scenes": scene_outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(scene_outputs)} scene controls: {args.output}")
    print(
        "selected windows:",
        sum(len(scene["candidate_windows"]) for scene in scene_outputs),
        "scenes with none:",
        sum(not scene["candidate_windows"] for scene in scene_outputs),
    )


if __name__ == "__main__":
    main()
