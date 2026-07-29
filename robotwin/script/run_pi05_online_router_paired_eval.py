"""Run a candidate-gated online replan-router evaluation.

The evaluation is deliberately conservative: it uses only held-out test
scenes, queries the frozen router only at candidate clocks fixed before this
run, enforces a configured replan budget/cooldown, and pairs every episode with a
deterministic no-router control.  A router query extracts frozen pi0.5
features but does not sample an action chunk or advance the VLA RNG stream.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

import numpy as np


WORKSPACE = Path("/home/ubuntu/Workspace")
PROJECT_ROOT = WORKSPACE / "Event-triggered-replanning-for-VLA-control"
ROBOTWIN_ROOT = WORKSPACE / "RoboTwin"
OPENPI_ROOT = WORKSPACE / "openpi"
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT / "temp/outputs/replan_router_minimal_validation"
)


def write_immutable(path: Path, payload) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(
                f"Refusing to overwrite incompatible file: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def observation_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path) as data:
        digest.update(
            np.ascontiguousarray(data["state"], dtype=np.float32).tobytes()
        )
        for key in (
            "head_camera_rgb",
            "left_camera_rgb",
            "right_camera_rgb",
        ):
            digest.update(np.ascontiguousarray(data[key]).tobytes())
    return digest.hexdigest()


def locate_existing(path_text: str, source_root: Path) -> Path:
    path = Path(path_text)
    candidates = (
        [path]
        if path.is_absolute()
        else [
            ROBOTWIN_ROOT / path,
            PROJECT_ROOT / path,
            source_root / path,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve {path_text!r}; checked {candidates}"
    )


def source_scene_metadata(feature_manifest: dict) -> tuple[dict, dict]:
    """Recover full candidate lists and plaintext prompts without label use."""
    nodes_by_seed: dict[int, list[int]] = {}
    prompts_by_seed: dict[int, str] = {}
    for source in feature_manifest["dataset"]["sources"]:
        source_root = Path(source["root"])
        plan_path = locate_existing(
            str(source_root / "experiment_plan.json"), source_root
        )
        plan = json.loads(plan_path.read_text())
        for control in plan["controls"]:
            seed = int(control["scene_seed"])
            nodes = sorted({int(x) for x in control["record_after_actions"]})
            if seed in nodes_by_seed and nodes_by_seed[seed] != nodes:
                raise ValueError(f"Conflicting candidate nodes for seed {seed}")
            nodes_by_seed[seed] = nodes
            case_dir = locate_existing(control["case_dir"], source_root)
            resolved = json.loads(
                (case_dir / "resolved_episode_manifest.json").read_text()
            )
            scene_seeds = [int(x) for x in resolved["scene_seeds"]]
            index = scene_seeds.index(seed)
            prompt = str(resolved["episode_instructions"][index])
            if seed in prompts_by_seed and prompts_by_seed[seed] != prompt:
                raise ValueError(f"Conflicting plaintext prompts for seed {seed}")
            prompts_by_seed[seed] = prompt
    return nodes_by_seed, prompts_by_seed


def read_single_trace(case_dir: Path) -> dict:
    paths = list((case_dir / "traces").glob("scene_*_episode_*.json"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one trace in {case_dir}, found {len(paths)}"
        )
    return json.loads(paths[0].read_text())


def target_sequence(trace: dict) -> list:
    chunks = {
        int(chunk["inference_call"]): chunk["chunk_actions"]
        for chunk in trace["chunks"]
    }
    return [
        chunks[int(row["inference_call"])][int(row["chunk_action_index"])]
        for row in trace["actions"]
    ]


def node_observation_path(case: dict, node: int) -> Path:
    return (
        Path(case["case_dir"])
        / "observations"
        / (
            f"episode_{int(case['occurrence_token']):03d}_"
            f"action_{node:03d}_after.npz"
        )
    )


def prepare_plan(args: argparse.Namespace) -> dict:
    if not args.output_dir.is_absolute():
        raise ValueError("--output-dir must be absolute")
    feature_manifest = json.loads(args.feature_manifest.read_text())
    if feature_manifest["feature_type"] != "vision_encoder":
        raise ValueError("The minimum online experiment expects vision_encoder")
    external_scene_source = args.scene_manifest is not None
    if external_scene_source:
        if args.resolved_scene_manifest is None:
            raise ValueError(
                "--scene-manifest requires --resolved-scene-manifest"
            )
        if not args.fixed_candidate_nodes and args.router_query_interval is None:
            raise ValueError(
                "--scene-manifest requires --fixed-candidate-nodes or "
                "--router-query-interval"
            )
        if args.fixed_candidate_nodes and args.router_query_interval is not None:
            raise ValueError(
                "Use only one of --fixed-candidate-nodes and "
                "--router-query-interval"
            )
        seed_manifest = json.loads(args.scene_manifest.read_text())
        resolved_manifest = json.loads(
            args.resolved_scene_manifest.read_text()
        )
        for payload in (seed_manifest, resolved_manifest):
            if (
                payload["task_name"] != args.task_name
                or payload["task_config"] != args.task_config
            ):
                raise ValueError("External scene manifest task mismatch")
        test_seeds = [int(seed) for seed in seed_manifest["scene_seeds"]]
        resolved_seeds = [
            int(seed) for seed in resolved_manifest["scene_seeds"]
        ]
        if test_seeds != resolved_seeds or len(test_seeds) != len(
            set(test_seeds)
        ):
            raise ValueError(
                "External seed/resolved manifests are not uniquely aligned"
            )
        prompts = resolved_manifest["episode_instructions"]
        if len(prompts) != len(test_seeds) or not all(
            isinstance(prompt, str) and prompt for prompt in prompts
        ):
            raise ValueError("External prompts must be resolved plaintext")
        prompts_by_seed = dict(zip(test_seeds, prompts, strict=True))
        if args.router_query_interval is not None:
            if args.router_query_interval < 1:
                raise ValueError("--router-query-interval must be positive")
            fixed_nodes = [
                node
                for node in range(
                    args.router_query_interval,
                    args.router_query_max_action + 1,
                    args.router_query_interval,
                )
                if node % args.r0 != 0
            ]
        else:
            fixed_nodes = sorted(
                {int(node) for node in args.fixed_candidate_nodes}
            )
        nodes_by_seed = {seed: fixed_nodes for seed in test_seeds}
        if args.scene_seed is not None:
            raise ValueError(
                "--scene-seed cannot be combined with --scene-manifest"
            )
    else:
        with np.load(args.feature_archive) as data:
            test_seeds = sorted(
                {
                    int(seed)
                    for seed, split in zip(
                        data["scene_seed"], data["split"], strict=True
                    )
                    if int(split) == 2
                }
            )
        if args.scene_seed is not None:
            if args.scene_seed not in test_seeds:
                raise ValueError(
                    f"Smoke seed {args.scene_seed} is not in held-out test scenes"
                )
            test_seeds = [args.scene_seed]
        nodes_by_seed, prompts_by_seed = source_scene_metadata(
            feature_manifest
        )
    missing = set(test_seeds) - set(nodes_by_seed)
    if missing:
        raise ValueError(f"Missing pre-frozen candidate nodes: {sorted(missing)}")
    for seed in test_seeds:
        nodes = nodes_by_seed[seed]
        if not nodes:
            raise ValueError(f"Empty candidate list for scene {seed}")
        if any(node <= 0 or node % args.r0 == 0 for node in nodes):
            raise ValueError(
                f"Scene {seed} has invalid/natural-boundary nodes: {nodes}"
            )

    router_nodes = {
        "purpose": "Candidate gate for held-out online router evaluation",
        "selection": (
            "Full candidate clocks fixed by the source counterfactual plans; "
            "labels and offline router predictions were not used to remove nodes."
        ),
        "scenes": [
            {
                "scene_seed": seed,
                "candidate_nodes": nodes_by_seed[seed],
            }
            for seed in test_seeds
        ],
    }
    router_nodes_path = args.output_dir / "router_nodes_manifest.json"
    write_immutable(router_nodes_path, router_nodes)

    controls = []
    routers = []
    occurrence = 0
    for arm, destination in (("control", controls), ("router", routers)):
        for seed in test_seeds:
            for repeat in range(args.repeats):
                case_dir = (
                    args.output_dir
                    / arm
                    / f"scene_{seed}"
                    / f"repeat_{repeat:02d}"
                )
                case = {
                    "arm": arm,
                    "scene_seed": seed,
                    "repeat": repeat,
                    "candidate_nodes": nodes_by_seed[seed],
                    "occurrence_token": occurrence,
                    "case_dir": str(case_dir),
                    "prompt_plaintext": prompts_by_seed[seed],
                    "prompt_sha256": hashlib.sha256(
                        prompts_by_seed[seed].encode("utf-8")
                    ).hexdigest(),
                }
                destination.append(case)
                occurrence += 1
                write_immutable(
                    case_dir / "seed_manifest.json",
                    {
                        "task_name": args.task_name,
                        "task_config": args.task_config,
                        # RoboTwin maps policy seed k to the expert-valid scene
                        # search base 100000 * (k + 1).
                        "base_seed": 100000 * (args.seed + 1),
                        "purpose": f"Online router paired {arm}",
                        "scene_seeds": [seed],
                    },
                )
                write_immutable(
                    case_dir / "resolved_episode_manifest.json",
                    {
                        "task_name": args.task_name,
                        "task_config": args.task_config,
                        "source_seed_manifest": str(
                            case_dir / "seed_manifest.json"
                        ),
                        "scene_seeds": [seed],
                        "episode_instructions": [prompts_by_seed[seed]],
                    },
                )
    return {
        "purpose": "Held-out candidate-gated online replan-router paired eval",
        "scope_limit": (
            "This is not an every-step deployment: the router is queried only "
            f"at pre-frozen candidate clocks and may trigger at most "
            f"{args.router_max_replans} times."
        ),
        "task_name": args.task_name,
        "task_config": args.task_config,
        "checkpoint_dir": str(args.checkpoint_dir),
        "server_config": args.server_config,
        "horizon": args.horizon,
        "r0": args.r0,
        "lambda": args.router_lambda,
        "max_replans": args.router_max_replans,
        "min_replan_interval": args.router_min_replan_interval,
        "router_query_interval": args.router_query_interval,
        "router_query_max_action": (
            args.router_query_max_action
            if args.router_query_interval is not None
            else None
        ),
        "absolute_r0_cadence": True,
        "deterministic_torch": True,
        "inference_seed": args.seed,
        "heldout_test_scenes": test_seeds,
        "external_scene_source": external_scene_source,
        "source_scene_manifest": (
            str(args.scene_manifest.resolve())
            if args.scene_manifest is not None
            else None
        ),
        "source_scene_manifest_sha256": (
            sha256_file(args.scene_manifest)
            if args.scene_manifest is not None
            else None
        ),
        "source_resolved_scene_manifest": (
            str(args.resolved_scene_manifest.resolve())
            if args.resolved_scene_manifest is not None
            else None
        ),
        "source_resolved_scene_manifest_sha256": (
            sha256_file(args.resolved_scene_manifest)
            if args.resolved_scene_manifest is not None
            else None
        ),
        "feature_archive": str(args.feature_archive.resolve()),
        "feature_archive_sha256": sha256_file(args.feature_archive),
        "feature_manifest": str(args.feature_manifest.resolve()),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "router_checkpoint": str(args.router_checkpoint.resolve()),
        "router_checkpoint_sha256": sha256_file(args.router_checkpoint),
        "router_nodes_manifest": str(router_nodes_path),
        "repeats": args.repeats,
        "controls": controls,
        "routers": routers,
    }


def eval_command(
    args: argparse.Namespace, plan: dict, case: dict, port: int
) -> list[str]:
    case_dir = Path(case["case_dir"])
    return [
        str(ROBOTWIN_ROOT / ".venv/bin/python"),
        "-m",
        "script.eval_policy_wandb",
        "--wandb-project",
        "robotwin-pi05-online-router",
        "--wandb-group",
        f"{args.task_name}-heldout-router-r{args.r0}",
        "--wandb-run-name",
        (
            f"{args.task_name}-{case['arm']}-seed{case['scene_seed']}-"
            f"repeat{case['repeat']:02d}"
        ),
        "--eval-num",
        "1",
        "--action-horizon",
        str(args.horizon),
        "--episode-id-offset",
        str(case["occurrence_token"]),
        "--metrics-json",
        str(case_dir / "metrics.json"),
        "--seed-manifest",
        str(case_dir / "seed_manifest.json"),
        "--resolved-manifest",
        str(case_dir / "resolved_episode_manifest.json"),
        "--trace-dir",
        str(case_dir / "traces"),
        "--observation-record-dir",
        str(case_dir / "observations"),
        "--observation-record-actions",
        *[str(node) for node in case["candidate_nodes"]],
        "--config",
        "policy/pi05/deploy_policy.yml",
        "--overrides",
        "--task_name",
        args.task_name,
        "--task_config",
        args.task_config,
        "--policy_name",
        "pi05_remote",
        "--model_name",
        args.model_name,
        "--checkpoint_id",
        "15000",
        "--ckpt_setting",
        f"online_router_{case['arm']}_seed{case['scene_seed']}",
        "--server_host",
        "127.0.0.1",
        "--server_port",
        str(port),
        "--seed",
        str(args.seed),
        "--instruction_type",
        "unseen",
        "--pi0_step",
        str(args.r0),
        "--pi05_intervention",
        "none",
        "--pi05_force_replan_before_actions",
        "[]",
        "--pi05_absolute_r0_cadence",
        "True",
        "--pi05_router_enabled",
        repr(case["arm"] == "router"),
        "--pi05_router_nodes_manifest",
        plan["router_nodes_manifest"],
        "--pi05_router_lambda",
        str(args.router_lambda),
        "--pi05_router_max_replans",
        str(args.router_max_replans),
        "--pi05_router_min_replan_interval",
        str(args.router_min_replan_interval),
    ]


def analyze_pair(plan: dict, control: dict, router: dict) -> dict:
    seed = int(control["scene_seed"])
    repeat = int(control["repeat"])
    result = {"scene_seed": seed, "repeat": repeat}
    try:
        control_trace = read_single_trace(Path(control["case_dir"]))
        router_trace = read_single_trace(Path(router["case_dir"]))
        control_targets = target_sequence(control_trace)
        router_targets = target_sequence(router_trace)
        queries = [
            query
            for chunk in router_trace["chunks"]
            for query in chunk.get("router_queries", [])
        ]
        triggers = [query for query in queries if query["trigger"]]
        trigger_nodes = [
            int(query["completed_actions"]) for query in triggers
        ]
        first_trigger_node = trigger_nodes[0] if trigger_nodes else None
        query_nodes = [int(query["completed_actions"]) for query in queries]
        candidate_nodes = [int(x) for x in router["candidate_nodes"]]
        node_checks = {}
        for node in query_nodes:
            control_path = node_observation_path(control, node)
            router_path = node_observation_path(router, node)
            control_fp = (
                observation_fingerprint(control_path)
                if control_path.exists()
                else None
            )
            router_fp = (
                observation_fingerprint(router_path)
                if router_path.exists()
                else None
            )
            recorded_query = next(
                item for item in queries
                if int(item["completed_actions"]) == node
            )
            node_checks[str(node)] = {
                "control_fingerprint": control_fp,
                "router_fingerprint": router_fp,
                "query_fingerprint": recorded_query[
                    "observation_fingerprint"
                ],
                "router_matches_query": (
                    router_fp is not None
                    and router_fp
                    == recorded_query["observation_fingerprint"]
                ),
                "control_matches_router": (
                    control_fp is not None and control_fp == router_fp
                ),
            }

        prior_trigger = None
        cooldown_queries_valid = True
        trigger_metadata_valid = True
        triggers_seen = 0
        for query in queries:
            node = int(query["completed_actions"])
            if (
                prior_trigger is not None
                and node - prior_trigger
                < int(plan.get("min_replan_interval", 0))
            ):
                cooldown_queries_valid = False
            if int(query.get("replans_before_query", triggers_seen)) != triggers_seen:
                trigger_metadata_valid = False
            expected_last = prior_trigger
            if query.get("last_replan_action") != expected_last:
                trigger_metadata_valid = False
            if query["trigger"]:
                prior_trigger = node
                triggers_seen += 1

        pre_divergence_nodes = [
            node
            for node in query_nodes
            if first_trigger_node is None or node <= first_trigger_node
        ]
        checks = {
            "scene_seed_identical": (
                int(control_trace["scene_seed"])
                == int(router_trace["scene_seed"])
                == seed
            ),
            "prompt_plaintext_identical": (
                control_trace["instruction"]
                == router_trace["instruction"]
                == control["prompt_plaintext"]
            ),
            "prompt_fingerprint_identical": (
                all(
                    chunk["prompt_fingerprint"] == control["prompt_sha256"]
                    for trace in (control_trace, router_trace)
                    for chunk in trace["chunks"]
                )
            ),
            "queries_only_at_candidates": (
                len(query_nodes) == len(set(query_nodes))
                and set(query_nodes).issubset(candidate_nodes)
            ),
            "all_query_observations_recorded": all(
                item["router_matches_query"] for item in node_checks.values()
            ),
            "pre_divergence_query_observations_match_control": all(
                node_checks[str(node)]["control_matches_router"]
                for node in pre_divergence_nodes
            ),
            "at_most_configured_triggers": (
                len(triggers) <= int(plan["max_replans"])
            ),
            "trigger_spacing_respected": all(
                later - earlier
                >= int(plan.get("min_replan_interval", 0))
                for earlier, later in zip(
                    trigger_nodes, trigger_nodes[1:]
                )
            ),
            "cooldown_queries_suppressed": cooldown_queries_valid,
            "trigger_metadata_consistent": trigger_metadata_valid,
            "lambda_exact": all(
                float(query["lambda"]) == float(plan["lambda"])
                for query in queries
            ),
        }
        replacement_audits = []
        if triggers:
            checks["both_have_trigger_prefix"] = (
                len(control_targets) >= first_trigger_node
                and len(router_targets) >= first_trigger_node
            )
            checks["prefix_targets_identical"] = (
                checks["both_have_trigger_prefix"]
                and sha256_json(control_targets[:first_trigger_node])
                == sha256_json(router_targets[:first_trigger_node])
            )
            for trigger in triggers:
                trigger_node = int(trigger["completed_actions"])
                marker_indices = [
                    index
                    for index, chunk in enumerate(router_trace["chunks"])
                    if trigger_node + 1
                    in chunk.get("router_trigger_before_actions", [])
                ]
                marker_index = (
                    marker_indices[0] if len(marker_indices) == 1 else None
                )
                replacement = (
                    router_trace["chunks"][marker_index + 1]
                    if marker_index is not None
                    and marker_index + 1 < len(router_trace["chunks"])
                    else None
                )
                replacement_fp = (
                    replacement.get("observation_fingerprint")
                    if replacement is not None
                    else None
                )
                expected_limit = (
                    int(plan["r0"]) - trigger_node % int(plan["r0"])
                )
                replacement_audits.append(
                    {
                        "trigger_node": trigger_node,
                        "marker_exact": len(marker_indices) == 1,
                        "replacement_observation_fingerprint": replacement_fp,
                        "replacement_uses_trigger_observation": (
                            replacement_fp
                            == trigger["observation_fingerprint"]
                        ),
                        "replacement_absolute_cadence": (
                            replacement is not None
                            and replacement.get("absolute_r0_cadence")
                            is True
                            and int(replacement.get("executed_r", -1))
                            == expected_limit
                            and int(
                                replacement.get(
                                    "absolute_r0_next_boundary", -1
                                )
                            )
                            == trigger_node + expected_limit
                        ),
                    }
                )
            checks["all_trigger_markers_exact"] = all(
                audit["marker_exact"] for audit in replacement_audits
            )
            checks["all_replacements_use_trigger_observation"] = all(
                audit["replacement_uses_trigger_observation"]
                for audit in replacement_audits
            )
            checks["all_replacements_follow_absolute_cadence"] = all(
                audit["replacement_absolute_cadence"]
                for audit in replacement_audits
            )
        else:
            checks["full_target_sequence_identical_without_trigger"] = (
                sha256_json(control_targets) == sha256_json(router_targets)
            )
            checks["outcome_identical_without_trigger"] = (
                bool(control_trace["episode_metrics"]["episode_success"])
                == bool(router_trace["episode_metrics"]["episode_success"])
            )

        control_success = bool(
            control_trace["episode_metrics"]["episode_success"]
        )
        router_success = bool(
            router_trace["episode_metrics"]["episode_success"]
        )
        transition = (
            "rescue"
            if not control_success and router_success
            else "harm"
            if control_success and not router_success
            else "both_success"
            if control_success and router_success
            else "both_failure"
        )
        result.update(
            {
                "status": "complete",
                "valid_pair": bool(queries) and all(checks.values()),
                "checks": checks,
                "candidate_nodes": candidate_nodes,
                "router_queries": queries,
                "query_node_observations": node_checks,
                "trigger_node": first_trigger_node,
                "trigger_nodes": trigger_nodes,
                "replacement_audits": replacement_audits,
                "control_success": control_success,
                "router_success": router_success,
                "transition": transition,
                "control_target_sha256": sha256_json(control_targets),
                "router_target_sha256": sha256_json(router_targets),
                "control_case_dir": control["case_dir"],
                "router_case_dir": router["case_dir"],
            }
        )
    except Exception as exc:
        result.update({"status": "analysis_error", "error": str(exc)})
    return result


def final_summary(plan: dict) -> dict:
    router_by_key = {
        (int(case["scene_seed"]), int(case["repeat"])): case
        for case in plan["routers"]
    }
    pairs = [
        analyze_pair(
            plan,
            control,
            router_by_key[
                (int(control["scene_seed"]), int(control["repeat"]))
            ],
        )
        for control in plan["controls"]
    ]
    complete = [item for item in pairs if item["status"] == "complete"]
    valid = [item for item in complete if item["valid_pair"]]
    reproducibility = {}
    for seed in plan["heldout_test_scenes"]:
        rows = [
            item
            for item in valid
            if item["scene_seed"] == seed
        ]
        signatures = [
            {
                "control_success": row["control_success"],
                "router_success": row["router_success"],
                "trigger_node": row["trigger_node"],
                "trigger_nodes": row.get(
                    "trigger_nodes",
                    [row["trigger_node"]]
                    if row["trigger_node"] is not None
                    else [],
                ),
                "queries": [
                    {
                        "node": query["completed_actions"],
                        "p_keep": query["p_keep"],
                        "p_replan": query["p_replan"],
                        "trigger": query["trigger"],
                        "fp": query["observation_fingerprint"],
                    }
                    for query in row["router_queries"]
                ],
                "control_targets": row["control_target_sha256"],
                "router_targets": row["router_target_sha256"],
            }
            for row in rows
        ]
        reproducibility[str(seed)] = {
            "valid_repeats": len(rows),
            "exactly_reproducible": (
                len(signatures) == int(plan["repeats"])
                and all(
                    signature == signatures[0]
                    for signature in signatures[1:]
                )
            ),
        }
    return {
        "purpose": plan["purpose"],
        "scope_limit": plan["scope_limit"],
        "design": {
            "scenes": len(plan["heldout_test_scenes"]),
            "repeats": plan["repeats"],
            "planned_episodes": len(plan["controls"]) + len(plan["routers"]),
            "lambda": plan["lambda"],
            "r0": plan["r0"],
            "horizon": plan["horizon"],
            "max_replans": plan["max_replans"],
            "min_replan_interval": plan.get("min_replan_interval", 0),
            "absolute_r0_cadence": plan["absolute_r0_cadence"],
            "shared_prefix_gate": (
                "same seed, plaintext prompt, prompt hash, node observation, "
                "and attempted action targets through the trigger"
            ),
        },
        "completed_pairs": len(complete),
        "valid_pairs": len(valid),
        "invalid_pairs": len(complete) - len(valid),
        "triggered_pairs": sum(
            item["trigger_node"] is not None for item in valid
        ),
        "total_router_replans": sum(
            len(item.get("trigger_nodes", [])) for item in valid
        ),
        "control_successes": sum(
            item["control_success"] for item in valid
        ),
        "router_successes": sum(item["router_success"] for item in valid),
        "rescues": sum(item["transition"] == "rescue" for item in valid),
        "harms": sum(item["transition"] == "harm" for item in valid),
        "both_success": sum(
            item["transition"] == "both_success" for item in valid
        ),
        "both_failure": sum(
            item["transition"] == "both_failure" for item in valid
        ),
        "reproducibility": reproducibility,
        "all_repeats_exactly_reproducible": all(
            item["exactly_reproducible"]
            for item in reproducibility.values()
        ),
        "pairs": pairs,
    }


def wait_for_port(port: int, server: subprocess.Popen, timeout: int = 240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"OpenPI server exited with code {server.returncode}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(2)
    raise TimeoutError(f"OpenPI server did not open port {port}")


def require_port_available(port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is unavailable") from exc


def stop_process_group(process: subprocess.Popen):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def progress(plan: dict) -> dict:
    cases = plan["controls"] + plan["routers"]
    complete = [
        case
        for case in cases
        if (Path(case["case_dir"]) / "metrics.json").exists()
    ]
    return {
        "planned_episodes": len(cases),
        "completed_episodes": len(complete),
        "pending_episodes": len(cases) - len(complete),
        "completed_controls": sum(
            case["arm"] == "control" for case in complete
        ),
        "completed_routers": sum(
            case["arm"] == "router" for case in complete
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-archive",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "features_vision_encoder.npz",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=(
            DEFAULT_ARTIFACT_ROOT / "features_vision_encoder_manifest.json"
        ),
    )
    parser.add_argument(
        "--router-checkpoint",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "router_vision_encoder_vision.pt",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/home/ubuntu/Model/pi0.5_robotwin2"),
    )
    parser.add_argument(
        "--server-config", default="pi05_robotwin2_multitask_pytorch"
    )
    parser.add_argument("--model-name", default="pi0.5_robotwin2")
    parser.add_argument("--task-name", default="move_playingcard_away")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--r0", type=int, default=25)
    parser.add_argument("--router-lambda", type=float, default=0.05)
    parser.add_argument("--router-max-replans", type=int, default=1)
    parser.add_argument(
        "--router-min-replan-interval",
        type=int,
        default=0,
        help="Minimum completed-action distance between router triggers.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--scene-seed", type=int)
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        help="External immutable scene set, e.g. a new unseen-scene manifest.",
    )
    parser.add_argument(
        "--resolved-scene-manifest",
        type=Path,
        help="Plaintext prompt manifest aligned with --scene-manifest.",
    )
    parser.add_argument(
        "--fixed-candidate-nodes",
        type=int,
        nargs="+",
        help="Leakage-free common router query clocks for external scenes.",
    )
    parser.add_argument(
        "--router-query-interval",
        type=int,
        help=(
            "Query every N completed actions on external scenes, excluding "
            "natural r0 boundaries."
        ),
    )
    parser.add_argument(
        "--router-query-max-action",
        type=int,
        default=398,
        help="Last completed-action clock eligible for interval queries.",
    )
    parser.add_argument("--server-gpu", default="0")
    parser.add_argument("--client-gpu", default="1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--parallel-groups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default="disabled",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.horizon != 50 or args.r0 != 25:
        parser.error("This validated minimum experiment requires H=50, r0=25")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.parallel_groups < 1:
        parser.error("--parallel-groups must be positive")
    if not 0 < args.router_lambda < 1:
        parser.error("--router-lambda must be between zero and one")
    if args.router_max_replans < 1:
        parser.error("--router-max-replans must be positive")
    if args.router_min_replan_interval < 0:
        parser.error("--router-min-replan-interval must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = prepare_plan(args)
    write_immutable(args.output_dir / "experiment_plan.json", plan)
    write_json(args.output_dir / "progress.json", progress(plan))
    print(
        f"Planned {len(plan['controls']) + len(plan['routers'])} episodes: "
        f"{len(plan['controls'])} control + {len(plan['routers'])} router",
        flush=True,
    )
    if args.prepare_only:
        return

    cases = plan["controls"] + plan["routers"]
    pending = [
        case
        for case in cases
        if not (Path(case["case_dir"]) / "metrics.json").exists()
    ]
    if not pending:
        summary_path = args.output_dir / "online_router_summary.json"
        write_json(summary_path, final_summary(plan))
        print(summary_path)
        return

    server_gpus = [item.strip() for item in args.server_gpu.split(",")]
    client_gpus = [item.strip() for item in args.client_gpu.split(",")]
    if len(server_gpus) not in (1, args.parallel_groups):
        parser.error(
            "--server-gpu must be one GPU or a comma-separated GPU per group"
        )
    if len(client_gpus) not in (1, args.parallel_groups):
        parser.error(
            "--client-gpu must be one GPU or a comma-separated GPU per group"
        )
    if len(server_gpus) == 1:
        server_gpus *= args.parallel_groups
    if len(client_gpus) == 1:
        client_gpus *= args.parallel_groups
    group_count = min(args.parallel_groups, len(pending))
    ports = [args.port + index for index in range(group_count)]
    for port in ports:
        require_port_available(port)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    servers = []
    server_logs = []
    try:
        for group, port in enumerate(ports):
            server_env = os.environ | {
                "CUDA_VISIBLE_DEVICES": server_gpus[group],
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": str(args.seed),
            }
            server_command = [
                str(OPENPI_ROOT / ".venv/bin/python"),
                str(
                    PROJECT_ROOT
                    / "openpi/scripts/serve_robotwin_router_policy.py"
                ),
                "--config",
                args.server_config,
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--router-checkpoint",
                str(args.router_checkpoint),
                "--action-horizon",
                str(args.horizon),
                "--port",
                str(port),
                "--inference-seed",
                str(args.seed),
                "--deterministic-torch",
            ]
            server_log = (
                logs
                / f"openpi_router_server_group_{group:02d}_port_{port}.log"
            ).open("a")
            server_logs.append(server_log)
            servers.append(
                subprocess.Popen(
                    server_command,
                    cwd=OPENPI_ROOT,
                    env=server_env,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        for port, server in zip(ports, servers, strict=True):
            wait_for_port(port, server)

        curobo = ROBOTWIN_ROOT / "envs_invent/curobo/src"
        indexed_pending = list(enumerate(pending, start=1))
        queues = [
            indexed_pending[group::group_count]
            for group in range(group_count)
        ]
        progress_lock = threading.Lock()
        stop_event = threading.Event()

        def run_group(group: int, port: int, queue: list) -> None:
            eval_env = os.environ | {
                "CUDA_VISIBLE_DEVICES": client_gpus[group],
                "WANDB_MODE": args.wandb_mode,
                "PYTHONHASHSEED": str(args.seed),
            }
            eval_env["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (
                    str(PROJECT_ROOT / "robotwin/policy"),
                    str(curobo),
                    eval_env.get("PYTHONPATH", ""),
                )
                if part
            )
            for index, case in queue:
                if stop_event.is_set():
                    return
                case_dir = Path(case["case_dir"])
                print(
                    f"[{index}/{len(pending)} group={group} port={port}] "
                    f"{case['arm']} seed={case['scene_seed']} "
                    f"repeat={case['repeat']}",
                    flush=True,
                )
                with (case_dir / "eval.log").open("w") as eval_log:
                    result = subprocess.run(
                        eval_command(args, plan, case, port),
                        cwd=ROBOTWIN_ROOT,
                        env=eval_env,
                        stdout=eval_log,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode != 0:
                    stop_event.set()
                    raise RuntimeError(
                        f"Evaluation failed; see {case_dir / 'eval.log'}"
                    )
                with progress_lock:
                    write_json(
                        args.output_dir / "progress.json", progress(plan)
                    )

        with ThreadPoolExecutor(max_workers=group_count) as executor:
            futures = [
                executor.submit(run_group, group, port, queue)
                for group, (port, queue) in enumerate(
                    zip(ports, queues, strict=True)
                )
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    stop_event.set()
                    for other in futures:
                        other.cancel()
                    raise
    finally:
        for server in servers:
            stop_process_group(server)
        for server_log in server_logs:
            server_log.close()

    summary_path = args.output_dir / "online_router_summary.json"
    summary = final_summary(plan)
    write_json(summary_path, summary)
    print(summary_path)
    if summary["valid_pairs"] != len(plan["controls"]):
        raise RuntimeError("One or more online router pairs failed validity")
    if plan["repeats"] > 1 and not summary[
        "all_repeats_exactly_reproducible"
    ]:
        raise RuntimeError("Repeated online router smoke was not reproducible")


if __name__ == "__main__":
    main()
