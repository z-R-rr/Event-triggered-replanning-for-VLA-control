"""Correct the V0 threshold sweep with reset-safe replays and paired controls.

This run preserves clean treatments from the original sweep, adds one current
control for every recorded r0=25 failure, and reruns only treatments whose
first query observation did not match the canonical V0 prefix. Every new case
has a unique reset-only episode ID. Final validity requires exact pre-trigger
observation and action-target agreement with its current control.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

from run_pi05_online_router_paired_eval import (
    node_observation_path,
    observation_fingerprint,
    read_single_trace,
    require_port_available,
    sha256_file,
    sha256_json,
    stop_process_group,
    target_sequence,
    wait_for_port,
    write_immutable,
    write_json,
)
from run_pi05_online_v0_threshold_sweep import (
    DEFAULT_SOURCE_METRICS,
    DEFAULT_SOURCE_RESOLVED,
    DEFAULT_THRESHOLDS,
    FEATURE_ROOT,
    OPENPI_ROOT,
    PROJECT_ROOT,
    ROBOTWIN_ROOT,
    source_failure_scenes,
    threshold_name,
)


DEFAULT_ORIGINAL_ROOT = (
    PROJECT_ROOT
    / "temp/outputs/move_playingcard_away_v0_threshold_sweep_r25_failures"
)


def trace_queries(trace: dict) -> list[dict]:
    return [
        query
        for chunk in trace["chunks"]
        for query in chunk.get("router_queries", [])
    ]


def load_original_results(original_root: Path) -> tuple[dict, dict]:
    summary_path = original_root / "v0_threshold_sweep_summary.json"
    plan_path = original_root / "experiment_plan.json"
    if not summary_path.exists() or not plan_path.exists():
        raise FileNotFoundError("Original threshold sweep is incomplete")
    return json.loads(summary_path.read_text()), json.loads(plan_path.read_text())


def detect_contaminated_cases(
    original_summary: dict,
    seeds: list[int],
) -> dict[str, list[int]]:
    summaries = original_summary["threshold_summaries"]
    canonical_names = ("lambda_0p05", "lambda_0p1", "lambda_0p2")
    by_name = {
        name: {
            int(result["scene_seed"]): result
            for result in summaries[name]["results"]
        }
        for name in summaries
    }
    contaminated = {name: [] for name in summaries}
    for seed in seeds:
        canonical_fingerprints = {
            by_name[name][seed]["queries"][0]["observation_fingerprint"]
            for name in canonical_names
        }
        if len(canonical_fingerprints) != 1:
            raise ValueError(
                f"Canonical thresholds disagree at first query for seed {seed}"
            )
        canonical = next(iter(canonical_fingerprints))
        for name in summaries:
            result = by_name[name][seed]
            if result["status"] != "complete" or not result["queries"]:
                contaminated[name].append(seed)
            elif result["queries"][0]["observation_fingerprint"] != canonical:
                contaminated[name].append(seed)
    expected = {
        "lambda_0p05": 0,
        "lambda_0p1": 0,
        "lambda_0p2": 0,
        "lambda_0p3": 3,
        "lambda_0p5": 25,
    }
    observed = {name: len(values) for name, values in contaminated.items()}
    if observed != expected:
        raise ValueError(
            f"Unexpected contamination pattern: {observed}, expected {expected}"
        )
    return contaminated


def write_case_manifests(
    args: argparse.Namespace,
    case: dict,
) -> None:
    case_dir = Path(case["case_dir"])
    role = case["role"]
    write_immutable(
        case_dir / "seed_manifest.json",
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "base_seed": 100000,
            "purpose": f"Corrected V0 threshold sweep {role}",
            "scene_seeds": [case["scene_seed"]],
        },
    )
    write_immutable(
        case_dir / "resolved_episode_manifest.json",
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "source_seed_manifest": str(case_dir / "seed_manifest.json"),
            "scene_seeds": [case["scene_seed"]],
            "episode_instructions": [case["prompt_plaintext"]],
        },
    )


def prepare_plan(args: argparse.Namespace) -> dict:
    if not args.output_dir.is_absolute():
        raise ValueError("--output-dir must be absolute")
    seeds, prompts = source_failure_scenes(args)
    original_summary, original_plan = load_original_results(args.original_root)
    if (
        original_plan["source_population"]["failure_seeds"] != seeds
        or original_plan["thresholds"] != list(DEFAULT_THRESHOLDS)
    ):
        raise ValueError("Original sweep population or thresholds changed")
    candidate_nodes = [int(value) for value in original_plan["candidate_nodes"]]
    contaminated = detect_contaminated_cases(original_summary, seeds)

    inputs_dir = args.output_dir / "inputs"
    seed_manifest = inputs_dir / "r25_failure_seed_manifest.json"
    resolved_manifest = inputs_dir / "r25_failure_resolved_manifest.json"
    router_nodes_manifest = inputs_dir / "router_nodes_manifest.json"
    write_immutable(
        seed_manifest,
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "base_seed": 100000,
            "purpose": "Recorded deterministic H50/r25 failures",
            "source_metrics": str(args.source_metrics.resolve()),
            "source_metrics_sha256": sha256_file(args.source_metrics),
            "selection": "episode_success == 0",
            "scene_seeds": seeds,
        },
    )
    write_immutable(
        resolved_manifest,
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "source_seed_manifest": str(seed_manifest),
            "source_resolved_manifest": str(args.source_resolved_manifest.resolve()),
            "source_resolved_manifest_sha256": sha256_file(
                args.source_resolved_manifest
            ),
            "scene_seeds": seeds,
            "episode_instructions": prompts,
        },
    )
    write_immutable(
        router_nodes_manifest,
        {
            "purpose": "Common corrected V0 threshold-sweep query clocks",
            "query_interval": 5,
            "natural_r0_boundaries_excluded": True,
            "scenes": [
                {"scene_seed": seed, "candidate_nodes": candidate_nodes}
                for seed in seeds
            ],
        },
    )

    checkpoint = args.v0_checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing V0 checkpoint: {checkpoint}")
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) != 4:
        raise ValueError("--gpus must contain exactly four IDs")
    workers = [
        {"worker": index, "gpu": gpu, "port": args.port_base + index}
        for index, gpu in enumerate(gpu_ids)
    ]

    prompt_by_seed = dict(zip(seeds, prompts, strict=True))
    source_index = {seed: index for index, seed in enumerate(seeds)}
    controls = []
    cases_to_run = []
    for seed in seeds:
        case = {
            "role": "control",
            "router_enabled": False,
            "threshold": None,
            "threshold_name": "control",
            "scene_seed": seed,
            "source_failure_index": source_index[seed],
            "occurrence_token": 9_000_000 + source_index[seed],
            "candidate_nodes": candidate_nodes,
            "case_dir": str(args.output_dir / "control" / f"scene_{seed}"),
            "prompt_plaintext": prompt_by_seed[seed],
            "prompt_sha256": hashlib.sha256(
                prompt_by_seed[seed].encode("utf-8")
            ).hexdigest(),
            "result_source": "corrected_rerun",
        }
        controls.append(case)
        cases_to_run.append(case)
        write_case_manifests(args, case)

    corrected_by_threshold = {
        threshold_name(value): [] for value in DEFAULT_THRESHOLDS
    }
    for threshold_index, threshold in enumerate(DEFAULT_THRESHOLDS):
        name = threshold_name(threshold)
        for seed in contaminated[name]:
            case = {
                "role": "treatment",
                "router_enabled": True,
                "threshold": threshold,
                "threshold_name": name,
                "scene_seed": seed,
                "source_failure_index": source_index[seed],
                "occurrence_token": (
                    (threshold_index + 1) * 100_000 + source_index[seed]
                ),
                "candidate_nodes": candidate_nodes,
                "case_dir": str(args.output_dir / name / f"scene_{seed}"),
                "prompt_plaintext": prompt_by_seed[seed],
                "prompt_sha256": hashlib.sha256(
                    prompt_by_seed[seed].encode("utf-8")
                ).hexdigest(),
                "result_source": "corrected_rerun",
            }
            corrected_by_threshold[name].append(case)
            cases_to_run.append(case)
            write_case_manifests(args, case)

    effective_cases = {}
    original_cases = original_plan["cases_by_threshold"]
    for threshold in DEFAULT_THRESHOLDS:
        name = threshold_name(threshold)
        original_by_seed = {
            int(case["scene_seed"]): case for case in original_cases[name]
        }
        corrected_by_seed = {
            int(case["scene_seed"]): case
            for case in corrected_by_threshold[name]
        }
        cases = []
        for seed in seeds:
            if seed in corrected_by_seed:
                cases.append(corrected_by_seed[seed])
            else:
                original = dict(original_by_seed[seed])
                original.update(
                    {
                        "role": "treatment",
                        "result_source": "original_prefix_clean",
                    }
                )
                cases.append(original)
        effective_cases[name] = cases

    return {
        "purpose": "Reset-safe correction of the V0 threshold sweep",
        "source_population": {
            "base_seed": 100000,
            "failure_count": len(seeds),
            "failure_seeds": seeds,
            "source_metrics": str(args.source_metrics.resolve()),
            "source_metrics_sha256": sha256_file(args.source_metrics),
        },
        "original_sweep": str(args.original_root.resolve()),
        "original_summary_sha256": sha256_file(
            args.original_root / "v0_threshold_sweep_summary.json"
        ),
        "contaminated_cases": contaminated,
        "correction_episode_count": len(cases_to_run),
        "horizon": 50,
        "r0": 25,
        "query_interval": 5,
        "candidate_nodes": candidate_nodes,
        "max_replans": 1,
        "min_replan_interval": 0,
        "absolute_r0_cadence": True,
        "thresholds": list(DEFAULT_THRESHOLDS),
        "router_feature_config": "V0",
        "router_checkpoint": str(checkpoint),
        "router_checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "server_config": args.server_config,
        "inference_seed": args.seed,
        "deterministic_torch": True,
        "workers": workers,
        "router_nodes_manifest": str(router_nodes_manifest),
        "controls": controls,
        "corrected_by_threshold": corrected_by_threshold,
        "effective_cases_by_threshold": effective_cases,
        "cases_to_run": cases_to_run,
    }


def eval_command(
    args: argparse.Namespace,
    plan: dict,
    case: dict,
    port: int,
) -> list[str]:
    case_dir = Path(case["case_dir"])
    arm = case["threshold_name"]
    command = [
        str(ROBOTWIN_ROOT / ".venv/bin/python"),
        "-m",
        "script.eval_policy_wandb",
        "--wandb-project",
        "robotwin-pi05-online-v0-threshold-correction",
        "--wandb-group",
        f"{args.task_name}-r25-failures-v0-corrected",
        "--wandb-run-name",
        f"{args.task_name}-{arm}-seed{case['scene_seed']}",
        "--eval-num",
        "1",
        "--action-horizon",
        "50",
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
        f"online_v0_threshold_corrected_{arm}_seed{case['scene_seed']}",
        "--server_host",
        "127.0.0.1",
        "--server_port",
        str(port),
        "--seed",
        str(args.seed),
        "--instruction_type",
        "unseen",
        "--pi0_step",
        "25",
        "--pi05_intervention",
        "none",
        "--pi05_force_replan_before_actions",
        "[]",
        "--pi05_absolute_r0_cadence",
        "True",
        "--pi05_router_enabled",
        repr(bool(case["router_enabled"])),
        "--pi05_router_nodes_manifest",
        plan["router_nodes_manifest"],
        "--pi05_router_lambda",
        str(case["threshold"] if case["threshold"] is not None else 0.05),
        "--pi05_router_max_replans",
        "1",
        "--pi05_router_min_replan_interval",
        "0",
    ]
    return command


def progress(plan: dict) -> dict:
    cases = plan["cases_to_run"]
    by_role = {}
    for role in ["control", "lambda_0p3", "lambda_0p5"]:
        selected = [
            case
            for case in cases
            if (
                case["role"] == "control"
                if role == "control"
                else case["threshold_name"] == role
            )
        ]
        by_role[role] = {
            "planned": len(selected),
            "completed": sum(
                (Path(case["case_dir"]) / "metrics.json").exists()
                for case in selected
            ),
        }
    return {
        "planned_episodes": len(cases),
        "completed_episodes": sum(
            (Path(case["case_dir"]) / "metrics.json").exists()
            for case in cases
        ),
        "by_role": by_role,
    }


def audit_pair(
    plan: dict,
    control: dict,
    treatment: dict,
) -> dict:
    seed = int(control["scene_seed"])
    result = {
        "scene_seed": seed,
        "threshold": treatment["threshold"],
        "treatment_source": treatment["result_source"],
    }
    try:
        control_trace = read_single_trace(Path(control["case_dir"]))
        treatment_trace = read_single_trace(Path(treatment["case_dir"]))
        control_targets = target_sequence(control_trace)
        treatment_targets = target_sequence(treatment_trace)
        queries = trace_queries(treatment_trace)
        triggers = [query for query in queries if query["trigger"]]
        trigger_node = (
            int(triggers[0]["completed_actions"]) if triggers else None
        )
        query_nodes = [int(query["completed_actions"]) for query in queries]
        pre_divergence_nodes = [
            node
            for node in query_nodes
            if trigger_node is None or node <= trigger_node
        ]
        node_checks = {}
        for query in queries:
            node = int(query["completed_actions"])
            control_path = node_observation_path(control, node)
            # Prefix-clean treatments reused from the original sweep did not
            # archive observation files or an occurrence token. Their query
            # fingerprint is still sufficient for the required comparison
            # against the newly recorded contemporaneous control.
            treatment_path = (
                node_observation_path(treatment, node)
                if "occurrence_token" in treatment
                else None
            )
            control_fp = (
                observation_fingerprint(control_path)
                if control_path.exists()
                else None
            )
            treatment_fp = (
                observation_fingerprint(treatment_path)
                if treatment_path is not None and treatment_path.exists()
                else None
            )
            query_fp = query["observation_fingerprint"]
            node_checks[str(node)] = {
                "control_fingerprint": control_fp,
                "treatment_fingerprint": treatment_fp,
                "query_fingerprint": query_fp,
                "control_matches_query": control_fp == query_fp,
                "recorded_treatment_matches_query": (
                    treatment_fp == query_fp if treatment_fp is not None else None
                ),
            }
        checks = {
            "scene_seed_identical": (
                int(control_trace["scene_seed"])
                == int(treatment_trace["scene_seed"])
                == seed
            ),
            "prompt_identical": (
                control_trace["instruction"]
                == treatment_trace["instruction"]
                == control["prompt_plaintext"]
            ),
            "queries_present": bool(queries),
            "queries_only_at_candidates": (
                len(query_nodes) == len(set(query_nodes))
                and set(query_nodes).issubset(set(plan["candidate_nodes"]))
            ),
            "lambda_exact": all(
                abs(float(query["lambda"]) - float(treatment["threshold"]))
                < 1e-12
                for query in queries
            ),
            "at_most_one_trigger": len(triggers) <= 1,
            "pre_trigger_observations_match_control": all(
                node_checks[str(node)]["control_matches_query"]
                for node in pre_divergence_nodes
            ),
            "recorded_treatment_observations_match_queries": all(
                check["recorded_treatment_matches_query"] in (None, True)
                for check in node_checks.values()
            ),
        }
        if trigger_node is not None:
            checks["pre_trigger_targets_identical"] = (
                len(control_targets) >= trigger_node
                and len(treatment_targets) >= trigger_node
                and sha256_json(control_targets[:trigger_node])
                == sha256_json(treatment_targets[:trigger_node])
            )
            marker_indices = [
                index
                for index, chunk in enumerate(treatment_trace["chunks"])
                if trigger_node + 1
                in chunk.get("router_trigger_before_actions", [])
            ]
            replacement = (
                treatment_trace["chunks"][marker_indices[0] + 1]
                if len(marker_indices) == 1
                and marker_indices[0] + 1 < len(treatment_trace["chunks"])
                else None
            )
            checks["trigger_marker_exact"] = len(marker_indices) == 1
            checks["replacement_uses_trigger_observation"] = (
                replacement is not None
                and replacement.get("observation_fingerprint")
                == triggers[0]["observation_fingerprint"]
            )
            expected_limit = 25 - trigger_node % 25
            checks["replacement_absolute_cadence"] = (
                replacement is not None
                and replacement.get("absolute_r0_cadence") is True
                and int(replacement.get("executed_r", -1)) == expected_limit
                and int(replacement.get("absolute_r0_next_boundary", -1))
                == trigger_node + expected_limit
            )
        else:
            checks["full_targets_identical_without_trigger"] = (
                sha256_json(control_targets) == sha256_json(treatment_targets)
            )
            checks["outcome_identical_without_trigger"] = (
                bool(control_trace["episode_metrics"]["episode_success"])
                == bool(treatment_trace["episode_metrics"]["episode_success"])
            )

        control_success = bool(control_trace["episode_metrics"]["episode_success"])
        treatment_success = bool(
            treatment_trace["episode_metrics"]["episode_success"]
        )
        transition = (
            "rescue"
            if not control_success and treatment_success
            else "harm"
            if control_success and not treatment_success
            else "both_success"
            if control_success and treatment_success
            else "both_failure"
        )
        result.update(
            {
                "status": "complete",
                "valid_pair": all(checks.values()),
                "checks": checks,
                "control_success": control_success,
                "treatment_success": treatment_success,
                "transition": transition,
                "triggered": bool(triggers),
                "trigger_node": trigger_node,
                "query_count": len(queries),
                "node_checks": node_checks,
                "control_case_dir": control["case_dir"],
                "treatment_case_dir": treatment["case_dir"],
            }
        )
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as error:
        result.update({"status": "incomplete", "error": str(error)})
    return result


def final_summary(plan: dict) -> dict:
    controls = {
        int(case["scene_seed"]): case for case in plan["controls"]
    }
    threshold_summaries = {}
    for name, cases in plan["effective_cases_by_threshold"].items():
        pairs = [
            audit_pair(plan, controls[int(case["scene_seed"])], case)
            for case in cases
        ]
        complete = [pair for pair in pairs if pair["status"] == "complete"]
        valid = [pair for pair in complete if pair["valid_pair"]]
        threshold_summaries[name] = {
            "threshold": cases[0]["threshold"],
            "planned_pairs": len(pairs),
            "complete_pairs": len(complete),
            "valid_pairs": len(valid),
            "triggered_pairs": sum(pair["triggered"] for pair in valid),
            "control_successes": sum(pair["control_success"] for pair in valid),
            "treatment_successes": sum(
                pair["treatment_success"] for pair in valid
            ),
            "rescues": sum(pair["transition"] == "rescue" for pair in valid),
            "harms": sum(pair["transition"] == "harm" for pair in valid),
            "both_success": sum(
                pair["transition"] == "both_success" for pair in valid
            ),
            "both_failure": sum(
                pair["transition"] == "both_failure" for pair in valid
            ),
            "pairs": pairs,
        }
    return {
        "purpose": plan["purpose"],
        "source_population": plan["source_population"],
        "design": {
            "feature_config": "V0",
            "thresholds": plan["thresholds"],
            "horizon": 50,
            "r0": 25,
            "query_interval": 5,
            "max_replans": 1,
            "new_episodes": plan["correction_episode_count"],
            "control_is_contemporaneous": True,
            "validity_gate": (
                "Exact pre-trigger observation fingerprints and action targets"
            ),
        },
        "contaminated_cases_replaced": plan["contaminated_cases"],
        "threshold_summaries": threshold_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument(
        "--source-resolved-manifest",
        type=Path,
        default=DEFAULT_SOURCE_RESOLVED,
    )
    parser.add_argument("--task-name", default="move_playingcard_away")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--r0", type=int, default=25)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--port-base", type=int, default=8400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-mode", default="disabled")
    parser.add_argument(
        "--v0-checkpoint",
        type=Path,
        default=FEATURE_ROOT / "checkpoints/V0.pt",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/home/ubuntu/Model/pi0.5_robotwin2"),
    )
    parser.add_argument("--server-config", default="pi05_robotwin2_multitask_pytorch")
    parser.add_argument("--model-name", default="pi0.5_robotwin2")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if args.horizon != 50 or args.r0 != 25:
        parser.error("The correction run is fixed to H=50 and r0=25")
    if args.port_base != 8400:
        parser.error("The correction run is fixed to ports 8400--8403")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = prepare_plan(args)
    write_immutable(args.output_dir / "experiment_plan.json", plan)
    write_json(args.output_dir / "progress.json", progress(plan))
    print(json.dumps(progress(plan), indent=2), flush=True)
    if args.prepare_only:
        return

    for worker in plan["workers"]:
        require_port_available(int(worker["port"]))
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    servers = {}
    server_logs = []
    stop_event = threading.Event()
    try:
        for worker in plan["workers"]:
            worker_id = int(worker["worker"])
            server_env = os.environ | {
                "CUDA_VISIBLE_DEVICES": str(worker["gpu"]),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": str(args.seed),
            }
            command = [
                str(OPENPI_ROOT / ".venv/bin/python"),
                str(PROJECT_ROOT / "openpi/scripts/serve_robotwin_router_policy.py"),
                "--config",
                args.server_config,
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--router-checkpoint",
                plan["router_checkpoint"],
                "--action-horizon",
                "50",
                "--port",
                str(worker["port"]),
                "--inference-seed",
                str(args.seed),
                "--deterministic-torch",
            ]
            log_stream = (
                logs_dir / f"openpi_worker_{worker_id}_port_{worker['port']}.log"
            ).open("a")
            server_logs.append(log_stream)
            servers[worker_id] = subprocess.Popen(
                command,
                cwd=OPENPI_ROOT,
                env=server_env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for worker in plan["workers"]:
            worker_id = int(worker["worker"])
            wait_for_port(int(worker["port"]), servers[worker_id])

        pending_queue: queue.Queue[dict] = queue.Queue()
        cases_by_seed = {}
        for case in plan["cases_to_run"]:
            cases_by_seed.setdefault(int(case["scene_seed"]), []).append(case)
        for seed in plan["source_population"]["failure_seeds"]:
            for case in sorted(
                cases_by_seed[seed],
                key=lambda value: (
                    value["role"] != "control",
                    value["threshold"] or 0.0,
                ),
            ):
                if not (Path(case["case_dir"]) / "metrics.json").exists():
                    pending_queue.put(case)

        curobo = ROBOTWIN_ROOT / "envs_invent/curobo/src"
        progress_lock = threading.Lock()

        def run_worker(worker: dict) -> None:
            worker_id = int(worker["worker"])
            eval_env = os.environ | {
                "CUDA_VISIBLE_DEVICES": str(worker["gpu"]),
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
            while not stop_event.is_set():
                try:
                    case = pending_queue.get_nowait()
                except queue.Empty:
                    return
                case_dir = Path(case["case_dir"])
                print(
                    f"[worker={worker_id} gpu={worker['gpu']} "
                    f"port={worker['port']}] {case['threshold_name']} "
                    f"seed={case['scene_seed']} episode_id="
                    f"{case['occurrence_token']}",
                    flush=True,
                )
                with (case_dir / "eval.log").open("w") as stream:
                    result = subprocess.run(
                        eval_command(args, plan, case, int(worker["port"])),
                        cwd=ROBOTWIN_ROOT,
                        env=eval_env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                pending_queue.task_done()
                if result.returncode != 0:
                    stop_event.set()
                    raise RuntimeError(
                        f"Worker {worker_id} failed; see {case_dir / 'eval.log'}"
                    )
                with progress_lock:
                    write_json(args.output_dir / "progress.json", progress(plan))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_worker, worker) for worker in plan["workers"]
            ]
            for future in as_completed(futures):
                future.result()
    finally:
        for server in servers.values():
            stop_process_group(server)
        for stream in server_logs:
            stream.close()

    summary = final_summary(plan)
    write_json(
        args.output_dir / "v0_threshold_sweep_corrected_summary.json",
        summary,
    )
    compact = {
        name: {
            key: value
            for key, value in threshold_summary.items()
            if key != "pairs"
        }
        for name, threshold_summary in summary["threshold_summaries"].items()
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
