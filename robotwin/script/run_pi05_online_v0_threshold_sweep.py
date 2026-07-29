"""Sweep the V0 online-router threshold on recorded deterministic r0 failures.

The source population is the failed subset of one immutable H=50/r0=25
100-scene evaluation beginning at scene seed 100000. Four identical frozen-V0
servers execute a shared queue; changing lambda never changes the checkpoint,
scene, prompt, query clocks, or control protocol.
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
    require_port_available,
    sha256_file,
    stop_process_group,
    wait_for_port,
    write_immutable,
    write_json,
)


WORKSPACE = Path("/home/ubuntu/Workspace")
PROJECT_ROOT = WORKSPACE / "Event-triggered-replanning-for-VLA-control"
ROBOTWIN_ROOT = WORKSPACE / "RoboTwin"
OPENPI_ROOT = WORKSPACE / "openpi"
FEATURE_ROOT = PROJECT_ROOT / "temp/outputs/replan_router_feature_ablation"
SOURCE_ROOT = (
    ROBOTWIN_ROOT
    / "eval_result/move_playingcard_away/pi05_remote"
    / "hr_eval100_pi05_robotwin2_deterministic_seed0_repl01"
)
DEFAULT_SOURCE_METRICS = SOURCE_ROOT / "trial_002_H50_r25/metrics.json"
DEFAULT_SOURCE_RESOLVED = SOURCE_ROOT / "resolved_episode_manifest.json"
DEFAULT_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)


def threshold_name(value: float) -> str:
    return f"lambda_{value:g}".replace(".", "p")


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("Thresholds must be unique")
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Thresholds must be strictly between zero and one")
    return thresholds


def source_failure_scenes(args: argparse.Namespace) -> tuple[list[int], list[str]]:
    metrics = json.loads(args.source_metrics.read_text())
    resolved = json.loads(args.source_resolved_manifest.read_text())
    if (
        metrics["action_horizon"] != args.horizon
        or metrics["pi0_step"] != args.r0
        or metrics["episodes"] != len(metrics["scene_seeds"])
        or len(metrics["episode_metrics"]) != len(metrics["scene_seeds"])
    ):
        raise ValueError("Source metrics do not match the fixed H/r0 protocol")
    if (
        resolved["task_name"] != args.task_name
        or resolved["task_config"] != args.task_config
        or metrics["scene_seeds"] != resolved["scene_seeds"]
        or len(resolved["episode_instructions"]) != len(resolved["scene_seeds"])
    ):
        raise ValueError("Source metrics and resolved manifest are not aligned")

    failed_seeds = []
    failed_prompts = []
    for seed, prompt, episode in zip(
        metrics["scene_seeds"],
        resolved["episode_instructions"],
        metrics["episode_metrics"],
        strict=True,
    ):
        if not bool(episode["episode_success"]):
            failed_seeds.append(int(seed))
            failed_prompts.append(str(prompt))
    if len(failed_seeds) != int(metrics["episodes"] - metrics["successes"]):
        raise AssertionError("Source failure count is inconsistent")
    if not failed_seeds or len(failed_seeds) != len(set(failed_seeds)):
        raise ValueError("Source failure scene set is empty or non-unique")
    return failed_seeds, failed_prompts


def prepare_plan(args: argparse.Namespace) -> dict:
    if not args.output_dir.is_absolute():
        raise ValueError("--output-dir must be absolute")
    thresholds = parse_thresholds(args.thresholds)
    seeds, prompts = source_failure_scenes(args)
    if args.scene_limit is not None:
        if not 1 <= args.scene_limit <= len(seeds):
            raise ValueError("--scene-limit is outside the failure scene set")
        seeds = seeds[: args.scene_limit]
        prompts = prompts[: args.scene_limit]

    candidate_nodes = [
        node
        for node in range(
            args.query_interval,
            args.query_max_action + 1,
            args.query_interval,
        )
        if node % args.r0 != 0
    ]
    if not candidate_nodes:
        raise ValueError("No eligible router query clocks")

    inputs_dir = args.output_dir / "inputs"
    failure_seed_manifest = inputs_dir / "r25_failure_seed_manifest.json"
    failure_resolved_manifest = inputs_dir / "r25_failure_resolved_manifest.json"
    router_nodes_manifest = inputs_dir / "router_nodes_manifest.json"
    write_immutable(
        failure_seed_manifest,
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "base_seed": 100000,
            "purpose": "Failures from the recorded deterministic H50/r25 run",
            "source_metrics": str(args.source_metrics.resolve()),
            "source_metrics_sha256": sha256_file(args.source_metrics),
            "selection": "episode_success == 0",
            "scene_seeds": seeds,
        },
    )
    write_immutable(
        failure_resolved_manifest,
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "source_seed_manifest": str(failure_seed_manifest),
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
            "purpose": "Common leakage-free V0 threshold-sweep query clocks",
            "query_interval": args.query_interval,
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
    workers = []
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) != 4:
        raise ValueError("--gpus must contain exactly four comma-separated IDs")
    for index, gpu in enumerate(gpu_ids):
        workers.append(
            {
                "worker": index,
                "gpu": gpu,
                "port": args.port_base + index,
            }
        )

    cases_by_threshold = {}
    for threshold_index, threshold in enumerate(thresholds):
        name = threshold_name(threshold)
        cases = []
        for source_index, (seed, prompt) in enumerate(
            zip(seeds, prompts, strict=True)
        ):
            case_dir = args.output_dir / name / f"scene_{seed}"
            case = {
                "threshold": threshold,
                "threshold_name": name,
                "scene_seed": seed,
                "source_failure_index": source_index,
                # Episode ID is reset-only metadata, not an entropy source.
                # It must nevertheless be unique across threshold replays so
                # a persistent server resets its primary inference index when
                # two copies of the same scene run consecutively.
                "occurrence_token": threshold_index * 100_000 + source_index,
                "candidate_nodes": candidate_nodes,
                "case_dir": str(case_dir),
                "prompt_plaintext": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            cases.append(case)
            write_immutable(
                case_dir / "seed_manifest.json",
                {
                    "task_name": args.task_name,
                    "task_config": args.task_config,
                    "base_seed": 100000,
                    "purpose": f"V0 online threshold sweep at lambda={threshold:g}",
                    "scene_seeds": [seed],
                },
            )
            write_immutable(
                case_dir / "resolved_episode_manifest.json",
                {
                    "task_name": args.task_name,
                    "task_config": args.task_config,
                    "source_seed_manifest": str(case_dir / "seed_manifest.json"),
                    "scene_seeds": [seed],
                    "episode_instructions": [prompt],
                },
            )
        cases_by_threshold[name] = cases

    return {
        "purpose": "Frozen-V0 threshold sweep on recorded r0=25 failures",
        "task_name": args.task_name,
        "task_config": args.task_config,
        "source_population": {
            "base_seed": 100000,
            "source_metrics": str(args.source_metrics.resolve()),
            "source_metrics_sha256": sha256_file(args.source_metrics),
            "selection": "episode_success == 0",
            "failure_count": len(seeds),
            "failure_seeds": seeds,
        },
        "horizon": args.horizon,
        "r0": args.r0,
        "query_interval": args.query_interval,
        "query_max_action": args.query_max_action,
        "candidate_nodes": candidate_nodes,
        "max_replans": args.router_max_replans,
        "min_replan_interval": 0,
        "absolute_r0_cadence": True,
        "thresholds": thresholds,
        "router_feature_config": "V0",
        "router_checkpoint": str(checkpoint),
        "router_checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "server_config": args.server_config,
        "deterministic_torch": True,
        "inference_seed": args.seed,
        "workers": workers,
        "failure_seed_manifest": str(failure_seed_manifest),
        "failure_resolved_manifest": str(failure_resolved_manifest),
        "router_nodes_manifest": str(router_nodes_manifest),
        "cases_by_threshold": cases_by_threshold,
    }


def eval_command(
    args: argparse.Namespace,
    plan: dict,
    case: dict,
    port: int,
) -> list[str]:
    case_dir = Path(case["case_dir"])
    threshold = float(case["threshold"])
    return [
        str(ROBOTWIN_ROOT / ".venv/bin/python"),
        "-m",
        "script.eval_policy_wandb",
        "--wandb-project",
        "robotwin-pi05-online-v0-threshold-sweep",
        "--wandb-group",
        f"{args.task_name}-r25-failures-v0",
        "--wandb-run-name",
        f"{args.task_name}-v0-lambda{threshold:g}-seed{case['scene_seed']}",
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
        f"online_v0_threshold_{threshold:g}_seed{case['scene_seed']}",
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
        "True",
        "--pi05_router_nodes_manifest",
        plan["router_nodes_manifest"],
        "--pi05_router_lambda",
        str(threshold),
        "--pi05_router_max_replans",
        str(args.router_max_replans),
        "--pi05_router_min_replan_interval",
        "0",
    ]


def progress(plan: dict) -> dict:
    cases_by_threshold = plan["cases_by_threshold"]
    return {
        "planned_episodes": sum(len(cases) for cases in cases_by_threshold.values()),
        "completed_episodes": sum(
            (Path(case["case_dir"]) / "metrics.json").exists()
            for cases in cases_by_threshold.values()
            for case in cases
        ),
        "by_threshold": {
            name: {
                "threshold": cases[0]["threshold"],
                "planned": len(cases),
                "completed": sum(
                    (Path(case["case_dir"]) / "metrics.json").exists()
                    for case in cases
                ),
            }
            for name, cases in cases_by_threshold.items()
        },
    }


def load_case_result(case: dict) -> dict:
    case_dir = Path(case["case_dir"])
    metrics_path = case_dir / "metrics.json"
    if not metrics_path.exists():
        return {
            "scene_seed": case["scene_seed"],
            "status": "missing",
        }
    metrics = json.loads(metrics_path.read_text())
    trace_paths = sorted((case_dir / "traces").glob("*.json"))
    if len(trace_paths) != 1:
        return {
            "scene_seed": case["scene_seed"],
            "status": "invalid_trace_count",
            "trace_count": len(trace_paths),
        }
    trace = json.loads(trace_paths[0].read_text())
    queries = [
        action["router_query"]
        for action in trace["actions"]
        if action.get("router_query") is not None
    ]
    trigger_queries = [query for query in queries if query["trigger"]]
    valid = (
        len(metrics["episode_metrics"]) == 1
        and int(trace["scene_seed"]) == int(case["scene_seed"])
        and all(
            abs(float(query["lambda"]) - float(case["threshold"])) < 1e-12
            for query in queries
        )
        and len(trigger_queries) <= 1
        and all(
            int(query["completed_actions"]) in set(case["candidate_nodes"])
            for query in queries
        )
    )
    return {
        "scene_seed": case["scene_seed"],
        "status": "complete" if valid else "invalid",
        "success": bool(metrics["episode_metrics"][0]["episode_success"]),
        "triggered": bool(trigger_queries),
        "trigger_node": (
            int(trigger_queries[0]["completed_actions"]) if trigger_queries else None
        ),
        "query_count": len(queries),
        "queries": queries,
        "metrics_path": str(metrics_path),
        "trace_path": str(trace_paths[0]),
    }


def final_summary(plan: dict) -> dict:
    summaries = {}
    for name, cases in plan["cases_by_threshold"].items():
        results = [load_case_result(case) for case in cases]
        valid = [result for result in results if result["status"] == "complete"]
        triggered = [result for result in valid if result["triggered"]]
        summaries[name] = {
            "threshold": cases[0]["threshold"],
            "planned": len(cases),
            "complete_valid": len(valid),
            "successes": sum(result["success"] for result in valid),
            "success_rate": (
                sum(result["success"] for result in valid) / len(valid)
                if valid
                else None
            ),
            "rescues_from_recorded_failure": sum(
                result["success"] for result in valid
            ),
            "triggered": len(triggered),
            "trigger_rate": len(triggered) / len(valid) if valid else None,
            "successes_among_triggered": sum(
                result["success"] for result in triggered
            ),
            "results": results,
        }
    return {
        "purpose": plan["purpose"],
        "source_population": plan["source_population"],
        "design": {
            "feature_config": plan["router_feature_config"],
            "thresholds": plan["thresholds"],
            "horizon": plan["horizon"],
            "r0": plan["r0"],
            "query_interval": plan["query_interval"],
            "max_replans": plan["max_replans"],
            "planned_episodes": sum(
                len(cases) for cases in plan["cases_by_threshold"].values()
            ),
        },
        "threshold_summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument(
        "--source-resolved-manifest",
        type=Path,
        default=DEFAULT_SOURCE_RESOLVED,
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--scene-limit", type=int)
    parser.add_argument("--task-name", default="move_playingcard_away")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--r0", type=int, default=25)
    parser.add_argument("--query-interval", type=int, default=5)
    parser.add_argument("--query-max-action", type=int, default=398)
    parser.add_argument("--router-max-replans", type=int, default=1)
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

    if (
        args.horizon != 50
        or args.r0 != 25
        or args.query_interval != 5
        or args.router_max_replans != 1
    ):
        parser.error(
            "This experiment is fixed to H=50, r0=25, query interval=5, "
            "max_replans=1"
        )
    if args.port_base != 8400 and args.scene_limit is None:
        parser.error("The full run is fixed to ports 8400--8403")

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
                str(args.horizon),
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
        # Interleave thresholds within each scene so partial progress remains
        # directly comparable if the run is interrupted.
        threshold_cases = list(plan["cases_by_threshold"].values())
        for scene_index in range(len(threshold_cases[0])):
            for cases in threshold_cases:
                case = cases[scene_index]
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
                    f"[worker={worker_id} gpu={worker['gpu']} port={worker['port']}] "
                    f"lambda={case['threshold']:g} seed={case['scene_seed']}",
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
    write_json(args.output_dir / "v0_threshold_sweep_summary.json", summary)
    print(json.dumps(summary["threshold_summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
