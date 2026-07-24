"""Run deterministic shared-control evaluation for all candidate replan nodes.

For each scene this launcher runs one unmodified r0 control and records the
post-action observation at every candidate node.  It then runs one treatment
per scene/node, forcing a replan immediately before one-based action ``t + 1``.
Every treatment is validated against its scene's shared control using:

* identical state + three RGB inputs after action t;
* identical policy target sequence for attempted actions 1..t;
* an observed forced-replan marker before action t+1; and
* a replacement chunk inferred from the recorded node observation.

The server is deterministic and every one-episode subprocess receives a
unique occurrence token.  The token resets server call indexing but is not
mixed into policy RNG entropy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np


WORKSPACE = Path("/home/ubuntu/Workspace")
ROBOTWIN_ROOT = WORKSPACE / "RoboTwin"
OPENPI_ROOT = WORKSPACE / "openpi"


def write_immutable(path: Path, payload) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite incompatible file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def wait_for_port(port: int, server: subprocess.Popen, timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"OpenPI server exited with code {server.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(2)
    raise TimeoutError(f"OpenPI server did not open port {port}")


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def observation_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path) as data:
        digest.update(np.ascontiguousarray(data["state"], dtype=np.float32).tobytes())
        for key in ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
            digest.update(np.ascontiguousarray(data[key]).tobytes())
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_single_trace(case_dir: Path) -> dict:
    paths = list((case_dir / "traces").glob("scene_*_episode_*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one trace in {case_dir}, found {len(paths)}")
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
        / f"episode_{int(case['occurrence_token']):03d}_action_{node:03d}_after.npz"
    )


def progress_report(plan: dict) -> dict:
    cases = plan["controls"] + plan["forced"]
    complete = [
        case for case in cases if (Path(case["case_dir"]) / "metrics.json").exists()
    ]
    return {
        "planned_trials": len(cases),
        "planned_controls": len(plan["controls"]),
        "planned_forced": len(plan["forced"]),
        "completed_trials": len(complete),
        "completed_controls": sum(case["arm"] == "control" for case in complete),
        "completed_forced": sum(case["arm"] == "forced" for case in complete),
        "pending_trials": len(cases) - len(complete),
    }


def final_summary(plan: dict) -> dict:
    control_by_seed = {
        int(case["scene_seed"]): case for case in plan["controls"]
    }
    pairs = []
    for forced_case in plan["forced"]:
        seed = int(forced_case["scene_seed"])
        node = int(forced_case["replan_after_actions"])
        control_case = control_by_seed[seed]
        control_metrics_path = Path(control_case["case_dir"]) / "metrics.json"
        forced_metrics_path = Path(forced_case["case_dir"]) / "metrics.json"
        if not control_metrics_path.exists() or not forced_metrics_path.exists():
            pairs.append(
                {
                    "scene_seed": seed,
                    "replan_after_actions": node,
                    "status": "pending",
                }
            )
            continue
        try:
            control_trace = read_single_trace(Path(control_case["case_dir"]))
            forced_trace = read_single_trace(Path(forced_case["case_dir"]))
            control_targets = target_sequence(control_trace)
            forced_targets = target_sequence(forced_trace)
            control_node_path = node_observation_path(control_case, node)
            forced_node_path = node_observation_path(forced_case, node)
            control_node_fp = (
                observation_fingerprint(control_node_path)
                if control_node_path.exists()
                else None
            )
            forced_node_fp = (
                observation_fingerprint(forced_node_path)
                if forced_node_path.exists()
                else None
            )
            markers = [
                int(action)
                for chunk in forced_trace["chunks"]
                for action in chunk.get("forced_replan_before_actions", [])
            ]
            marker_chunk_index = next(
                (
                    index
                    for index, chunk in enumerate(forced_trace["chunks"])
                    if node + 1 in chunk.get("forced_replan_before_actions", [])
                ),
                None,
            )
            replacement_fp = None
            if (
                marker_chunk_index is not None
                and marker_chunk_index + 1 < len(forced_trace["chunks"])
            ):
                replacement_fp = forced_trace["chunks"][
                    marker_chunk_index + 1
                ].get("observation_fingerprint")
            checks = {
                "control_reached_node": control_node_fp is not None,
                "forced_reached_node": forced_node_fp is not None,
                "node_observation_identical": (
                    control_node_fp is not None
                    and control_node_fp == forced_node_fp
                ),
                "both_have_full_prefix": (
                    len(control_targets) >= node and len(forced_targets) >= node
                ),
                "prefix_targets_identical": (
                    len(control_targets) >= node
                    and len(forced_targets) >= node
                    and sha256_json(control_targets[:node])
                    == sha256_json(forced_targets[:node])
                ),
                "expected_forced_marker": markers == [node + 1],
                "replacement_uses_node_observation": (
                    forced_node_fp is not None and replacement_fp == forced_node_fp
                ),
                "instruction_identical": (
                    control_trace["instruction"] == forced_trace["instruction"]
                ),
            }
            control_success = bool(
                control_trace["episode_metrics"]["episode_success"]
            )
            forced_success = bool(forced_trace["episode_metrics"]["episode_success"])
            if not control_success and forced_success:
                transition = "rescue"
            elif control_success and not forced_success:
                transition = "harm"
            elif control_success and forced_success:
                transition = "both_success"
            else:
                transition = "both_failure"
            pairs.append(
                {
                    "scene_seed": seed,
                    "replan_after_actions": node,
                    "force_before_one_based_action": node + 1,
                    "status": "complete",
                    "valid_pair": all(checks.values()),
                    "checks": checks,
                    "control_success": control_success,
                    "forced_success": forced_success,
                    "transition": transition,
                    "control_node_observation_fingerprint": control_node_fp,
                    "forced_node_observation_fingerprint": forced_node_fp,
                    "control_prefix_target_sha256": (
                        sha256_json(control_targets[:node])
                        if len(control_targets) >= node
                        else None
                    ),
                    "forced_prefix_target_sha256": (
                        sha256_json(forced_targets[:node])
                        if len(forced_targets) >= node
                        else None
                    ),
                    "replacement_observation_fingerprint": replacement_fp,
                    "forced_replan_before_actions": markers,
                    "control_case_dir": control_case["case_dir"],
                    "forced_case_dir": forced_case["case_dir"],
                }
            )
        except Exception as exc:
            pairs.append(
                {
                    "scene_seed": seed,
                    "replan_after_actions": node,
                    "status": "analysis_error",
                    "error": str(exc),
                }
            )

    complete = [pair for pair in pairs if pair["status"] == "complete"]
    valid = [pair for pair in complete if pair["valid_pair"]]
    eligible = [pair for pair in valid if not pair["control_success"]]
    by_scene = []
    for seed in sorted(control_by_seed):
        scene_pairs = [pair for pair in valid if pair["scene_seed"] == seed]
        scene_eligible = [
            pair for pair in scene_pairs if not pair["control_success"]
        ]
        by_scene.append(
            {
                "scene_seed": seed,
                "valid_pairs": len(scene_pairs),
                "eligible_control_failure_pairs": len(scene_eligible),
                "rescued_nodes": [
                    pair["replan_after_actions"]
                    for pair in scene_eligible
                    if pair["forced_success"]
                ],
                "harmful_nodes": [
                    pair["replan_after_actions"]
                    for pair in scene_pairs
                    if pair["control_success"] and not pair["forced_success"]
                ],
            }
        )
    return {
        "purpose": "Deterministic causal audit of one forced replan at candidate nodes.",
        "important_limit": (
            "Candidate scenes/nodes were selected exploratorily from prior traces; "
            "node trials within a scene are dependent and are not independent samples."
        ),
        "design": {
            "shared_controls": len(plan["controls"]),
            "forced_node_trials": len(plan["forced"]),
            "total_trials": len(plan["controls"]) + len(plan["forced"]),
            "server_deterministic_torch": True,
            "control": "one unmodified r0 trajectory per scene",
            "treatment": "discard old chunk tail after action t and infer before action t+1",
            "validity_gate": (
                "same instruction, bitwise node observation, identical attempted "
                "action targets 1..t, forced marker, and replacement observation"
            ),
        },
        "completed_pairs": len(complete),
        "valid_pairs": len(valid),
        "invalid_pairs": len(complete) - len(valid),
        "eligible_control_failure_pairs": len(eligible),
        "rescues": sum(pair["transition"] == "rescue" for pair in valid),
        "harms": sum(pair["transition"] == "harm" for pair in valid),
        "both_success": sum(pair["transition"] == "both_success" for pair in valid),
        "both_failure": sum(pair["transition"] == "both_failure" for pair in valid),
        "rescue_rate_among_eligible_pairs": (
            sum(pair["transition"] == "rescue" for pair in eligible) / len(eligible)
            if eligible
            else None
        ),
        "by_scene": by_scene,
        "pairs": pairs,
    }


def prepare_plan(args: argparse.Namespace) -> dict:
    nodes = json.loads(args.nodes.read_text())
    seeds = json.loads(args.seed_manifest.read_text())
    resolved = json.loads(args.resolved_manifest.read_text())
    if seeds["task_name"] != args.task_name or seeds["task_config"] != args.task_config:
        raise ValueError("Seed manifest task/config mismatch")
    if seeds["scene_seeds"] != resolved["scene_seeds"]:
        raise ValueError("Seed and resolved manifests are not aligned")
    prompt_by_seed = dict(
        zip(
            [int(seed) for seed in resolved["scene_seeds"]],
            resolved["episode_instructions"],
            strict=True,
        )
    )
    node_by_seed = {
        int(scene["scene_seed"]): [int(node) for node in scene["replan_nodes"]]
        for scene in nodes["scenes"]
    }
    if set(node_by_seed) != set(prompt_by_seed):
        raise ValueError("Node scenes and resolved-manifest scenes differ")

    controls = []
    forced = []
    occurrence_token = 0
    for seed in seeds["scene_seeds"]:
        seed = int(seed)
        case_dir = args.output_dir / "control" / f"scene_{seed}"
        controls.append(
            {
                "arm": "control",
                "scene_seed": seed,
                "record_after_actions": node_by_seed[seed],
                "occurrence_token": occurrence_token,
                "case_dir": str(case_dir),
            }
        )
        occurrence_token += 1
    for seed in seeds["scene_seeds"]:
        seed = int(seed)
        for node in node_by_seed[seed]:
            case_dir = (
                args.output_dir / "forced" / f"scene_{seed}" / f"after_{node:03d}"
            )
            forced.append(
                {
                    "arm": "forced",
                    "scene_seed": seed,
                    "replan_after_actions": node,
                    "force_before_one_based_action": node + 1,
                    "record_after_actions": [node],
                    "occurrence_token": occurrence_token,
                    "case_dir": str(case_dir),
                }
            )
            occurrence_token += 1
    plan = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "checkpoint_dir": str(args.checkpoint_dir),
        "server_config": args.server_config,
        "horizon": args.horizon,
        "r0": args.r0,
        "inference_seed": args.seed,
        "deterministic_torch": True,
        "source_nodes": str(args.nodes),
        "source_seed_manifest": str(args.seed_manifest),
        "source_resolved_manifest": str(args.resolved_manifest),
        "controls": controls,
        "forced": forced,
    }
    if not controls or not forced:
        raise ValueError(
            f"Paired plan needs non-empty controls and forced trials; "
            f"got {len(controls)} + {len(forced)}"
        )
    for case in controls + forced:
        case_dir = Path(case["case_dir"])
        seed = int(case["scene_seed"])
        write_immutable(
            case_dir / "seed_manifest.json",
            {
                "task_name": args.task_name,
                "task_config": args.task_config,
                "base_seed": int(seeds["base_seed"]),
                "purpose": f"Deterministic paired replan {case['arm']}",
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
                "episode_instructions": [prompt_by_seed[seed]],
            },
        )
    return plan


def eval_command(args: argparse.Namespace, case: dict) -> list[str]:
    case_dir = Path(case["case_dir"])
    forced_actions = (
        [int(case["force_before_one_based_action"])]
        if case["arm"] == "forced"
        else []
    )
    name_suffix = (
        f"after{case['replan_after_actions']}"
        if case["arm"] == "forced"
        else "control"
    )
    command = [
        str(ROBOTWIN_ROOT / ".venv/bin/python"),
        "-m",
        "script.eval_policy_wandb",
        "--wandb-project",
        "robotwin-pi05-paired-replan-validation",
        "--wandb-group",
        f"{args.task_name}-r{args.r0}-paired-candidate-nodes",
        "--wandb-run-name",
        f"{args.task_name}-seed{case['scene_seed']}-{name_suffix}",
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
        *[str(node) for node in case["record_after_actions"]],
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
        f"paired_replan_seed{case['scene_seed']}_{name_suffix}",
        "--server_host",
        "127.0.0.1",
        "--server_port",
        str(args.port),
        "--seed",
        str(args.seed),
        "--instruction_type",
        "unseen",
        "--pi0_step",
        str(args.r0),
        "--pi05_intervention",
        "none",
        "--pi05_force_replan_before_actions",
        repr(forced_actions),
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--resolved-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--server-config", default="pi05_robotwin2_multitask_pytorch")
    parser.add_argument("--model-name", default="pi0.5_robotwin2")
    parser.add_argument("--task-name", default="place_a2b_left")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--r0", type=int, default=25)
    parser.add_argument("--server-gpu", default="0")
    parser.add_argument("--client-gpu", default="1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wandb-mode", choices=("offline", "online", "disabled"), default="disabled"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write and validate the task-specific plan without launching GPUs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = prepare_plan(args)
    write_immutable(args.output_dir / "experiment_plan.json", plan)
    write_json(args.output_dir / "progress.json", progress_report(plan))
    if args.prepare_only:
        print(args.output_dir / "experiment_plan.json")
        return

    cases = plan["controls"] + plan["forced"]
    pending = [
        case
        for case in cases
        if not (Path(case["case_dir"]) / "metrics.json").exists()
    ]
    print(
        f"Planned {len(cases)} trials "
        f"({len(plan['controls'])} control + {len(plan['forced'])} forced); "
        f"pending {len(pending)}",
        flush=True,
    )
    if not pending:
        summary_path = args.output_dir / "paired_summary.json"
        write_json(summary_path, final_summary(plan))
        print(summary_path)
        return

    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    server_env = os.environ | {
        "CUDA_VISIBLE_DEVICES": args.server_gpu,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": str(args.seed),
    }
    server_command = [
        str(OPENPI_ROOT / ".venv/bin/python"),
        "scripts/serve_robotwin_policy.py",
        "--config",
        args.server_config,
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--action-horizon",
        str(args.horizon),
        "--port",
        str(args.port),
        "--inference-seed",
        str(args.seed),
        "--deterministic-torch",
    ]
    with (logs / "openpi_server.log").open("a") as server_log:
        server = subprocess.Popen(
            server_command,
            cwd=OPENPI_ROOT,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_port(args.port, server)
            eval_env = os.environ | {
                "CUDA_VISIBLE_DEVICES": args.client_gpu,
                "WANDB_MODE": args.wandb_mode,
                "PYTHONHASHSEED": str(args.seed),
            }
            curobo = ROBOTWIN_ROOT / "envs_invent" / "curobo" / "src"
            eval_env["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(curobo), eval_env.get("PYTHONPATH", "")) if part
            )
            for index, case in enumerate(pending, start=1):
                case_dir = Path(case["case_dir"])
                suffix = (
                    f"after={case['replan_after_actions']}"
                    if case["arm"] == "forced"
                    else "shared-control"
                )
                print(
                    f"[{index}/{len(pending)}] {case['arm']} "
                    f"scene={case['scene_seed']} {suffix}",
                    flush=True,
                )
                with (case_dir / "eval.log").open("w") as eval_log:
                    result = subprocess.run(
                        eval_command(args, case),
                        cwd=ROBOTWIN_ROOT,
                        env=eval_env,
                        stdout=eval_log,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode != 0:
                    raise RuntimeError(f"Evaluation failed; see {case_dir / 'eval.log'}")
                write_json(args.output_dir / "progress.json", progress_report(plan))
        finally:
            stop_process_group(server)

    summary_path = args.output_dir / "paired_summary.json"
    write_json(summary_path, final_summary(plan))
    print(summary_path)


if __name__ == "__main__":
    main()
