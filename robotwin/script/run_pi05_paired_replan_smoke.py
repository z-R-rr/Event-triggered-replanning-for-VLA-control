"""Deterministic paired smoke test for one forced-replan boundary.

The control and treatment start from the same scene, instruction, checkpoint,
policy RNG stream, and r0.  A node ``t`` means that both arms execute actions
1..t from the same chunks.  The treatment then replans immediately before
one-based action ``t + 1``; the control continues the old chunk.

This launcher deliberately repeats each arm inside one evaluation process.
The repeated scene seed is paired with occurrence ids 0, 1, 2, which makes the
OpenPI server reset its per-episode inference index without adding entropy.
"""

from __future__ import annotations

import argparse
import copy
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


def write_immutable(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite incompatible input: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


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


def sha256_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def observation_fingerprint(path: Path) -> str:
    """Match policy.pi05_remote._observation_fingerprint exactly."""
    digest = hashlib.sha256()
    with np.load(path) as data:
        digest.update(np.ascontiguousarray(data["state"], dtype=np.float32).tobytes())
        for key in ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
            digest.update(np.ascontiguousarray(data[key]).tobytes())
    return digest.hexdigest()


def trace_paths(case_dir: Path, repeats: int) -> list[Path]:
    paths = sorted((case_dir / "traces").glob("scene_*_episode_*.json"))
    if len(paths) != repeats:
        raise RuntimeError(
            f"Expected {repeats} traces under {case_dir / 'traces'}, found {len(paths)}"
        )
    return paths


def target_sequence(trace: dict) -> list:
    chunks = {
        int(chunk["inference_call"]): chunk["chunk_actions"]
        for chunk in trace["chunks"]
    }
    targets = []
    for row in trace["actions"]:
        call = int(row["inference_call"])
        index = int(row["chunk_action_index"])
        targets.append(chunks[call][index])
    return targets


def _canonicalize_unordered_contact_pairs(value):
    """Canonicalize SAPIEN's unordered contact enumeration for exact hashing."""
    if isinstance(value, dict):
        canonical = {
            key: _canonicalize_unordered_contact_pairs(item)
            for key, item in value.items()
        }
        if isinstance(canonical.get("pairs"), list):
            canonical["pairs"] = sorted(
                canonical["pairs"],
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":"), allow_nan=False
                ),
            )
        return canonical
    if isinstance(value, list):
        return [_canonicalize_unordered_contact_pairs(item) for item in value]
    return value


def normalized_trace(trace: dict) -> dict:
    """Remove occurrence/path/timing fields and sort unordered contact pairs."""
    normalized = copy.deepcopy(trace)
    normalized.pop("episode_id", None)
    normalized.pop("policy_episode_id", None)
    metrics = normalized.get("episode_metrics", {})
    for key in ("episode", "running_success_rate", "elapsed_seconds"):
        metrics.pop(key, None)
    for chunk in normalized.get("chunks", []):
        chunk.pop("observation_record_path", None)
    for action in normalized.get("actions", []):
        action.pop("observation_record_path_after_action", None)
    return _canonicalize_unordered_contact_pairs(normalized)


def arm_report(
    case_dir: Path,
    repeats: int,
    node: int,
    forced: bool,
    *,
    r0: int = 25,
    absolute_r0_cadence: bool = False,
) -> dict:
    traces = []
    rows = []
    for path in trace_paths(case_dir, repeats):
        trace = json.loads(path.read_text())
        episode_id = int(trace["episode_id"])
        node_path = (
            case_dir
            / "observations"
            / f"episode_{episode_id:03d}_action_{node:03d}_after.npz"
        )
        if not node_path.exists():
            raise RuntimeError(f"Missing node observation: {node_path}")
        fingerprint = observation_fingerprint(node_path)
        targets = target_sequence(trace)
        markers = [
            int(action)
            for chunk in trace["chunks"]
            for action in chunk.get("forced_replan_before_actions", [])
        ]
        forced_replan_chunk = next(
            (
                index
                for index, chunk in enumerate(trace["chunks"])
                if node + 1 in chunk.get("forced_replan_before_actions", [])
            ),
            None,
        )
        replacement_fingerprint = None
        replacement_execution_limit = None
        replacement_next_boundary = None
        if forced_replan_chunk is not None and forced_replan_chunk + 1 < len(trace["chunks"]):
            replacement_chunk = trace["chunks"][forced_replan_chunk + 1]
            replacement_fingerprint = replacement_chunk.get("observation_fingerprint")
            replacement_execution_limit = replacement_chunk.get("executed_r")
            replacement_next_boundary = replacement_chunk.get(
                "absolute_r0_next_boundary"
            )
        chunk_starts = [0]
        action_clock = 0
        for chunk in trace["chunks"][1:]:
            action_clock += int(chunk["previous_chunk_cursor"])
            chunk_starts.append(action_clock)
        rows.append(
            {
                "episode_id": episode_id,
                "trace_path": str(path),
                "success": bool(trace["episode_metrics"]["episode_success"]),
                "effective_policy_actions": int(
                    trace["episode_metrics"]["effective_policy_actions"]
                ),
                "inference_calls": int(trace["episode_metrics"]["inference_calls"]),
                "scene_steps": int(trace["episode_metrics"]["scene_steps"]),
                "node_observation_path": str(node_path),
                "node_observation_fingerprint": fingerprint,
                "prefix_target_sha256": sha256_json(targets[:node]),
                "full_target_sha256": sha256_json(targets),
                "normalized_trace_sha256": sha256_json(normalized_trace(trace)),
                "forced_replan_before_actions": markers,
                "replacement_observation_fingerprint": replacement_fingerprint,
                "replacement_execution_limit": replacement_execution_limit,
                "replacement_next_boundary": replacement_next_boundary,
                "all_chunks_absolute_r0_cadence": all(
                    chunk.get("absolute_r0_cadence") is True
                    for chunk in trace["chunks"]
                ),
                "chunk_start_actions": chunk_starts,
            }
        )
        traces.append(trace)

    exact_fields = (
        "success",
        "effective_policy_actions",
        "inference_calls",
        "scene_steps",
        "node_observation_fingerprint",
        "prefix_target_sha256",
        "full_target_sha256",
        "normalized_trace_sha256",
        "forced_replan_before_actions",
        "replacement_observation_fingerprint",
        "replacement_execution_limit",
        "replacement_next_boundary",
        "all_chunks_absolute_r0_cadence",
        "chunk_start_actions",
    )
    checks = {
        f"{field}_identical": len(
            {json.dumps(row[field], sort_keys=True) for row in rows}
        )
        == 1
        for field in exact_fields
    }
    expected_markers = [node + 1] if forced else []
    checks["expected_forced_marker"] = all(
        row["forced_replan_before_actions"] == expected_markers for row in rows
    )
    if forced:
        checks["replacement_uses_node_observation"] = all(
            row["replacement_observation_fingerprint"]
            == row["node_observation_fingerprint"]
            for row in rows
        )
    if absolute_r0_cadence:
        checks["absolute_r0_cadence_enabled"] = all(
            row["all_chunks_absolute_r0_cadence"] for row in rows
        )
        checks["absolute_replan_boundaries"] = all(
            all(
                start == node or start % r0 == 0
                for start in row["chunk_start_actions"]
            )
            for row in rows
        )
        if forced:
            expected_limit = r0 - node % r0
            checks["replacement_ends_at_next_absolute_boundary"] = all(
                row["replacement_execution_limit"] == expected_limit
                and row["replacement_next_boundary"] == node + expected_limit
                and (node + expected_limit) % r0 == 0
                for row in rows
            )
    return {
        "arm": "forced" if forced else "control",
        "repeats": repeats,
        "checks": checks,
        "exactly_reproducible": all(checks.values()),
        "runs": rows,
    }


def analyze(
    output_dir: Path,
    repeats: int,
    scene_seed: int,
    node: int,
    r0: int = 25,
    absolute_r0_cadence: bool = False,
) -> dict:
    control = arm_report(
        output_dir / "control",
        repeats,
        node,
        forced=False,
        r0=r0,
        absolute_r0_cadence=absolute_r0_cadence,
    )
    forced = arm_report(
        output_dir / "forced",
        repeats,
        node,
        forced=True,
        r0=r0,
        absolute_r0_cadence=absolute_r0_cadence,
    )
    all_rows = control["runs"] + forced["runs"]
    cross_arm_checks = {
        "same_scene_seed": all(
            json.loads(Path(row["trace_path"]).read_text())["scene_seed"] == scene_seed
            for row in all_rows
        ),
        "all_node_observations_identical": len(
            {row["node_observation_fingerprint"] for row in all_rows}
        )
        == 1,
        "all_prefix_targets_identical": len(
            {row["prefix_target_sha256"] for row in all_rows}
        )
        == 1,
    }
    passed = (
        control["exactly_reproducible"]
        and forced["exactly_reproducible"]
        and all(cross_arm_checks.values())
    )
    return {
        "purpose": "Deterministic paired control/forced-replan smoke test.",
        "scene_seed": scene_seed,
        "r0": r0,
        "replan_after_actions": node,
        "force_before_one_based_action": node + 1,
        "absolute_r0_cadence": bool(absolute_r0_cadence),
        "repeats_per_arm": repeats,
        "determinism_contract": {
            "server": "torch deterministic algorithms; cuDNN deterministic; TF32 disabled",
            "policy_rng": "same episode_seed; occurrence id resets inference index but adds no entropy",
            "node_state": "bitwise SHA-256 over state and all three RGB inputs",
            "prefix": f"bitwise-identical policy targets for effective actions 1..{node}",
            "within_arm": (
                "exact normalized full execution trace, targets, outcome, and counters; "
                "unordered SAPIEN contact-pair enumeration is content-sorted"
            ),
        },
        "control": control,
        "forced": forced,
        "cross_arm_checks": cross_arm_checks,
        "passed": passed,
    }


def make_eval_command(
    args: argparse.Namespace,
    arm: str,
    case_dir: Path,
    forced: bool,
) -> list[str]:
    force_actions = [args.node + 1] if forced else []
    return [
        str(ROBOTWIN_ROOT / ".venv/bin/python"),
        "-m",
        "script.eval_policy_wandb",
        "--wandb-project",
        "robotwin-pi05-paired-replan-smoke",
        "--wandb-group",
        f"{args.task_name}-r{args.r0}-scene{args.scene_seed}-after{args.node}",
        "--wandb-run-name",
        f"{args.task_name}-{arm}-repeat{args.repeats}",
        "--eval-num",
        str(args.repeats),
        "--action-horizon",
        str(args.horizon),
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
        str(args.node),
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
        f"paired_smoke_{arm}_scene{args.scene_seed}_after{args.node}",
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
        repr(force_actions),
        "--pi05_absolute_r0_cadence",
        repr(bool(args.absolute_r0_cadence)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-seed-manifest", type=Path, required=True)
    parser.add_argument("--source-resolved-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--server-config", default="pi05_robotwin2_multitask_pytorch")
    parser.add_argument("--model-name", default="pi0.5_robotwin2")
    parser.add_argument("--task-name", default="place_a2b_left")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--scene-seed", type=int, default=100000)
    parser.add_argument("--node", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
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
        "--absolute-r0-cadence",
        action="store_true",
        help=(
            "Insert the forced replan while keeping later natural boundaries "
            "anchored at r0, 2*r0, ..."
        ),
    )
    args = parser.parse_args()

    source_seeds = json.loads(args.source_seed_manifest.read_text())
    source_resolved = json.loads(args.source_resolved_manifest.read_text())
    if source_seeds["task_name"] != args.task_name:
        raise ValueError("Source seed manifest task mismatch")
    if source_seeds["task_config"] != args.task_config:
        raise ValueError("Source seed manifest task-config mismatch")
    prompt_by_seed = dict(
        zip(
            [int(seed) for seed in source_resolved["scene_seeds"]],
            source_resolved["episode_instructions"],
            strict=True,
        )
    )
    if args.scene_seed not in prompt_by_seed:
        raise ValueError(f"Scene seed {args.scene_seed} is absent from resolved manifest")
    if args.node <= 0:
        raise ValueError("Smoke node must be a positive post-action boundary")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "checkpoint_dir": str(args.checkpoint_dir),
        "server_config": args.server_config,
        "model_name": args.model_name,
        "scene_seed": args.scene_seed,
        "instruction": prompt_by_seed[args.scene_seed],
        "horizon": args.horizon,
        "r0": args.r0,
        "replan_after_actions": args.node,
        "control": "continue the old chunk at the node",
        "forced": f"discard old tail and infer before action {args.node + 1}",
        "repeats_per_arm": args.repeats,
        "deterministic_torch": True,
        "absolute_r0_cadence": bool(args.absolute_r0_cadence),
        "inference_seed": args.seed,
    }
    write_immutable(args.output_dir / "protocol.json", protocol)
    for arm in ("control", "forced"):
        case_dir = args.output_dir / arm
        write_immutable(
            case_dir / "seed_manifest.json",
            {
                "task_name": args.task_name,
                "task_config": args.task_config,
                # eval_policy.py derives its manifest base from the evaluator
                # seed, while scene_seeds may intentionally repeat any
                # selected manifest scene for the reproducibility smoke.
                "base_seed": 100000 * (1 + args.seed),
                "purpose": f"Paired deterministic smoke: {arm}",
                "scene_seeds": [args.scene_seed] * args.repeats,
            },
        )
        write_immutable(
            case_dir / "resolved_episode_manifest.json",
            {
                "task_name": args.task_name,
                "task_config": args.task_config,
                "source_seed_manifest": str(case_dir / "seed_manifest.json"),
                "scene_seeds": [args.scene_seed] * args.repeats,
                "episode_instructions": [prompt_by_seed[args.scene_seed]] * args.repeats,
            },
        )

    report_path = args.output_dir / "reproducibility_report.json"
    if all((args.output_dir / arm / "metrics.json").exists() for arm in ("control", "forced")):
        report = analyze(
            args.output_dir,
            args.repeats,
            args.scene_seed,
            args.node,
            args.r0,
            args.absolute_r0_cadence,
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(report_path)
        if not report["passed"]:
            raise RuntimeError("Existing paired smoke does not satisfy reproducibility contract")
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
    with (logs / "openpi_server.log").open("w") as server_log:
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
            for arm, forced in (("control", False), ("forced", True)):
                case_dir = args.output_dir / arm
                if (case_dir / "metrics.json").exists():
                    continue
                print(f"Running {arm}: {args.repeats} exact repeats", flush=True)
                with (logs / f"{arm}_eval.log").open("w") as eval_log:
                    result = subprocess.run(
                        make_eval_command(args, arm, case_dir, forced),
                        cwd=ROBOTWIN_ROOT,
                        env=eval_env,
                        stdout=eval_log,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{arm} evaluation failed; see {logs / f'{arm}_eval.log'}"
                    )
        finally:
            stop_process_group(server)

    report = analyze(
        args.output_dir,
        args.repeats,
        args.scene_seed,
        args.node,
        args.r0,
        args.absolute_r0_cadence,
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)
    if not report["passed"]:
        raise RuntimeError("Paired smoke does not satisfy reproducibility contract")


if __name__ == "__main__":
    main()
