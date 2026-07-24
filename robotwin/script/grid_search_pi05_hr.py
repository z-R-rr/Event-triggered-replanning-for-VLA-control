"""Exhaustive, reproducible H/r grid evaluation for the RoboTwin pi05 policy.

Each candidate uses a fresh OpenPI server and the same immutable scene-seed
manifest. Progress is checkpointed in ``grid_results.json`` after every pair,
so interrupted runs resume without repeating completed configurations.
"""

import argparse
import json
from pathlib import Path
import os
import signal
import shutil
import socket
import subprocess
import sys
import time


ROOT = Path("/home/ubuntu/Workspace")
OPENPI_ROOT = ROOT / "openpi"
ROBOTWIN_ROOT = ROOT / "RoboTwin"
CHECKPOINT_ROOT = (
    OPENPI_ROOT
    / "outputs/robotwin_place_block_aba_grid/checkpoints/pi05_robotwin_place_block_aba_grid_lora"
    / "place-block-aba-grid-pi05-base-20260712T145757Z"
)


def wait_for_port(port: int, server: subprocess.Popen, timeout_s: int = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Policy server exited early with code {server.returncode}.")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(2)
    raise TimeoutError(f"Policy server did not open port {port} within {timeout_s}s.")


def stop_server(server: subprocess.Popen) -> None:
    """Terminate a server and any descendants it created before the next trial."""
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def is_grid_wandb_run(path: Path) -> bool:
    metadata_path = path / "files/wandb-metadata.json"
    if not metadata_path.exists():
        return False
    try:
        run_args = json.loads(metadata_path.read_text()).get("args", [])
        return run_args[run_args.index("--ckpt_setting") + 1].startswith("grid_trial_")
    except (ValueError, IndexError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-values", type=int, nargs="+", default=[45, 50, 55, 60, 65, 70])
    parser.add_argument("--r-values", type=int, nargs="+", default=[30, 35, 40, 45, 50])
    parser.add_argument("--eval-num", type=int, default=50)
    parser.add_argument("--checkpoint-step", type=int, default=15000)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Explicit OpenPI checkpoint directory. Overrides the single-task --checkpoint-step default.",
    )
    parser.add_argument(
        "--server-config", default="pi05_robotwin_place_block_aba_grid_lora",
        help="Registered OpenPI training config used to construct policy metadata/transforms.",
    )
    parser.add_argument("--task-name", default="place_block_aba_grid")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument(
        "--allow-output-task-mismatch", action="store_true",
        help="Allow an output path outside eval_result/<task-name>/... (normally rejected to prevent mislabelled trials).",
    )
    parser.add_argument(
        "--model-name", default="place-block-aba-grid-pi05-base-20260712T145757Z",
        help="RoboTwin run metadata only; it does not choose the server checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--server-gpu", default="2")
    parser.add_argument("--client-gpu", default="3")
    parser.add_argument(
        "--deterministic-xla",
        action="store_true",
        help="Request deterministic GPU XLA kernels for paired diagnostic runs.",
    )
    parser.add_argument(
        "--deterministic-torch",
        action="store_true",
        help="Require deterministic PyTorch CUDA kernels for paired diagnostic runs.",
    )
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--wandb-project", default="robotwin-pi05-finetune")
    parser.add_argument("--wandb-group", default="robotwin-place-block-aba-grid-grid50")
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    parser.add_argument(
        "--seed-manifest", type=Path, default=None,
        help="Use an existing fixed scene manifest (must contain --eval-num seeds).",
    )
    parser.add_argument(
        "--trace-root", type=Path, default=None,
        help="Optional root for compact per-action diagnostic traces; omit for normal evaluations.",
    )
    parser.add_argument(
        "--resolved-manifest", type=Path, default=None,
        help="Companion output/input manifest that locks actual instruction text per episode.",
    )
    parser.add_argument(
        "--pi05-intervention", choices=("none", "clear_after_first_place", "pause_then_clear_after_first_place", "oracle_replan_all_phase_changes", "replan_on_topp_failure"), default="none",
        help="Diagnostic-only replan intervention; default leaves the policy unchanged.",
    )
    parser.add_argument("--pi05-pause-steps", type=int, default=0, help="Physics-only pause used with pause_then_clear_after_first_place.")
    parser.add_argument(
        "--pi05-force-replan-before-actions", type=int, nargs="*", default=[],
        help="Diagnostic-only global action indices at which to cut the current chunk before execution.",
    )
    parser.add_argument(
        "--trial-tag", default="grid_trial",
        help="Prefix for RoboTwin evaluator directories; use a unique tag for an isolated diagnostic run.",
    )
    parser.add_argument("--csl-probes", action="store_true", help="Enable non-intervening VLA chunk-survival shadow probes.")
    parser.add_argument("--csl-probe-interval", type=int, default=10)
    parser.add_argument("--csl-compare-horizon", type=int, default=8)
    parser.add_argument(
        "--strict-manifest-expert-revalidation", action="store_true",
        help="Fail a trial if a later expert replay disagrees with manifest-time validity.",
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROBOTWIN_ROOT / "eval_result/place_block_aba_grid/pi05_remote/grid_search_15000_seed0_eval50",
    )
    args = parser.parse_args()

    # The experiment directory is part of the audit trail.  Catch the easy-to-
    # miss case where --task-name is changed but a previous task's output root
    # is copied into the command.
    output_parts = args.output_dir.resolve().parts
    if "eval_result" in output_parts and not args.allow_output_task_mismatch:
        task_index = output_parts.index("eval_result") + 1
        if task_index < len(output_parts) and output_parts[task_index] != args.task_name:
            raise ValueError(
                f"Output task component {output_parts[task_index]!r} disagrees with "
                f"--task-name {args.task_name!r}: {args.output_dir}. "
                "Use a matching eval_result/<task-name>/... root."
            )

    pairs = [(h, r) for h in args.h_values for r in args.r_values if r <= h]
    if not pairs:
        raise ValueError("No valid H/r pairs (require r <= H).")
    checkpoint_dir = args.checkpoint_dir or (CHECKPOINT_ROOT / str(args.checkpoint_step))
    if not (checkpoint_dir / "params").exists() and not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Checkpoint needs either JAX params/ or PyTorch model.safetensors: {checkpoint_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    results_path = args.output_dir / "grid_results.json"
    manifest_path = args.seed_manifest or (args.output_dir / f"seed_manifest_seed{args.seed}_eval{args.eval_num}.json")
    resolved_manifest_path = args.resolved_manifest or (args.output_dir / "resolved_episode_manifest.json")
    # Fail before loading a multi-GB policy if a copied output directory points
    # to another task's immutable scene manifest.
    if args.seed_manifest is not None and not manifest_path.exists():
        raise FileNotFoundError(
            f"Explicit --seed-manifest does not exist: {manifest_path}. "
            "Refusing to let the evaluator generate a different default manifest."
        )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("task_name") != args.task_name
            or manifest.get("task_config") != args.task_config
            or not isinstance(manifest.get("scene_seeds"), list)
            or len(manifest["scene_seeds"]) != args.eval_num
        ):
            raise ValueError(
                f"Incompatible seed manifest for task {args.task_name!r}: {manifest_path}. "
                "Use a fresh output directory; immutable manifests are never overwritten."
            )
    if args.reset:
        for path in [logs_dir, *args.output_dir.glob("trial_*")]:
            if path.exists():
                shutil.rmtree(path)
        reset_files = [results_path, args.output_dir / "best_result.json"]
        # An explicit manifest is an immutable experimental input. Never
        # delete it merely because this output directory is being reset.
        if args.seed_manifest is None:
            reset_files.append(manifest_path)
        for path in reset_files:
            path.unlink(missing_ok=True)
        episode_root = ROBOTWIN_ROOT / "eval_result" / args.task_name / "pi05_remote" / args.task_config
        for path in episode_root.glob("grid_trial_*"):
            shutil.rmtree(path)
        for path in (ROBOTWIN_ROOT / "wandb").glob("offline-run-*"):
            if is_grid_wandb_run(path):
                shutil.rmtree(path)
    logs_dir.mkdir(exist_ok=True)

    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.setdefault("checkpoint_step", args.checkpoint_step)
    if results.get("task_name") not in (None, args.task_name):
        raise ValueError(f"Existing grid results belong to task {results['task_name']!r}, not {args.task_name!r}.")
    if results.get("task_config") not in (None, args.task_config):
        raise ValueError(f"Existing grid results use task_config {results['task_config']!r}, not {args.task_config!r}.")
    results.setdefault("task_name", args.task_name)
    results.setdefault("task_config", args.task_config)
    results.setdefault("seed", args.seed)
    results.setdefault("eval_num", args.eval_num)
    results.setdefault("manifest", str(manifest_path))
    results.setdefault("resolved_manifest", str(resolved_manifest_path))
    results.setdefault("trials", {})

    for index, (horizon, execute_steps) in enumerate(pairs):
        key = f"H{horizon}_r{execute_steps}"
        previous = results["trials"].get(key, {})
        if previous.get("status") == "complete":
            print(f"Skipping completed {key}: {previous['success_rate']:.3f}")
            continue

        trial_dir = args.output_dir / f"trial_{index:03d}_{key}"
        trial_dir.mkdir(exist_ok=True)
        metrics_path = trial_dir / "metrics.json"
        server_log = logs_dir / f"trial_{index:03d}_server.log"
        eval_log = logs_dir / f"trial_{index:03d}_eval.log"
        server_env = os.environ | {"CUDA_VISIBLE_DEVICES": args.server_gpu}
        if args.deterministic_torch:
            # cuBLAS reads this at process start; the server additionally
            # enables torch deterministic algorithms before loading the model.
            server_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            server_env["PYTHONHASHSEED"] = str(args.seed)
        if (checkpoint_dir / "model.safetensors").exists():
            # The mixed RoboTwin checkpoint is PyTorch.  JAX is still imported
            # for the server protocol, but must not preallocate the GPU that
            # PyTorch owns for inference.
            server_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        else:
            server_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
        if args.deterministic_xla:
            # Keep any caller-provided flags and only append the documented XLA
            # deterministic-kernel switch. This is intentionally opt-in: it
            # can trade inference throughput for paired-evaluation fidelity.
            existing_xla_flags = server_env.get("XLA_FLAGS", "").strip()
            server_env["XLA_FLAGS"] = " ".join(
                flag for flag in (existing_xla_flags, "--xla_gpu_deterministic_ops") if flag
            )
        server_cmd = [
            str(OPENPI_ROOT / ".venv/bin/python"), "scripts/serve_robotwin_policy.py",
            "--config", args.server_config,
            "--checkpoint-dir", str(checkpoint_dir),
            "--action-horizon", str(horizon), "--port", str(args.port),
            "--inference-seed", str(args.seed),
        ]
        if args.deterministic_torch:
            server_cmd.append("--deterministic-torch")
        with server_log.open("w") as server_file:
            server = subprocess.Popen(
                server_cmd,
                cwd=OPENPI_ROOT,
                env=server_env,
                stdout=server_file,
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
                # RoboTwin vendors CuRobo (including its Python-3.10 CUDA extensions)
                # under envs_invent.  Keep it on the RoboTwin client interpreter's
                # import path; the OpenPI server remains in its own venv.
                curobo_src = ROBOTWIN_ROOT / "envs_invent" / "curobo" / "src"
                if curobo_src.is_dir():
                    existing_pythonpath = eval_env.get("PYTHONPATH", "")
                    eval_env["PYTHONPATH"] = os.pathsep.join(
                        part for part in (str(curobo_src), existing_pythonpath) if part
                    )
                eval_cmd = [
                    str(ROBOTWIN_ROOT / ".venv/bin/python"), "-m", "script.eval_policy_wandb",
                    "--wandb-project", args.wandb_project,
                    "--wandb-group", args.wandb_group,
                    "--wandb-run-name", f"hr-grid-step-{args.checkpoint_step}-{key}",
                    "--eval-num", str(args.eval_num),
                    "--action-horizon", str(horizon),
                    "--metrics-json", str(metrics_path),
                    "--seed-manifest", str(manifest_path),
                    "--resolved-manifest", str(resolved_manifest_path),
                    "--config", "policy/pi05/deploy_policy.yml", "--overrides",
                    "--task_name", args.task_name, "--task_config", args.task_config,
                    "--policy_name", "pi05_remote",
                    "--model_name", args.model_name,
                    "--checkpoint_id", str(args.checkpoint_step),
                    "--ckpt_setting", f"{args.trial_tag}_{index}_{key}",
                    "--server_host", "127.0.0.1", "--server_port", str(args.port),
                    "--seed", str(args.seed), "--instruction_type", "unseen",
                    "--pi0_step", str(execute_steps),
                    "--pi05_intervention", args.pi05_intervention,
                    "--pi05_pause_steps", str(args.pi05_pause_steps),
                    "--pi05_force_replan_before_actions", repr(args.pi05_force_replan_before_actions),
                ]
                if args.csl_probes:
                    eval_cmd.extend([
                        "--csl-probes",
                        "--csl-probe-interval", str(args.csl_probe_interval),
                        "--csl-compare-horizon", str(args.csl_compare_horizon),
                    ])
                if args.strict_manifest_expert_revalidation:
                    eval_cmd.append("--strict-manifest-expert-revalidation")
                if args.trace_root is not None:
                    eval_cmd.extend(["--trace-dir", str(args.trace_root / key)])
                with eval_log.open("w") as eval_file:
                    result = subprocess.run(eval_cmd, cwd=ROBOTWIN_ROOT, env=eval_env, stdout=eval_file, stderr=subprocess.STDOUT)
                if result.returncode != 0 or not metrics_path.exists():
                    raise RuntimeError(f"Evaluation failed; see {eval_log}")
                metrics = json.loads(metrics_path.read_text())
                results["trials"][key] = {
                    "status": "complete", "H": horizon, "r": execute_steps,
                    "success_rate": float(metrics["success_rate"]),
                    "successes": int(metrics["successes"]), "episodes": int(metrics["episodes"]),
                    "metrics_path": str(metrics_path),
                }
                print(f"Completed {key}: {metrics['successes']}/{metrics['episodes']}")
            except Exception as exc:
                results["trials"][key] = {"status": "failed", "H": horizon, "r": execute_steps, "error": str(exc)}
                print(f"Failed {key}: {exc}", file=sys.stderr)
            finally:
                stop_server(server)
                results_path.write_text(json.dumps(results, indent=2))

    complete = [value for value in results["trials"].values() if value["status"] == "complete"]
    if complete:
        best = max(complete, key=lambda value: (value["success_rate"], -value["H"], -value["r"]))
        (args.output_dir / "best_result.json").write_text(json.dumps(best, indent=2))
        print(json.dumps({"completed": len(complete), "total": len(pairs), "best": best}, indent=2))
    if args.wandb_mode == "offline":
        print(f"Deferred W&B upload:\n  cd {ROBOTWIN_ROOT} && .venv/bin/wandb sync wandb/offline-run-*")


if __name__ == "__main__":
    main()
