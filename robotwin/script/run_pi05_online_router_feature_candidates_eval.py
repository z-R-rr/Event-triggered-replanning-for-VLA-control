"""Run feature routers on one shared unseen-scene set.

Each arm owns one deterministic pi0.5 server and one sequential RoboTwin client
queue. The queues run concurrently. Router queries do not sample actions; a
trigger permits at most one replacement chunk per episode. A later candidate
batch may reuse the immutable control cases from the first batch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading

from run_pi05_online_router_paired_eval import (
    analyze_pair,
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
MINIMAL_ROOT = PROJECT_ROOT / "temp/outputs/replan_router_minimal_validation"


def parse_candidates(args: argparse.Namespace) -> dict[str, Path]:
    if not args.candidate:
        return {
            "V0": args.v0_checkpoint.resolve(),
            "E1": args.e1_checkpoint.resolve(),
            "E2": args.e2_checkpoint.resolve(),
        }
    candidates: dict[str, Path] = {}
    for raw in args.candidate:
        if "=" not in raw:
            raise ValueError("--candidate must use NAME=/absolute/checkpoint.pt")
        name, path_text = raw.split("=", 1)
        if name == "control" or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Invalid candidate name: {name!r}")
        path = Path(path_text)
        if not path.is_absolute():
            raise ValueError(f"Candidate checkpoint must be absolute: {path}")
        if name in candidates:
            raise ValueError(f"Duplicate candidate name: {name}")
        candidates[name] = path.resolve()
    maximum = 4 if args.control_source_plan else 3
    if not 1 <= len(candidates) <= maximum:
        raise ValueError(
            f"Provide between one and {maximum} --candidate entries"
        )
    return candidates


def build_arm_specs(
    args: argparse.Namespace,
    candidates: dict[str, Path],
    execution_arm_order: tuple[str, ...],
) -> dict[str, dict]:
    first_checkpoint = next(iter(candidates.values()))
    checkpoints = {"control": first_checkpoint, **candidates}
    gpus = [value.strip() for value in args.gpus.split(",")]
    if len(gpus) != len(execution_arm_order):
        raise ValueError(
            f"--gpus must contain exactly {len(execution_arm_order)} "
            "comma-separated IDs"
        )
    specs = {}
    for index, arm in enumerate(execution_arm_order):
        checkpoint = checkpoints[arm]
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing {arm} checkpoint: {checkpoint}")
        specs[arm] = {
            "arm": arm,
            "router_enabled": arm != "control",
            "gpu": gpus[index],
            "port": args.port_base + index,
            "router_checkpoint": str(checkpoint),
            "router_checkpoint_sha256": sha256_file(checkpoint),
        }
    return specs


def prepare_plan(args: argparse.Namespace) -> dict:
    candidates = parse_candidates(args)
    arm_order = ("control", *candidates)
    execution_arm_order = (
        tuple(candidates) if args.control_source_plan else arm_order
    )
    if not args.output_dir.is_absolute():
        raise ValueError("--output-dir must be absolute")
    seed_manifest = json.loads(args.scene_manifest.read_text())
    resolved_manifest = json.loads(args.resolved_scene_manifest.read_text())
    for payload in (seed_manifest, resolved_manifest):
        if (
            payload["task_name"] != args.task_name
            or payload["task_config"] != args.task_config
        ):
            raise ValueError("Scene manifest task/config mismatch")
    seeds = [int(value) for value in seed_manifest["scene_seeds"]]
    resolved_seeds = [int(value) for value in resolved_manifest["scene_seeds"]]
    prompts = [str(value) for value in resolved_manifest["episode_instructions"]]
    strata = seed_manifest.get("control_outcome_strata")
    resolved_strata = resolved_manifest.get("control_outcome_strata")
    if strata is not None:
        strata = [str(value) for value in strata]
        if strata != [str(value) for value in resolved_strata or []]:
            raise ValueError("Control-outcome strata mismatch between manifests")
        if len(strata) != len(seeds):
            raise ValueError("Control-outcome strata are not aligned with scenes")
        allowed_strata = {"control_failure", "control_success"}
        if set(strata) - allowed_strata:
            raise ValueError(f"Unexpected control-outcome strata: {set(strata)}")
    else:
        strata = ["unstratified"] * len(seeds)
    if seeds != resolved_seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seed and resolved manifests are not uniquely aligned")
    if len(seeds) < args.scene_count:
        raise ValueError(f"Need {args.scene_count} scenes, manifest has {len(seeds)}")
    seeds = seeds[: args.scene_count]
    prompts = prompts[: args.scene_count]
    strata = strata[: args.scene_count]
    if args.scene_limit is not None:
        if not 1 <= args.scene_limit <= len(seeds):
            raise ValueError("--scene-limit is outside the selected scene set")
        seeds = seeds[: args.scene_limit]
        prompts = prompts[: args.scene_limit]
        strata = strata[: args.scene_limit]

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
    router_nodes_path = args.output_dir / "router_nodes_manifest.json"
    write_immutable(
        router_nodes_path,
        {
            "purpose": (
                "Leakage-free common online query clocks for "
                + "/".join(candidates)
            ),
            "query_interval": args.query_interval,
            "natural_r0_boundaries_excluded": True,
            "scenes": [
                {"scene_seed": seed, "candidate_nodes": candidate_nodes}
                for seed in seeds
            ],
        },
    )

    arm_specs = build_arm_specs(args, candidates, execution_arm_order)
    cases_by_arm = {arm: [] for arm in arm_order}
    if args.control_source_plan:
        source_plan_path = args.control_source_plan.resolve()
        source_plan = json.loads(source_plan_path.read_text())
        for field, expected in (
            ("task_name", args.task_name),
            ("task_config", args.task_config),
            ("horizon", args.horizon),
            ("r0", args.r0),
            ("lambda", args.router_lambda),
            ("max_replans", args.router_max_replans),
            ("router_query_interval", args.query_interval),
            ("router_query_max_action", args.query_max_action),
            ("inference_seed", args.seed),
        ):
            if source_plan.get(field) != expected:
                raise ValueError(
                    f"Source control plan {field} mismatch: "
                    f"{source_plan.get(field)!r} != {expected!r}"
                )
        source_controls = {
            int(case["scene_seed"]): case
            for case in source_plan["cases_by_arm"]["control"]
        }
        if set(source_controls) != set(seeds):
            raise ValueError("Source control plan scene set does not match")
        for occurrence, (seed, prompt, stratum) in enumerate(
            zip(seeds, prompts, strata, strict=True)
        ):
            copy_case = dict(source_controls[seed])
            if (
                copy_case["prompt_plaintext"] != prompt
                or int(copy_case["occurrence_token"]) != occurrence
            ):
                raise ValueError(
                    f"Source control prompt/order mismatch for scene {seed}"
                )
            copy_case["expected_control_outcome"] = stratum
            cases_by_arm["control"].append(copy_case)

    arms_to_create = (
        tuple(candidates) if args.control_source_plan else arm_order
    )
    for arm in arms_to_create:
        for occurrence, (seed, prompt, stratum) in enumerate(
            zip(seeds, prompts, strata, strict=True)
        ):
            case_dir = args.output_dir / arm / f"scene_{seed}"
            case = {
                "arm": arm,
                "router_enabled": arm != "control",
                "scene_seed": seed,
                "repeat": 0,
                "candidate_nodes": candidate_nodes,
                "occurrence_token": occurrence,
                "case_dir": str(case_dir),
                "prompt_plaintext": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "expected_control_outcome": stratum,
            }
            cases_by_arm[arm].append(case)
            write_immutable(
                case_dir / "seed_manifest.json",
                {
                    "task_name": args.task_name,
                    "task_config": args.task_config,
                    "base_seed": 100000 * (args.seed + 1),
                    "purpose": f"Online feature candidate {arm}",
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

    source_control = (
        {
            "experiment_plan": str(args.control_source_plan.resolve()),
            "experiment_plan_sha256": sha256_file(args.control_source_plan),
        }
        if args.control_source_plan
        else None
    )
    return {
        "purpose": (
            f"{len(arm_order)}-arm unseen-scene online comparison: "
            + "/".join(arm_order)
        ),
        "scope_limit": (
            f"Query every {args.query_interval} completed actions, exclude "
            f"natural r0 boundaries, and trigger at most "
            f"{args.router_max_replans} replan per episode."
        ),
        "task_name": args.task_name,
        "task_config": args.task_config,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "server_config": args.server_config,
        "horizon": args.horizon,
        "r0": args.r0,
        "lambda": args.router_lambda,
        "max_replans": args.router_max_replans,
        "min_replan_interval": 0,
        "router_query_interval": args.query_interval,
        "router_query_max_action": args.query_max_action,
        "absolute_r0_cadence": True,
        "deterministic_torch": True,
        "inference_seed": args.seed,
        "heldout_test_scenes": seeds,
        "control_outcome_strata": strata,
        "source_scene_manifest": str(args.scene_manifest.resolve()),
        "source_scene_manifest_sha256": sha256_file(args.scene_manifest),
        "source_resolved_scene_manifest": str(args.resolved_scene_manifest.resolve()),
        "source_resolved_scene_manifest_sha256": sha256_file(
            args.resolved_scene_manifest
        ),
        "feature_archive": str(args.feature_archive.resolve()),
        "feature_archive_sha256": sha256_file(args.feature_archive),
        "feature_manifest": str(args.feature_manifest.resolve()),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "router_nodes_manifest": str(router_nodes_path),
        "arm_order": list(arm_order),
        "execution_arm_order": list(execution_arm_order),
        "arm_specs": arm_specs,
        "source_control": source_control,
        "cases_by_arm": cases_by_arm,
        # analyze_pair consumes these common plan fields.
        "controls": cases_by_arm["control"],
        "routers": [case for arm in arm_order[1:] for case in cases_by_arm[arm]],
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
        "robotwin-pi05-online-feature-candidates",
        "--wandb-group",
        f"{args.task_name}-{'-'.join(plan['arm_order'][1:])}-unseen20",
        "--wandb-run-name",
        f"{args.task_name}-{case['arm']}-seed{case['scene_seed']}",
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
        f"online_feature_candidate_{case['arm']}_seed{case['scene_seed']}",
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
        repr(case["router_enabled"]),
        "--pi05_router_nodes_manifest",
        plan["router_nodes_manifest"],
        "--pi05_router_lambda",
        str(args.router_lambda),
        "--pi05_router_max_replans",
        str(args.router_max_replans),
        "--pi05_router_min_replan_interval",
        "0",
    ]


def progress(plan: dict) -> dict:
    return {
        "planned_episodes": sum(len(cases) for cases in plan["cases_by_arm"].values()),
        "new_planned_episodes": sum(
            len(plan["cases_by_arm"][arm])
            for arm in plan["execution_arm_order"]
        ),
        "reused_control_episodes": (
            len(plan["cases_by_arm"]["control"])
            if "control" not in plan["execution_arm_order"]
            else 0
        ),
        "completed_episodes": sum(
            (Path(case["case_dir"]) / "metrics.json").exists()
            for cases in plan["cases_by_arm"].values()
            for case in cases
        ),
        "by_arm": {
            arm: {
                "planned": len(cases),
                "completed": sum(
                    (Path(case["case_dir"]) / "metrics.json").exists() for case in cases
                ),
            }
            for arm, cases in plan["cases_by_arm"].items()
        },
    }


def final_summary(plan: dict) -> dict:
    controls = {
        int(case["scene_seed"]): case for case in plan["cases_by_arm"]["control"]
    }
    candidate_summaries = {}
    for arm in plan["arm_order"][1:]:
        pairs = [
            analyze_pair(plan, controls[int(case["scene_seed"])], case)
            for case in plan["cases_by_arm"][arm]
        ]
        complete = [pair for pair in pairs if pair["status"] == "complete"]
        valid = [pair for pair in complete if pair["valid_pair"]]
        by_control_stratum = {}
        for stratum, expected_control_success in (
            ("control_failure", False),
            ("control_success", True),
        ):
            stratum_pairs = [
                pair
                for pair in valid
                if controls[int(pair["scene_seed"])].get(
                    "expected_control_outcome"
                )
                == stratum
            ]
            if not stratum_pairs:
                continue
            control_matches = sum(
                bool(pair["control_success"]) is expected_control_success
                for pair in stratum_pairs
            )
            router_successes = sum(
                bool(pair["router_success"]) for pair in stratum_pairs
            )
            by_control_stratum[stratum] = {
                "planned_pairs": sum(
                    case.get("expected_control_outcome") == stratum
                    for case in plan["cases_by_arm"][arm]
                ),
                "valid_pairs": len(stratum_pairs),
                "control_outcome_matches_selection": control_matches,
                "router_successes": router_successes,
                "router_success_rate": router_successes / len(stratum_pairs),
                "rescues": sum(
                    pair["transition"] == "rescue" for pair in stratum_pairs
                ),
                "harms": sum(
                    pair["transition"] == "harm" for pair in stratum_pairs
                ),
            }
        candidate_summaries[arm] = {
            "planned_pairs": len(pairs),
            "complete_pairs": len(complete),
            "valid_pairs": len(valid),
            "triggered_pairs": sum(pair["trigger_node"] is not None for pair in valid),
            "control_successes": sum(pair["control_success"] for pair in valid),
            "router_successes": sum(pair["router_success"] for pair in valid),
            "rescues": sum(pair["transition"] == "rescue" for pair in valid),
            "harms": sum(pair["transition"] == "harm" for pair in valid),
            "both_success": sum(pair["transition"] == "both_success" for pair in valid),
            "both_failure": sum(pair["transition"] == "both_failure" for pair in valid),
            "by_control_stratum": by_control_stratum,
            "pairs": pairs,
        }
    return {
        "purpose": plan["purpose"],
        "design": {
            "scenes": len(plan["heldout_test_scenes"]),
            "arms": list(plan["arm_order"]),
            "execution_arms": list(plan["execution_arm_order"]),
            "planned_episodes": sum(
                len(cases) for cases in plan["cases_by_arm"].values()
            ),
            "new_executed_episodes": sum(
                len(plan["cases_by_arm"][arm])
                for arm in plan["execution_arm_order"]
            ),
            "reused_control_episodes": (
                len(plan["cases_by_arm"]["control"])
                if "control" not in plan["execution_arm_order"]
                else 0
            ),
            "query_interval": plan["router_query_interval"],
            "lambda": plan["lambda"],
            "horizon": plan["horizon"],
            "r0": plan["r0"],
            "max_replans": plan["max_replans"],
            "ports": {arm: spec["port"] for arm, spec in plan["arm_specs"].items()},
            "gpus": {arm: spec["gpu"] for arm, spec in plan["arm_specs"].items()},
        },
        "candidate_summaries": candidate_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--resolved-scene-manifest", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=20)
    parser.add_argument(
        "--control-source-plan",
        type=Path,
        help=(
            "Reuse immutable control cases from a completed first-batch "
            "experiment plan; permits four router candidates on four GPUs."
        ),
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        help="Smoke-only prefix of the immutable scene set.",
    )
    parser.add_argument("--task-name", default="move_playingcard_away")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--r0", type=int, default=25)
    parser.add_argument("--query-interval", type=int, default=5)
    parser.add_argument("--query-max-action", type=int, default=398)
    parser.add_argument("--router-max-replans", type=int, default=1)
    parser.add_argument("--router-lambda", type=float, default=0.05)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--port-base", type=int, default=8400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-mode", default="disabled")
    parser.add_argument(
        "--feature-archive",
        type=Path,
        default=MINIMAL_ROOT / "features_vision_encoder.npz",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=MINIMAL_ROOT / "features_vision_encoder_manifest.json",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        metavar="NAME=/ABSOLUTE/CHECKPOINT.pt",
        help=(
            "Router candidate. Repeat up to three times with a new control, "
            "or four times with --control-source-plan. If omitted, uses the "
            "legacy V0/E1/E2 checkpoint arguments."
        ),
    )
    parser.add_argument(
        "--v0-checkpoint",
        type=Path,
        default=FEATURE_ROOT / "checkpoints/V0.pt",
    )
    parser.add_argument(
        "--e1-checkpoint",
        type=Path,
        default=FEATURE_ROOT / "checkpoints/E1.pt",
    )
    parser.add_argument(
        "--e2-checkpoint",
        type=Path,
        default=FEATURE_ROOT / "checkpoints/E2.pt",
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
            "This experiment is fixed to H=50, r0=25, query interval=5, max_replans=1"
        )
    if args.scene_count < 1:
        parser.error("--scene-count must be positive")
    if args.port_base != 8400 and args.scene_limit is None:
        parser.error("The full run must start at port 8400")
    if not 0.0 < args.router_lambda < 1.0:
        parser.error("--router-lambda must be between zero and one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = prepare_plan(args)
    write_immutable(args.output_dir / "experiment_plan.json", plan)
    write_json(args.output_dir / "progress.json", progress(plan))
    print(json.dumps(progress(plan), indent=2), flush=True)
    if args.prepare_only:
        return

    for spec in plan["arm_specs"].values():
        require_port_available(int(spec["port"]))
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    servers = {}
    server_logs = []
    stop_event = threading.Event()
    try:
        for arm in plan["execution_arm_order"]:
            spec = plan["arm_specs"][arm]
            server_env = os.environ | {
                "CUDA_VISIBLE_DEVICES": str(spec["gpu"]),
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
                spec["router_checkpoint"],
                "--action-horizon",
                str(args.horizon),
                "--port",
                str(spec["port"]),
                "--inference-seed",
                str(args.seed),
                "--deterministic-torch",
            ]
            log_stream = (logs / f"openpi_{arm}_port_{spec['port']}.log").open("a")
            server_logs.append(log_stream)
            servers[arm] = subprocess.Popen(
                command,
                cwd=OPENPI_ROOT,
                env=server_env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for arm in plan["execution_arm_order"]:
            wait_for_port(int(plan["arm_specs"][arm]["port"]), servers[arm])

        curobo = ROBOTWIN_ROOT / "envs_invent/curobo/src"
        progress_lock = threading.Lock()

        def run_arm(arm: str) -> None:
            spec = plan["arm_specs"][arm]
            eval_env = os.environ | {
                "CUDA_VISIBLE_DEVICES": str(spec["gpu"]),
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
            pending = [
                case
                for case in plan["cases_by_arm"][arm]
                if not (Path(case["case_dir"]) / "metrics.json").exists()
            ]
            for index, case in enumerate(pending, start=1):
                if stop_event.is_set():
                    return
                case_dir = Path(case["case_dir"])
                print(
                    f"[{arm} {index}/{len(pending)}] "
                    f"seed={case['scene_seed']} port={spec['port']}",
                    flush=True,
                )
                with (case_dir / "eval.log").open("w") as stream:
                    result = subprocess.run(
                        eval_command(args, plan, case, int(spec["port"])),
                        cwd=ROBOTWIN_ROOT,
                        env=eval_env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode != 0:
                    stop_event.set()
                    raise RuntimeError(f"{arm} failed; see {case_dir / 'eval.log'}")
                with progress_lock:
                    write_json(args.output_dir / "progress.json", progress(plan))

        with ThreadPoolExecutor(
            max_workers=len(plan["execution_arm_order"])
        ) as executor:
            futures = {
                executor.submit(run_arm, arm): arm
                for arm in plan["execution_arm_order"]
            }
            for future in as_completed(futures):
                future.result()
    finally:
        for server in servers.values():
            stop_process_group(server)
        for stream in server_logs:
            stream.close()

    summary = final_summary(plan)
    summary_path = args.output_dir / "online_feature_candidates_summary.json"
    write_json(summary_path, summary)
    expected = len(plan["heldout_test_scenes"])
    invalid = {
        arm: payload["valid_pairs"]
        for arm, payload in summary["candidate_summaries"].items()
        if payload["valid_pairs"] != expected
    }
    if invalid:
        raise RuntimeError(f"Incomplete/invalid paired analysis: {invalid}")
    stratum_mismatches = {}
    for arm, payload in summary["candidate_summaries"].items():
        mismatched = {
            stratum: {
                "valid_pairs": values["valid_pairs"],
                "control_outcome_matches_selection": values[
                    "control_outcome_matches_selection"
                ],
            }
            for stratum, values in payload["by_control_stratum"].items()
            if values["control_outcome_matches_selection"] != values["valid_pairs"]
        }
        if mismatched:
            stratum_mismatches[arm] = mismatched
    if stratum_mismatches:
        raise RuntimeError(
            "Control outcomes no longer match the frozen balanced cohort: "
            f"{stratum_mismatches}"
        )
    print(summary_path)


if __name__ == "__main__":
    main()
