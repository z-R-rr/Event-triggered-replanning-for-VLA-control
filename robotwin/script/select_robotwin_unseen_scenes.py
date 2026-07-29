"""Select deterministic expert-valid RoboTwin scenes absent from prior data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import yaml


WORKSPACE = Path("/home/ubuntu/Workspace")
PROJECT_ROOT = WORKSPACE / "Event-triggered-replanning-for-VLA-control"
ROBOTWIN_ROOT = WORKSPACE / "RoboTwin"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def existing_task_seeds(task_name: str, excluded_output: Path) -> tuple[set[int], list[str]]:
    roots = [
        ROBOTWIN_ROOT / "eval_result" / task_name,
        PROJECT_ROOT / "temp",
    ]
    seen: set[int] = set()
    sources = []
    excluded_output = excluded_output.resolve()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                if excluded_output in path.resolve().parents:
                    continue
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("task_name") != task_name:
                continue
            seeds = payload.get("scene_seeds")
            if not isinstance(seeds, list):
                continue
            try:
                values = {int(seed) for seed in seeds}
            except (TypeError, ValueError):
                continue
            if values:
                seen.update(values)
                sources.append(str(path.resolve()))
    feature_archive = (
        PROJECT_ROOT
        / "temp/outputs/replan_router_minimal_validation/"
        "features_vision_encoder.npz"
    )
    if feature_archive.exists():
        with np.load(feature_archive) as archive:
            seen.update(int(seed) for seed in archive["scene_seed"])
        sources.append(str(feature_archive.resolve()))
    return seen, sorted(sources)


def build_task_args(task_name: str, task_config: str) -> dict:
    # Match the evaluator launch environment.  Scene selection instantiates
    # RoboTwin tasks (and therefore the CuRobo planner) even though it does
    # not run policy inference.
    curobo_src = ROBOTWIN_ROOT / "envs_invent" / "curobo" / "src"
    if curobo_src.is_dir():
        sys.path.insert(0, str(curobo_src))
    sys.path.insert(0, str(ROBOTWIN_ROOT))
    sys.path.insert(0, str(ROBOTWIN_ROOT / "script"))
    from envs import CONFIGS_PATH
    import script.eval_policy as eval_policy

    with (ROBOTWIN_ROOT / f"task_config/{task_config}.yml").open(
        encoding="utf-8"
    ) as stream:
        task_args = yaml.load(stream.read(), Loader=yaml.FullLoader)
    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["ckpt_setting"] = "unseen_scene_selection"

    embodiment = task_args["embodiment"]
    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"),
        encoding="utf-8",
    ) as stream:
        embodiment_types = yaml.load(stream.read(), Loader=yaml.FullLoader)

    def embodiment_file(name: str) -> str:
        return embodiment_types[name]["file_path"]

    if len(embodiment) == 1:
        task_args["left_robot_file"] = embodiment_file(embodiment[0])
        task_args["right_robot_file"] = embodiment_file(embodiment[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        task_args["left_robot_file"] = embodiment_file(embodiment[0])
        task_args["right_robot_file"] = embodiment_file(embodiment[1])
        task_args["embodiment_dis"] = embodiment[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"Unsupported embodiment: {embodiment}")
    task_args["left_embodiment_config"] = eval_policy.get_embodiment_config(
        task_args["left_robot_file"]
    )
    task_args["right_embodiment_config"] = eval_policy.get_embodiment_config(
        task_args["right_robot_file"]
    )
    with open(
        os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8"
    ) as stream:
        cameras = yaml.load(stream.read(), Loader=yaml.FullLoader)
    head_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = cameras[head_type]["h"]
    task_args["head_camera_w"] = cameras[head_type]["w"]
    task_args["policy_name"] = "pi05_remote"
    task_args["eval_mode"] = True
    return task_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default="move_playingcard_away")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--candidate-start", type=int, default=200000)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--instruction-type", default="unseen")
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        parser.error("--output-dir must be absolute")
    if args.count < 1:
        parser.error("--count must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_dir / "unseen_seed_manifest.json"
    resolved_path = args.output_dir / "unseen_resolved_episode_manifest.json"
    audit_path = args.output_dir / "unseen_selection_audit.json"
    if seed_path.exists() or resolved_path.exists() or audit_path.exists():
        if all(path.exists() for path in (seed_path, resolved_path, audit_path)):
            print(seed_path)
            return
        raise FileExistsError("Incomplete existing unseen-scene selection")

    seen, seen_sources = existing_task_seeds(
        args.task_name, args.output_dir
    )
    task_args = build_task_args(args.task_name, args.task_config)
    import script.eval_policy as eval_policy
    from envs.utils.create_actor import UnStableError

    task_env = eval_policy.class_decorator(args.task_name)
    selected = []
    prompts = []
    rejected_seen = []
    rejected_invalid = []
    candidate = args.candidate_start
    while len(selected) < args.count:
        if candidate in seen:
            rejected_seen.append(candidate)
            candidate += 1
            continue
        seed_everything(candidate)
        valid = False
        episode_info = None
        error = None
        try:
            task_env.setup_demo(
                now_ep_num=len(selected),
                seed=candidate,
                is_test=True,
                **task_args,
            )
            episode_info = task_env.play_once()
            valid = bool(task_env.plan_success and task_env.check_success())
        except UnStableError as exc:
            error = f"UnStableError: {exc}"
        except Exception as exc:  # preserve a full selection audit
            error = f"{type(exc).__name__}: {exc}"
        finally:
            task_env.close_env()
        if valid:
            descriptions = eval_policy.generate_episode_descriptions(
                args.task_name, [episode_info["info"]], args.count
            )
            candidates = descriptions[0][args.instruction_type]
            prompt = str(candidates[candidate % len(candidates)])
            selected.append(candidate)
            prompts.append(prompt)
            print(
                f"[{len(selected):03d}/{args.count}] seed={candidate} "
                f"prompt={prompt!r}",
                flush=True,
            )
        else:
            rejected_invalid.append(
                {"scene_seed": candidate, "error": error}
            )
            print(f"[reject] seed={candidate} error={error}", flush=True)
        candidate += 1

    seed_manifest = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "base_seed": 100000 * (args.policy_seed + 1),
        "purpose": (
            f"{args.count} expert-valid scenes absent from all discovered prior "
            f"{args.task_name} manifests and router features"
        ),
        "scene_seeds": selected,
    }
    write_immutable(seed_path, seed_manifest)
    write_immutable(
        resolved_path,
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "source_seed_manifest": str(seed_path),
            "scene_seeds": selected,
            "episode_instructions": prompts,
        },
    )
    write_immutable(
        audit_path,
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "definition_of_unseen": (
                "Seed absent from all task seed/resolved manifests under "
                "RoboTwin eval_result and project temp, and absent from the "
                "router feature archive, at selection start."
            ),
            "candidate_start": args.candidate_start,
            "selected_count": len(selected),
            "selected_unique": len(set(selected)) == len(selected),
            "selected_disjoint_from_seen": not (set(selected) & seen),
            "selected_seeds": selected,
            "seen_seed_count": len(seen),
            "seen_seed_min": min(seen) if seen else None,
            "seen_seed_max": max(seen) if seen else None,
            "seen_source_count": len(seen_sources),
            "seen_sources_sha256": hashlib.sha256(
                json.dumps(seen_sources, sort_keys=True).encode()
            ).hexdigest(),
            "rejected_seen": rejected_seen,
            "rejected_invalid": rejected_invalid,
            "seed_manifest": str(seed_path),
            "seed_manifest_sha256": sha256_file(seed_path),
            "resolved_manifest": str(resolved_path),
            "resolved_manifest_sha256": sha256_file(resolved_path),
        },
    )
    print(seed_path)


if __name__ == "__main__":
    main()
