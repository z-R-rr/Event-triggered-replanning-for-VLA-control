"""Recover exact pi0.5 prompt text by matching historical trace fingerprints.

This utility never edits the seed manifest, traces, or grid results.  It
replays each expert scene only far enough to regenerate all language
candidates, then accepts a prompt only when its SHA-256 exactly matches the
fingerprint recorded by every available H/r trace for that scene.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import yaml


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(ROBOTWIN_ROOT / "description" / "utils"))

from envs import CONFIGS_PATH  # noqa: E402
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def prompt_fingerprints(trace_root: Path) -> dict[int, str]:
    by_seed: dict[int, set[str]] = {}
    for path in sorted(trace_root.glob("H*_r*/scene_*.json")):
        trace = json.loads(path.read_text())
        seed = int(trace["scene_seed"])
        values = {
            chunk["prompt_fingerprint"]
            for chunk in trace.get("chunks", [])
            if chunk.get("prompt_fingerprint")
        }
        if len(values) != 1:
            raise ValueError(f"{path}: expected one prompt fingerprint, got {sorted(values)}")
        by_seed.setdefault(seed, set()).update(values)
    inconsistent = {seed: values for seed, values in by_seed.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"Prompt differs across H/r for scene seeds: {inconsistent}")
    return {seed: next(iter(values)) for seed, values in by_seed.items()}


def build_task(task_name: str, task_config: str):
    with (ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml").open() as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    args.update(
        {
            "task_name": task_name,
            "task_config": task_config,
            "ckpt_setting": "prompt_recovery",
            "policy_name": "pi05_remote",
            "eval_mode": True,
            "render_freq": 0,
        }
    )
    with Path(CONFIGS_PATH, "_embodiment_config.yml").open() as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)
    with Path(CONFIGS_PATH, "_camera_config.yml").open() as handle:
        cameras = yaml.load(handle.read(), Loader=yaml.FullLoader)
    camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[camera_type]["h"]
    args["head_camera_w"] = cameras[camera_type]["w"]

    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"Unexpected embodiment config: {embodiment}")
    for side in ("left", "right"):
        robot_file = Path(args[f"{side}_robot_file"])
        with (robot_file / "config.yml").open() as handle:
            args[f"{side}_embodiment_config"] = yaml.load(handle.read(), Loader=yaml.FullLoader)

    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)(), args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--max-descriptions", type=int, default=100)
    parser.add_argument("--resume-report", type=Path, default=None)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--prompt-override",
        action="append",
        default=[],
        metavar="SEED=TEXT",
        help="Supply an independently recovered plaintext; its SHA-256 must match the trace.",
    )
    args = parser.parse_args()

    source = json.loads(args.seed_manifest.read_text())
    task_name = source["task_name"]
    task_config = source["task_config"]
    scene_seeds = [int(seed) for seed in source["scene_seeds"]]
    expected = prompt_fingerprints(args.trace_root)
    missing_trace = [seed for seed in scene_seeds if seed not in expected]
    if missing_trace:
        raise ValueError(f"Missing trace fingerprints for {len(missing_trace)} seeds: {missing_trace}")

    task, task_args = build_task(task_name, task_config)
    resumed: dict[int, str] = {}
    if args.resume_report and args.resume_report.exists():
        previous = json.loads(args.resume_report.read_text())
        resumed = {
            int(row["scene_seed"]): row["recovered_instruction"]
            for row in previous.get("episodes_detail", [])
            if row.get("recovered_instruction") is not None
        }
        print(f"Resuming with {len(resumed)} previously verified prompts.", flush=True)
    for value in args.prompt_override:
        seed_text, prompt = value.split("=", maxsplit=1)
        seed = int(seed_text)
        actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if seed not in expected or actual != expected[seed]:
            raise ValueError(
                f"Prompt override for seed {seed} does not match trace fingerprint: {actual}"
            )
        resumed[seed] = prompt
        print(f"Accepted exact-hash prompt override for seed {seed}.", flush=True)

    recovered: list[str | None] = []
    rows = []
    try:
        for index, seed in enumerate(scene_seeds):
            if seed in resumed:
                recovered.append(resumed[seed])
                rows.append(
                    {
                        "episode_index": index,
                        "scene_seed": seed,
                        "expected_prompt_fingerprint": expected[seed],
                        "candidate_count": None,
                        "unique_candidate_count": None,
                        "matching_prompt_count": 1,
                        "recovered_instruction": resumed[seed],
                        "attempts": 0,
                        "error": None,
                        "resumed": True,
                    }
                )
                continue

            error = None
            candidates: list[str] = []
            matches: list[str] = []
            attempts = 0
            all_candidates: set[str] = set()
            for attempt in range(args.max_attempts):
                attempts = attempt + 1
                seed_everything(seed)
                try:
                    task.setup_demo(now_ep_num=index, seed=seed, is_test=True, **task_args)
                    episode = task.play_once()
                    generated = generate_episode_descriptions(
                        task_name, [episode["info"]], args.max_descriptions
                    )
                    candidates = generated[0][args.instruction_type]
                    all_candidates.update(candidates)
                    matches = sorted(
                        {
                            candidate
                            for candidate in all_candidates
                            if hashlib.sha256(candidate.encode("utf-8")).hexdigest() == expected[seed]
                        }
                    )
                except Exception as exc:  # Keep a complete recovery audit.
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    task.close_env(
                        clear_cache=((index + 1) % task_args["clear_cache_freq"] == 0)
                    )
                if matches:
                    break

            prompt = matches[0] if len(matches) == 1 else None
            recovered.append(prompt)
            rows.append(
                {
                    "episode_index": index,
                    "scene_seed": seed,
                    "expected_prompt_fingerprint": expected[seed],
                    "candidate_count": len(all_candidates),
                    "unique_candidate_count": len(all_candidates),
                    "matching_prompt_count": len(matches),
                    "recovered_instruction": prompt,
                    "attempts": attempts,
                    "error": error,
                    "resumed": False,
                }
            )
            print(
                f"[{index + 1:03d}/{len(scene_seeds)}] seed={seed} "
                f"matches={len(matches)} candidates={len(all_candidates)} attempts={attempts}"
                + (f" error={error}" if error else ""),
                flush=True,
            )
    finally:
        try:
            task.close_env()
        except Exception:
            pass

    matched = sum(prompt is not None for prompt in recovered)
    report = {
        "purpose": "Recover historical prompt plaintext by exact SHA-256 trace matching.",
        "source_seed_manifest": str(args.seed_manifest),
        "trace_root": str(args.trace_root),
        "task_name": task_name,
        "task_config": task_config,
        "instruction_type": args.instruction_type,
        "episodes": len(scene_seeds),
        "exactly_recovered": matched,
        "all_exactly_recovered": matched == len(scene_seeds),
        "episodes_detail": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if matched != len(scene_seeds):
        raise RuntimeError(
            f"Recovered {matched}/{len(scene_seeds)} prompts; wrote audit report only: {args.report}"
        )

    resolved = {
        "task_name": task_name,
        "task_config": task_config,
        "source_seed_manifest": str(args.seed_manifest),
        "scene_seeds": scene_seeds,
        "episode_instructions": recovered,
        "recovery": {
            "method": "SHA-256 exact match against historical trace prompt_fingerprint",
            "trace_root": str(args.trace_root),
            "report": str(args.report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resolved, indent=2, ensure_ascii=False))
    print(f"Recovered all {matched} prompts: {args.output}")


if __name__ == "__main__":
    main()
