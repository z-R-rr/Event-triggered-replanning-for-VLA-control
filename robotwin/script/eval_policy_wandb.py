"""RoboTwin evaluation wrapper with compact Weights & Biases logging.

This leaves ``script/eval_policy.py`` unchanged.  It replaces only that
module's ``eval_policy`` function at runtime, so existing environment and
policy integrations are reused unchanged.

For each run, the wrapper logs success rate plus ``checkpoint_step`` and
``pi0_step``. It preserves/uploads only the first successful and first failed
episode video; all other video files from the run are removed.

Place ``--wandb-project`` and ``--wandb-run-name`` before ``--overrides``.
"""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import wandb

try:
    import script.eval_policy as base_eval
except ModuleNotFoundError:  # Supports `python script/eval_policy_wandb.py` too.
    import eval_policy as base_eval


LAST_SCENE_SEEDS: list[int] = []
LAST_EPISODE_METRICS: list[dict] = []


def _seed_everything(seed: int) -> None:
    """Seed Python/NumPy/Torch before every scene construction."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _create_or_load_seed_manifest(
    *, task_env, args: dict, st_seed: int, test_num: int, manifest_path: Path
) -> list[int]:
    """Persist the exact expert-valid scenes used by every H/r trial."""
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        scene_seeds = manifest.get("scene_seeds")
        if (
            manifest.get("task_name") != args["task_name"]
            or manifest.get("task_config") != args["task_config"]
            or manifest.get("base_seed") != st_seed
            or not isinstance(scene_seeds, list)
            or len(scene_seeds) != test_num
        ):
            raise ValueError(f"Incompatible seed manifest: {manifest_path}")
        return [int(seed) for seed in scene_seeds]

    scene_seeds: list[int] = []
    candidate_seed = st_seed
    while len(scene_seeds) < test_num:
        _seed_everything(candidate_seed)
        try:
            task_env.setup_demo(now_ep_num=len(scene_seeds), seed=candidate_seed, is_test=True, **args)
            task_env.play_once()
            valid = task_env.plan_success and task_env.check_success()
        except base_eval.UnStableError:
            valid = False
        except Exception as exc:
            print(f"Expert rollout error for candidate seed {candidate_seed}: {exc}")
            valid = False
        finally:
            task_env.close_env()
        if valid:
            scene_seeds.append(candidate_seed)
        candidate_seed += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "task_name": args["task_name"],
                "task_config": args["task_config"],
                "base_seed": st_seed,
                "scene_seeds": scene_seeds,
            },
            indent=2,
        )
    )
    return scene_seeds


def _load_or_create_resolved_manifest(
    *, path: Path, source_manifest: Path, task_name: str, task_config: str, scene_seeds: list[int]
) -> dict:
    """Persist the exact language input per manifest occurrence.

    The seed manifest is an immutable scene input.  This companion manifest is
    an output of the first run and becomes an input on later paired runs, so a
    regenerated expert description cannot silently change the VLA prompt.
    """
    if path.exists():
        resolved = json.loads(path.read_text())
        if (
            resolved.get("task_name") != task_name
            or resolved.get("task_config") != task_config
            or [int(seed) for seed in resolved.get("scene_seeds", [])] != scene_seeds
            or not isinstance(resolved.get("episode_instructions"), list)
            or len(resolved["episode_instructions"]) != len(scene_seeds)
        ):
            raise ValueError(f"Incompatible resolved instruction manifest: {path}")
        return resolved
    return {
        "task_name": task_name,
        "task_config": task_config,
        "source_seed_manifest": str(source_manifest),
        "scene_seeds": scene_seeds,
        "episode_instructions": [None] * len(scene_seeds),
    }


def _write_resolved_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    temporary.replace(path)


def eval_policy_with_wandb(
    task_name,
    task_env,
    args,
    model,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
    seed_manifest=None,
    trace_dir=None,
    observation_record_dir=None,
    observation_record_actions=None,
    csl_probes=False,
    csl_probe_interval=10,
    csl_compare_horizon=8,
    allow_manifest_expert_drift=True,
    resolved_manifest_path=None,
    episode_id_offset=0,
):
    """Evaluate and retain at most one success and one failure video."""
    global LAST_SCENE_SEEDS, LAST_EPISODE_METRICS
    task_env.suc = 0
    task_env.test_num = 0
    kept_video_paths: dict[str, Path] = {}

    eval_func = base_eval.eval_function_decorator(args["policy_name"], "eval")
    reset_func = base_eval.eval_function_decorator(args["policy_name"], "reset_model")
    args["eval_mode"] = True

    if seed_manifest is None:
        raise ValueError("A --seed-manifest is required for reproducible evaluation.")
    manifest_path = Path(seed_manifest)
    scene_seeds = _create_or_load_seed_manifest(
        task_env=task_env,
        args=args,
        st_seed=st_seed,
        test_num=test_num,
        manifest_path=manifest_path,
    )
    if resolved_manifest_path is None:
        resolved_manifest_path = manifest_path.with_name(f"{manifest_path.stem}_resolved_instructions.json")
    resolved_manifest_path = Path(resolved_manifest_path)
    resolved_manifest = _load_or_create_resolved_manifest(
        path=resolved_manifest_path,
        source_manifest=manifest_path,
        task_name=args["task_name"],
        task_config=args["task_config"],
        scene_seeds=scene_seeds,
    )
    LAST_SCENE_SEEDS = scene_seeds
    LAST_EPISODE_METRICS = []
    print(f"Seed manifest: {manifest_path}")
    print(f"Resolved instruction manifest: {resolved_manifest_path}")
    print(f"Scene seeds: {scene_seeds}")
    wandb.config.update({"seed_manifest": str(manifest_path), "scene_seeds": scene_seeds}, allow_val_change=True)

    for now_id, now_seed in enumerate(scene_seeds):
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        # Regenerate the expert trace only to derive this scene's instruction.
        # A manifest seed that is no longer expert-valid is an error, never a
        # reason to silently substitute a different test scene.
        _seed_everything(now_seed)
        try:
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info = task_env.play_once()
            task_env.close_env()
        except base_eval.UnStableError:
            task_env.close_env()
            args["render_freq"] = render_freq
            raise RuntimeError(f"Manifest scene {now_seed} became unstable.")
        except Exception as exc:
            task_env.close_env()
            args["render_freq"] = render_freq
            raise RuntimeError(f"Expert rollout failed for manifest scene {now_seed}: {exc}") from exc

        if not (task_env.plan_success and task_env.check_success()):
            # Expert validity was established when this immutable manifest was
            # created.  Some multi-object scenes have a non-deterministic
            # TOPP/physics expert replay even though the seeded policy scene
            # is usable.  Do not turn that replay-only fluctuation into
            # selective episode dropping.  Strict revalidation remains
            # available for debugging a manifest.
            if not allow_manifest_expert_drift:
                args["render_freq"] = render_freq
                raise RuntimeError(f"Manifest scene {now_seed} is no longer expert-valid.")
            print(
                f"Warning: manifest scene {now_seed} failed expert revalidation; "
                "continuing because it was validated when the manifest was created."
            )

        args["render_freq"] = render_freq
        _seed_everything(now_seed)
        task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        instruction = resolved_manifest["episode_instructions"][now_id]
        if instruction is None:
            results = base_eval.generate_episode_descriptions(task_name, [episode_info["info"]], test_num)
            instruction_candidates = results[0][instruction_type]
            # Fix the exact natural-language realization per scene. A global
            # RNG draw here makes paired H/r trials differ even for a shared
            # scene seed.
            instruction = instruction_candidates[now_seed % len(instruction_candidates)]
            resolved_manifest["episode_instructions"][now_id] = instruction
            _write_resolved_manifest(resolved_manifest_path, resolved_manifest)
        task_env.set_instruction(instruction=instruction)

        video_path = None
        if task_env.eval_video_path is not None:
            video_path = Path(task_env.eval_video_path) / f"episode{task_env.test_num}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", video_size, "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264", "-crf", "23", str(video_path),
                ],
                stdin=subprocess.PIPE,
            )
            task_env._set_eval_video_ffmpeg(ffmpeg)

        # Count only policy execution, not task construction or the preceding
        # expert rollout used to derive the language instruction.
        task_env.effective_policy_actions = 0
        task_env.policy_scene_steps = 0
        reset_func(model)
        if hasattr(model, "set_episode_seed"):
            model.set_episode_seed(now_seed)
        policy_episode_id = int(episode_id_offset) + now_id
        if hasattr(model, "set_episode_id"):
            # The occurrence token resets the server's primary inference
            # index but is deliberately not mixed into policy RNG entropy.
            # A launcher can therefore replay one scene in separate
            # one-episode processes without inheriting the prior case index.
            model.set_episode_id(policy_episode_id)
        if hasattr(model, "set_trace_enabled"):
            model.set_trace_enabled(trace_dir is not None)
        if hasattr(model, "set_observation_record_dir"):
            model.set_observation_record_dir(observation_record_dir)
        if hasattr(model, "set_observation_record_actions"):
            model.set_observation_record_actions(observation_record_actions)
        if hasattr(model, "configure_csl"):
            model.configure_csl(csl_probes, csl_probe_interval, csl_compare_horizon)
        success = False
        while task_env.take_action_cnt < task_env.step_lim:
            eval_func(task_env, model, task_env.get_obs())
            if task_env.eval_success:
                success = True
                break

        # Stop exactly when policy execution terminates, before ffmpeg/video
        # finalization and environment teardown. The policy timestamps the
        # first actual inference call, excluding scene construction/reset.
        first_inference_start = getattr(model, "first_inference_start", None)
        policy_elapsed_seconds = (
            time.perf_counter() - first_inference_start
            if first_inference_start is not None
            else 0.0
        )

        if task_env.eval_video_path is not None:
            task_env._del_eval_video_ffmpeg()

        outcome = "success" if success else "failure"
        if success:
            task_env.suc += 1
        if video_path is not None and video_path.exists():
            if outcome not in kept_video_paths:
                retained_path = video_path.with_name(f"{outcome}_example.mp4")
                video_path.rename(retained_path)
                kept_video_paths[outcome] = retained_path
            else:
                video_path.unlink()

        task_env.close_env(clear_cache=((now_id + 1) % args["clear_cache_freq"] == 0))
        if task_env.render_freq:
            task_env.viewer.close()
        episode_metrics = {
            "episode": task_env.test_num + 1,
            "episode_success": int(success),
            "running_success_rate": task_env.suc / (task_env.test_num + 1),
            "inference_calls": int(getattr(model, "inference_calls", 0)),
            "effective_policy_actions": int(getattr(task_env, "effective_policy_actions", 0)),
            "scene_steps": int(getattr(task_env, "policy_scene_steps", 0)),
            "elapsed_seconds": policy_elapsed_seconds,
        }
        LAST_EPISODE_METRICS.append(episode_metrics)
        if trace_dir is not None:
            # A manifest may intentionally contain the same seed multiple
            # times for reproducibility checks.  Keep every closed-loop run:
            # naming only by scene seed silently overwrote earlier repeats.
            episode_trace_path = Path(trace_dir) / f"scene_{now_seed}_episode_{now_id:03d}.json"
            episode_trace_path.parent.mkdir(parents=True, exist_ok=True)
            episode_trace_path.write_text(json.dumps({
                "scene_seed": now_seed,
                "episode_id": now_id,
                "policy_episode_id": policy_episode_id,
                "instruction": instruction,
                "episode_metrics": episode_metrics,
                "actions": getattr(model, "action_traces", []),
                "chunks": getattr(model, "chunk_traces", []),
            }, indent=2))
        wandb.log(episode_metrics, step=task_env.test_num + 1)
        print(
            f"Episode {task_env.test_num + 1}/{test_num}: {outcome} | "
            f"inference_calls={episode_metrics['inference_calls']} "
            f"effective_policy_actions={episode_metrics['effective_policy_actions']} "
            f"scene_steps={episode_metrics['scene_steps']} "
            f"elapsed_seconds={episode_metrics['elapsed_seconds']:.2f}"
        )
        task_env.test_num += 1

    for outcome, path in kept_video_paths.items():
        wandb.log({f"{outcome}_example": wandb.Video(str(path), format="mp4")})

    totals = {
        key: sum(metric[key] for metric in LAST_EPISODE_METRICS)
        for key in ("inference_calls", "effective_policy_actions", "scene_steps", "elapsed_seconds")
    }
    wandb.log({
        "success_rate": task_env.suc / test_num,
        "successes": task_env.suc,
        "episodes": test_num,
        **{f"{key}_total": value for key, value in totals.items()},
        **{f"{key}_mean": value / test_num for key, value in totals.items()},
    })
    return scene_seeds[-1] + 1, task_env.suc


def main() -> None:
    # ``--seed`` belongs to RoboTwin's later ``--overrides`` parser. Disable
    # argparse abbreviation so it is never mistaken for ``--seed-manifest``.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--wandb-project", default="robotwin-pi05-finetune")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="robotwin-place-block-aba-grid-eval")
    parser.add_argument("--eval-num", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=50, help="Predicted action chunk length H on server.")
    parser.add_argument("--metrics-json", default=None, help="Optional path for machine-readable final metrics.")
    parser.add_argument("--trace-dir", default=None, help="Optional compact per-action trace output directory; use only for diagnostic replays.")
    parser.add_argument(
        "--observation-record-dir",
        default=None,
        help="Optional directory for compressed raw VLA inputs at primary inference boundaries.",
    )
    parser.add_argument(
        "--observation-record-actions", type=int, nargs="*", default=[],
        help="Global actions whose post-execution observations should be saved.",
    )
    parser.add_argument("--csl-probes", action="store_true", help="Record non-intervening VLA chunk-survival shadow probes.")
    parser.add_argument("--csl-probe-interval", type=int, default=10)
    parser.add_argument("--csl-compare-horizon", type=int, default=8)
    parser.add_argument("--seed-manifest", required=True, help="Fixed expert-valid scene seed manifest for this evaluation.")
    parser.add_argument(
        "--resolved-manifest",
        default=None,
        help="Companion manifest that locks the exact instruction text per episode occurrence.",
    )
    parser.add_argument(
        "--strict-manifest-expert-revalidation", action="store_true",
        help="Fail if an expert replay later disagrees with manifest-time validity (debugging only).",
    )
    parser.add_argument(
        "--episode-id-offset", type=int, default=0,
        help=(
            "Occurrence-token offset sent to the policy server. It resets the "
            "per-episode inference index but does not add RNG entropy."
        ),
    )
    wandb_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    user_args = base_eval.parse_args_and_config()
    if int(user_args["pi0_step"]) > wandb_args.action_horizon:
        raise ValueError("pi0_step (executed actions r) cannot exceed action_horizon (predicted chunk H).")

    run_name = wandb_args.wandb_run_name or (
        f"{user_args['model_name']}-eval-step-{user_args['checkpoint_id']}-H{wandb_args.action_horizon}"
        f"-r{user_args['pi0_step']}"
    )
    wandb.init(
        project=wandb_args.wandb_project,
        group=wandb_args.wandb_group,
        name=run_name,
        config={
            "task_name": user_args["task_name"],
            "task_config": user_args["task_config"],
            "checkpoint_step": int(user_args["checkpoint_id"]),
            "action_horizon": wandb_args.action_horizon,
            "pi0_step": int(user_args["pi0_step"]),
            "executed_actions": int(user_args["pi0_step"]),
            "instruction_type": user_args["instruction_type"],
            "eval_num": wandb_args.eval_num,
            "strict_manifest_expert_revalidation": wandb_args.strict_manifest_expert_revalidation,
        },
    )
    try:
        original_eval = base_eval.eval_policy
        base_eval.eval_policy = lambda *args, **kwargs: eval_policy_with_wandb(
            *args, **(kwargs | {
                "test_num": wandb_args.eval_num,
                "seed_manifest": wandb_args.seed_manifest,
                "resolved_manifest_path": wandb_args.resolved_manifest,
                "trace_dir": wandb_args.trace_dir,
                "observation_record_dir": wandb_args.observation_record_dir,
                "observation_record_actions": wandb_args.observation_record_actions,
                "csl_probes": wandb_args.csl_probes,
                "csl_probe_interval": wandb_args.csl_probe_interval,
                "csl_compare_horizon": wandb_args.csl_compare_horizon,
                "allow_manifest_expert_drift": not wandb_args.strict_manifest_expert_revalidation,
                "episode_id_offset": wandb_args.episode_id_offset,
            })
        )
        base_eval.main(user_args)
        base_eval.eval_policy = original_eval
        if wandb_args.metrics_json:
            metrics_path = Path(wandb_args.metrics_json)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "checkpoint_step": int(user_args["checkpoint_id"]),
                        "action_horizon": wandb_args.action_horizon,
                        "pi0_step": int(user_args["pi0_step"]),
                        "success_rate": float(wandb.run.summary.get("success_rate", float("nan"))),
                        "successes": int(wandb.run.summary.get("successes", 0)),
                        "episodes": int(wandb.run.summary.get("episodes", 0)),
                        "scene_seeds": LAST_SCENE_SEEDS,
                        "seed_manifest": wandb_args.seed_manifest,
                        "resolved_manifest": wandb_args.resolved_manifest,
                        "episode_metrics": LAST_EPISODE_METRICS,
                        **{
                            key: wandb.run.summary.get(key)
                            for key in (
                                "inference_calls_total", "inference_calls_mean",
                                "effective_policy_actions_total", "effective_policy_actions_mean",
                                "scene_steps_total", "scene_steps_mean",
                                "elapsed_seconds_total", "elapsed_seconds_mean",
                            )
                        },
                    },
                    indent=2,
                )
            )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
