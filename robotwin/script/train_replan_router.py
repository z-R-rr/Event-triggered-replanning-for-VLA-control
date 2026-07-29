"""Offline pi0.5 feature extraction and event-triggered replan-router training.

This script never starts RoboTwin and never calls ``sample_actions``.  It
loads the published pi0.5 checkpoint in eval/inference mode, extracts frozen
features from observations already archived by strict shared-prefix paired
evaluations, and trains a sub-1M-parameter outcome router.

The router predicts two Bernoulli outcomes:

    p_keep   = P(keep succeeds | z_v, z_a)
    p_replan = P(replan succeeds | z_v, z_a)

The binary decision is ``p_replan - p_keep > lambda``.  Both-failure pairs
are excluded before scene-grouped train/validation/test splitting.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
from typing import Any
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WORKSPACE = Path("/home/ubuntu/Workspace")
PROJECT_ROOT = WORKSPACE / "Event-triggered-replanning-for-VLA-control"
ROBOTWIN_ROOT = WORKSPACE / "RoboTwin"
OPENPI_ROOT = WORKSPACE / "openpi"
DEFAULT_CHECKPOINT = Path("/home/ubuntu/Model/pi0.5_robotwin2")
DEFAULT_SERVER_CONFIG = "pi05_robotwin2_multitask_pytorch"
DEFAULT_ALOHA_URDF = (
    ROBOTWIN_ROOT / "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
)

SPLIT_NAMES = ("train", "val", "test")
SPLIT_TO_ID = {name: index for index, name in enumerate(SPLIT_NAMES)}
THRESHOLD_SWEEP = (0.05, 0.1, 0.2, 0.3, 0.5)
FEATURE_ABLATION_ORDER = tuple(f"V{index}" for index in range(8))
ADDITIONAL_ACTION_MODELING_ORDER = ("E1", "E2")
DEFAULT_TEMPORAL_K = 5
DEFAULT_EEF_WAYPOINTS = 5

MOVE_PLAYINGCARD_ROOTS = (
    ROBOTWIN_ROOT / "eval_result/move_playingcard_away/pi05_remote/"
    "hr_eval100_pi05_robotwin2_deterministic_seed0_repl01/"
    "paired_replan_node_eval_r25_absolute_cadence_round1_round2_union_deterministic",
    PROJECT_ROOT
    / "temp/outputs/move_playingcard_away_r25_initial_success_top4_absolute/"
    "eval_absolute_path_shard_0",
    PROJECT_ROOT
    / "temp/outputs/move_playingcard_away_r25_initial_success_top4_absolute/"
    "eval_absolute_path_shard_1",
)


@dataclasses.dataclass(frozen=True)
class FeatureConfig:
    name: str
    chunk_state: bool = False
    action_mode: str = "none"
    delta_vision: bool = False
    action_hidden_tail: bool = False

    @property
    def modules(self) -> tuple[str, ...]:
        modules = ["vision"]
        if self.chunk_state:
            modules.append("chunk_state")
        if self.action_mode != "none":
            modules.append(self.action_mode)
        if self.delta_vision:
            modules.append("delta_vision")
        if self.action_hidden_tail:
            modules.append("action_expert_hidden_tail")
        return tuple(modules)


FEATURE_CONFIGS = {
    "V0": FeatureConfig("V0"),
    "V1": FeatureConfig("V1", chunk_state=True),
    "V2": FeatureConfig("V2", action_mode="future_action_tail"),
    "V3": FeatureConfig("V3", action_mode="action_statistics"),
    "V4": FeatureConfig("V4", delta_vision=True),
    "V5": FeatureConfig("V5", chunk_state=True, action_mode="future_action_tail"),
    "V6": FeatureConfig("V6", chunk_state=True, delta_vision=True),
    "V7": FeatureConfig(
        "V7",
        chunk_state=True,
        action_mode="future_action_tail",
        delta_vision=True,
    ),
    # Kept outside V0--V7 so the requested full-chunk implementation is
    # available for a diagnostic run without silently adding it to the
    # prescribed ablation table.
    "C3": FeatureConfig("C3", action_mode="full_action_chunk"),
    # Geometry-aware action diagnostic. It is reported separately and is
    # never added to the prescribed --run-feature-ablation sequence.
    "E1": FeatureConfig("E1", action_mode="eef_trajectory"),
    "E2": FeatureConfig("E2", chunk_state=True, action_mode="eef_trajectory"),
    # Frozen pi0.5 action-expert final-layer hidden state, pooled only over the
    # unexecuted [cursor:H) action-token suffix at flow time tau=0.
    "Z1": FeatureConfig("Z1", action_hidden_tail=True),
}


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def observation_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        digest.update(np.ascontiguousarray(data["state"], dtype=np.float32).tobytes())
        for key in ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
            digest.update(np.ascontiguousarray(data[key]).tobytes())
    return digest.hexdigest()


def write_json_immutable(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite incompatible file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically refresh a generated index without touching run artifacts."""
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def save_npz_immutable(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite feature archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_torch_immutable(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite router checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def default_output_dir(task: str) -> Path:
    return PROJECT_ROOT / "temp/outputs/replan_router" / task


def default_feature_ablation_dir() -> Path:
    return PROJECT_ROOT / "temp/outputs/replan_router_feature_ablation"


def discover_data_roots(task: str) -> list[Path]:
    if task == "move_playingcard_away":
        roots = list(MOVE_PLAYINGCARD_ROOTS)
    else:
        roots = []
        search_roots = (
            ROBOTWIN_ROOT / "eval_result" / task,
            PROJECT_ROOT / "temp/outputs",
        )
        for search_root in search_roots:
            if search_root.exists():
                roots.extend(
                    path.parent
                    for path in search_root.rglob("paired_summary.json")
                    if task in str(path)
                )

    missing = [root for root in roots if not (root / "paired_summary.json").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing paired summaries:\n" + "\n".join(f"  {path}" for path in missing)
        )
    if not roots:
        raise FileNotFoundError(
            f"No paired data found for task={task!r}; provide one or more --data-root."
        )
    return sorted(set(path.resolve() for path in roots))


def _single_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one {description}, found {len(paths)}: {paths}"
        )
    return paths[0]


def load_decision_samples(
    task: str,
    roots: list[Path],
    *,
    expected_horizon: int = 50,
    expected_action_dim: int = 14,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: dict[tuple[str, int, int], Path] = {}
    source_reports = []
    excluded_both_failure = 0

    for root in roots:
        summary_path = root / "paired_summary.json"
        summary = json.loads(summary_path.read_text())
        plan_path = root / "experiment_plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        summary_task = plan.get("task_name", task)
        if summary_task != task:
            raise ValueError(f"Task mismatch at {root}: {summary_task!r} != {task!r}")
        if not bool(summary.get("design", {}).get("absolute_r0_cadence", False)):
            raise ValueError(f"Data root is not absolute-cadence paired eval: {root}")

        source_counts: dict[str, int] = {
            "complete": 0,
            "valid": 0,
            "included": 0,
            "excluded_both_failure": 0,
        }
        for pair in summary["pairs"]:
            if pair.get("status") != "complete":
                continue
            source_counts["complete"] += 1
            if not pair.get("valid_pair", False):
                raise ValueError(
                    f"Invalid shared-prefix pair in source data: "
                    f"seed={pair.get('scene_seed')} node={pair.get('replan_after_actions')}"
                )
            source_counts["valid"] += 1

            keep_success = int(bool(pair["control_success"]))
            replan_success = int(bool(pair["forced_success"]))
            if keep_success == 0 and replan_success == 0:
                excluded_both_failure += 1
                source_counts["excluded_both_failure"] += 1
                continue

            seed = int(pair["scene_seed"])
            node = int(pair["replan_after_actions"])
            sample_key = (task, seed, node)
            if sample_key in seen:
                raise ValueError(
                    f"Duplicate decision sample {sample_key} in {seen[sample_key]} and {root}"
                )
            seen[sample_key] = root

            case_dir = root / "forced" / f"scene_{seed}" / f"after_{node:03d}"
            trace_path = _single_path(
                sorted((case_dir / "traces").glob("scene_*_episode_*.json")),
                f"trace under {case_dir}",
            )
            observation_path = _single_path(
                sorted(
                    (case_dir / "observations").glob(f"*action_{node:03d}_after.npz")
                ),
                f"node observation under {case_dir}",
            )
            trace = json.loads(trace_path.read_text())
            if int(trace["scene_seed"]) != seed:
                raise ValueError(f"Trace seed mismatch: {trace_path}")
            instruction = str(trace["instruction"])
            prompt_sha256 = hashlib.sha256(instruction.encode("utf-8")).hexdigest()

            marker_indices = [
                index
                for index, chunk in enumerate(trace["chunks"])
                if node + 1 in chunk.get("forced_replan_before_actions", [])
            ]
            if len(marker_indices) != 1:
                raise ValueError(
                    f"Expected one forced marker before action {node + 1}: {trace_path}"
                )
            marker_index = marker_indices[0]
            if marker_index + 1 >= len(trace["chunks"]):
                raise ValueError(
                    f"Forced marker has no replacement chunk: {trace_path}"
                )
            old_chunk_row = trace["chunks"][marker_index]
            replacement_row = trace["chunks"][marker_index + 1]
            old_chunk = np.asarray(old_chunk_row["chunk_actions"], dtype=np.float32)
            if old_chunk.shape != (expected_horizon, expected_action_dim):
                raise ValueError(
                    f"Unexpected old chunk shape {old_chunk.shape}, expected "
                    f"{(expected_horizon, expected_action_dim)}: {trace_path}"
                )
            old_cursor = int(replacement_row["previous_chunk_cursor"])
            if not 0 < old_cursor < expected_horizon:
                raise ValueError(f"Invalid old chunk cursor {old_cursor}: {trace_path}")
            natural_interval = int(old_chunk_row.get("executed_r", 0))
            if natural_interval <= 0:
                raise ValueError(
                    f"Missing positive natural replanning interval: {trace_path}"
                )
            next_boundary = old_chunk_row.get("absolute_r0_next_boundary")
            if next_boundary is None:
                next_boundary = ((node // natural_interval) + 1) * natural_interval
            distance_to_next_r0 = int(next_boundary) - node
            if not 0 < distance_to_next_r0 <= natural_interval:
                raise ValueError(
                    "Invalid distance to next natural boundary "
                    f"{distance_to_next_r0}: {trace_path}"
                )
            previous_router_clocks = [
                int(action) - 1
                for chunk in trace["chunks"][:marker_index]
                for action in chunk.get("forced_replan_before_actions", [])
                if int(action) - 1 < node
            ]
            # Strict-prefix counterfactual episodes have no earlier router
            # trigger. In that censored case, distance is measured from the
            # episode origin (completed-action clock zero).
            last_replan_distance = node - (
                max(previous_router_clocks) if previous_router_clocks else 0
            )
            if last_replan_distance < 0:
                raise ValueError(f"Invalid prior router marker: {trace_path}")

            actual_fp = observation_fingerprint(observation_path)
            expected_fp = pair.get("forced_node_observation_fingerprint")
            if expected_fp is not None and actual_fp != expected_fp:
                raise ValueError(
                    f"Node observation fingerprint mismatch: {observation_path}"
                )
            replacement_fp = replacement_row.get("observation_fingerprint")
            if replacement_fp != actual_fp:
                raise ValueError(
                    f"Replacement did not use node observation: {trace_path}"
                )
            if old_chunk_row.get("prompt_fingerprint") != prompt_sha256:
                raise ValueError(f"Prompt fingerprint mismatch: {trace_path}")

            raw_delta = replan_success - keep_success
            if raw_delta == 1:
                transition = "rescue"
            elif raw_delta == -1:
                transition = "harm"
            else:
                transition = "both_success"
            source_counts["included"] += 1
            samples.append(
                {
                    "sample_id": f"{task}:seed{seed}:t{node}",
                    "task": task,
                    "scene_seed": seed,
                    "timestep": node,
                    "instruction": instruction,
                    "prompt_sha256": prompt_sha256,
                    "observation_path": str(observation_path.resolve()),
                    "observation_fingerprint": actual_fp,
                    "trace_path": str(trace_path.resolve()),
                    "old_action_chunk": old_chunk,
                    "old_chunk_cursor": old_cursor,
                    "cursor_ratio": old_cursor / expected_horizon,
                    "remaining_ratio": (expected_horizon - old_cursor)
                    / expected_horizon,
                    "time_since_plan": old_cursor,
                    "distance_to_next_r0": distance_to_next_r0,
                    "last_replan_distance": last_replan_distance,
                    "natural_replan_interval": natural_interval,
                    "old_chunk_sha256": hashlib.sha256(
                        np.ascontiguousarray(old_chunk).tobytes()
                    ).hexdigest(),
                    "keep_success": keep_success,
                    "replan_success": replan_success,
                    "raw_delta": raw_delta,
                    "router_label": int(raw_delta == 1),
                    "transition": transition,
                    "source_eval_root": str(root),
                }
            )

        source_reports.append(
            {
                "root": str(root),
                "paired_summary": str(summary_path),
                "paired_summary_sha256": sha256_file(summary_path),
                **source_counts,
            }
        )

    samples.sort(key=lambda row: (row["scene_seed"], row["timestep"]))
    transition_counts: dict[str, int] = {}
    for sample in samples:
        transition = sample["transition"]
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
    if not samples:
        raise ValueError(
            "No eligible samples remain after excluding both-failure pairs"
        )
    if set(transition_counts) - {"rescue", "harm", "both_success"}:
        raise AssertionError(f"Unexpected transitions: {transition_counts}")

    dataset_fingerprint = sha256_json(
        [
            {
                key: sample[key]
                for key in (
                    "sample_id",
                    "observation_fingerprint",
                    "old_chunk_sha256",
                    "keep_success",
                    "replan_success",
                    "prompt_sha256",
                )
            }
            for sample in samples
        ]
    )
    report = {
        "task": task,
        "absolute_r0_cadence_required": True,
        "both_failure_excluded": True,
        "sample_count": len(samples),
        "scene_count": len({sample["scene_seed"] for sample in samples}),
        "excluded_both_failure_count": excluded_both_failure,
        "transition_counts": transition_counts,
        "router_label_counts": {
            "keep": sum(sample["router_label"] == 0 for sample in samples),
            "replan": sum(sample["router_label"] == 1 for sample in samples),
        },
        "dataset_fingerprint": dataset_fingerprint,
        "sources": source_reports,
    }
    return samples, report


def _split_counts(size: int) -> tuple[int, int, int]:
    if size < 3:
        raise ValueError(f"Scene stratum is too small for 70/15/15 splitting: {size}")
    train = int(round(0.70 * size))
    val = max(1, int(round(0.15 * size)))
    test = size - train - val
    if test < 1:
        train -= 1 - test
        test = 1
    return train, val, test


def assign_scene_splits(
    samples: list[dict[str, Any]], seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    by_scene: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_scene.setdefault(int(sample["scene_seed"]), []).append(sample)

    strata: dict[str, list[int]] = {
        "has_rescue": [],
        "has_harm": [],
        "both_success_only": [],
    }
    for scene_seed, rows in by_scene.items():
        transitions = {row["transition"] for row in rows}
        if "rescue" in transitions:
            stratum = "has_rescue"
        elif "harm" in transitions:
            stratum = "has_harm"
        else:
            stratum = "both_success_only"
        strata[stratum].append(scene_seed)

    scene_to_split: dict[int, str] = {}
    stratum_report: dict[str, Any] = {}
    for stratum_index, (stratum, scene_seeds) in enumerate(strata.items()):
        scene_seeds = sorted(scene_seeds)
        rng = np.random.default_rng(seed + 10_007 * (stratum_index + 1))
        order = np.asarray(scene_seeds, dtype=np.int64)
        rng.shuffle(order)
        train_count, val_count, test_count = _split_counts(len(order))
        boundaries = (train_count, train_count + val_count)
        assignments = {
            "train": order[: boundaries[0]].tolist(),
            "val": order[boundaries[0] : boundaries[1]].tolist(),
            "test": order[boundaries[1] :].tolist(),
        }
        if len(assignments["test"]) != test_count:
            raise AssertionError("Split arithmetic error")
        for split_name, seeds in assignments.items():
            for scene_seed in seeds:
                if int(scene_seed) in scene_to_split:
                    raise AssertionError(f"Scene assigned twice: {scene_seed}")
                scene_to_split[int(scene_seed)] = split_name
        stratum_report[stratum] = {
            "scene_count": len(scene_seeds),
            "split_scene_counts": {
                name: len(assignments[name]) for name in SPLIT_NAMES
            },
            "split_scene_seeds": assignments,
        }

    split_ids = np.asarray(
        [SPLIT_TO_ID[scene_to_split[int(row["scene_seed"])]] for row in samples],
        dtype=np.int8,
    )
    split_report = {
        "method": "deterministic scene-grouped stratified 70/15/15",
        "seed": seed,
        "strata": stratum_report,
        "splits": {},
    }
    for split_name in SPLIT_NAMES:
        split_id = SPLIT_TO_ID[split_name]
        indices = np.flatnonzero(split_ids == split_id)
        split_samples = [samples[index] for index in indices]
        split_report["splits"][split_name] = {
            "sample_count": len(indices),
            "scene_count": len({row["scene_seed"] for row in split_samples}),
            "scene_seeds": sorted({row["scene_seed"] for row in split_samples}),
            "transition_counts": {
                transition: sum(
                    row["transition"] == transition for row in split_samples
                )
                for transition in ("rescue", "both_success", "harm")
            },
        }

    train_scenes = set(split_report["splits"]["train"]["scene_seeds"])
    val_scenes = set(split_report["splits"]["val"]["scene_seeds"])
    test_scenes = set(split_report["splits"]["test"]["scene_seeds"])
    if (
        train_scenes & val_scenes
        or train_scenes & test_scenes
        or val_scenes & test_scenes
    ):
        raise AssertionError("Scene leakage detected")
    if train_scenes | val_scenes | test_scenes != set(by_scene):
        raise AssertionError("Some scenes are absent from splits")
    return split_ids, split_report


def _stack_tree(rows: list[Any]) -> Any:
    first = rows[0]
    if isinstance(first, dict):
        if not all(set(row) == set(first) for row in rows):
            raise ValueError("Transformed observation keys differ within batch")
        return {key: _stack_tree([row[key] for row in rows]) for key in first}
    return np.stack([np.asarray(row) for row in rows], axis=0)


def _tree_to_torch(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, dict):
        return {key: _tree_to_torch(value, device) for key, value in tree.items()}
    array = np.asarray(tree)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array).to(device)


def load_frozen_policy(
    checkpoint_dir: Path,
    server_config: str,
    horizon: int,
    device: torch.device,
):
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
    except ImportError as exc:
        raise RuntimeError(
            "OpenPI imports are unavailable. Run this script with "
            "/home/ubuntu/Workspace/openpi/.venv/bin/python."
        ) from exc

    config = training_config.get_config(server_config)
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, action_horizon=horizon)
    )
    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        pytorch_device=str(device),
    )
    model = policy._model  # noqa: SLF001 - offline access to the frozen model is intentional.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Failed to freeze pi0.5")
    return policy, model


def extract_batch_features(
    policy,
    model,
    samples: list[dict[str, Any]],
    feature_type: str,
    device: torch.device,
) -> np.ndarray:
    try:
        from openpi.models import model as model_types
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    except ImportError as exc:
        raise RuntimeError("OpenPI model imports are unavailable") from exc

    transformed_rows = []
    for sample in samples:
        with np.load(sample["observation_path"], allow_pickle=False) as observation:
            raw = {
                "state": np.asarray(observation["state"], dtype=np.float32),
                "images": {
                    "cam_high": np.ascontiguousarray(
                        np.transpose(observation["head_camera_rgb"], (2, 0, 1))
                    ),
                    "cam_left_wrist": np.ascontiguousarray(
                        np.transpose(observation["left_camera_rgb"], (2, 0, 1))
                    ),
                    "cam_right_wrist": np.ascontiguousarray(
                        np.transpose(observation["right_camera_rgb"], (2, 0, 1))
                    ),
                },
                "prompt": sample["instruction"],
            }
        if feature_type == "action_expert_hidden_tail":
            raw["actions"] = np.asarray(
                sample["old_action_chunk"], dtype=np.float32
            )
        transformed_rows.append(policy._input_transform(raw))  # noqa: SLF001

    inputs = _tree_to_torch(_stack_tree(transformed_rows), device)
    observation = model_types.Observation.from_dict(inputs)
    images, image_masks, language_tokens, language_masks, state = (
        model._preprocess_observation(observation, train=False)  # noqa: SLF001
    )

    if feature_type == "vision_encoder":
        camera_features = []
        for image, image_mask in zip(images, image_masks, strict=True):
            tokens = model.paligemma_with_expert.embed_image(image)
            pooled = tokens.mean(dim=1)
            pooled = pooled * image_mask[:, None].to(pooled.dtype)
            camera_features.append(pooled)
        features = torch.cat(camera_features, dim=-1)
    elif feature_type == "vlm_hidden":
        prefix_embeddings, prefix_pad_masks, prefix_attention_masks = (
            model.embed_prefix(images, image_masks, language_tokens, language_masks)
        )
        if (
            model.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embeddings = prefix_embeddings.to(torch.bfloat16)
        attention_2d = make_att_2d_masks(prefix_pad_masks, prefix_attention_masks)
        attention_4d = model._prepare_attention_masks_4d(attention_2d)  # noqa: SLF001
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_hidden, _), _ = model.paligemma_with_expert.forward(
            attention_mask=attention_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=False,
        )
        language_length = int(language_masks.shape[1])
        language_hidden = prefix_hidden[:, -language_length:, :]
        language_mask = language_masks[:, :, None].to(language_hidden.dtype)
        features = (language_hidden * language_mask).sum(dim=1) / language_mask.sum(
            dim=1
        ).clamp_min(1.0)
    elif feature_type == "action_expert_hidden_tail":
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        normalized_actions = inputs["actions"].to(
            dtype=model.action_in_proj.weight.dtype
        )
        if normalized_actions.shape[1:] != (
            model.config.action_horizon,
            model.config.action_dim,
        ):
            raise ValueError(
                "Transformed old chunk does not match pi0.5 action space: "
                f"{normalized_actions.shape}"
            )
        prefix_embeddings, prefix_pad_masks, prefix_attention_masks = (
            model.embed_prefix(
                images, image_masks, language_tokens, language_masks
            )
        )
        flow_time = torch.zeros(
            len(samples), dtype=torch.float32, device=device
        )
        suffix_embeddings, suffix_pad_masks, suffix_attention_masks, adarms_cond = (
            model.embed_suffix(state, normalized_actions, flow_time)
        )
        if (
            model.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embeddings = prefix_embeddings.to(torch.bfloat16)
            suffix_embeddings = suffix_embeddings.to(torch.bfloat16)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        attention_masks = torch.cat(
            [prefix_attention_masks, suffix_attention_masks], dim=1
        )
        attention_2d = make_att_2d_masks(pad_masks, attention_masks)
        attention_4d = model._prepare_attention_masks_4d(  # noqa: SLF001
            attention_2d
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (hidden_outputs, _), _ = model.paligemma_with_expert.forward(
            attention_mask=attention_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, suffix_embeddings],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        action_hidden = hidden_outputs[
            :, -model.config.action_horizon :
        ].float()
        pooled_rows = []
        for row, sample in zip(action_hidden, samples, strict=True):
            cursor = int(sample["old_chunk_cursor"])
            if not 0 < cursor < model.config.action_horizon:
                raise ValueError(
                    f"Invalid Z1 old_chunk_cursor={cursor} for "
                    f"{sample['sample_id']}"
                )
            pooled_rows.append(row[cursor:].mean(dim=0))
        features = torch.stack(pooled_rows, dim=0)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    if features.ndim != 2 or features.shape[0] != len(samples):
        raise AssertionError(f"Unexpected frozen feature shape: {features.shape}")
    return features.float().cpu().numpy()


def extract_features(
    samples: list[dict[str, Any]],
    dataset_report: dict[str, Any],
    split_ids: np.ndarray,
    split_report: dict[str, Any],
    *,
    feature_type: str,
    output_dir: Path,
    checkpoint_dir: Path,
    server_config: str,
    horizon: int,
    action_dim: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Path, Path]:
    feature_path = output_dir / f"features_{feature_type}.npz"
    manifest_path = output_dir / f"features_{feature_type}_manifest.json"
    if feature_path.exists() or manifest_path.exists():
        if not feature_path.exists() or not manifest_path.exists():
            raise FileExistsError(
                f"Feature archive is incomplete; refusing to overwrite: {feature_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["dataset"]["dataset_fingerprint"]
            != dataset_report["dataset_fingerprint"]
            or manifest["feature_type"] != feature_type
            or manifest["checkpoint_dir"] != str(checkpoint_dir.resolve())
        ):
            raise FileExistsError(
                f"Existing feature archive has incompatible provenance: {feature_path}"
            )
        logging.info("Reusing immutable feature archive %s", feature_path)
        return feature_path, manifest_path

    policy, model = load_frozen_policy(checkpoint_dir, server_config, horizon, device)
    feature_batches = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            stop = min(start + batch_size, len(samples))
            logging.info(
                "Extracting %s features %d:%d/%d",
                feature_type,
                start,
                stop,
                len(samples),
            )
            feature_batches.append(
                extract_batch_features(
                    policy, model, samples[start:stop], feature_type, device
                )
            )
    visual_features = np.concatenate(feature_batches, axis=0).astype(np.float32)
    if len(visual_features) != len(samples):
        raise AssertionError("Feature/sample length mismatch")

    old_chunks = np.stack(
        [sample["old_action_chunk"] for sample in samples], axis=0
    ).astype(np.float32)
    if old_chunks.shape[1:] != (horizon, action_dim):
        raise AssertionError(f"Unexpected action chunk array: {old_chunks.shape}")

    arrays = {
        "zv": visual_features,
        "old_action_chunk": old_chunks,
        "old_chunk_cursor": np.asarray(
            [sample["old_chunk_cursor"] for sample in samples], dtype=np.int16
        ),
        "chunk_state": np.asarray(
            [
                [
                    sample["cursor_ratio"],
                    sample["remaining_ratio"],
                    sample["time_since_plan"],
                    sample["distance_to_next_r0"],
                    sample["last_replan_distance"],
                ]
                for sample in samples
            ],
            dtype=np.float32,
        ),
        "scene_seed": np.asarray(
            [sample["scene_seed"] for sample in samples], dtype=np.int64
        ),
        "timestep": np.asarray(
            [sample["timestep"] for sample in samples], dtype=np.int16
        ),
        "keep_success": np.asarray(
            [sample["keep_success"] for sample in samples], dtype=np.int8
        ),
        "replan_success": np.asarray(
            [sample["replan_success"] for sample in samples], dtype=np.int8
        ),
        "raw_delta": np.asarray(
            [sample["raw_delta"] for sample in samples], dtype=np.int8
        ),
        "router_label": np.asarray(
            [sample["router_label"] for sample in samples], dtype=np.int8
        ),
        "split": split_ids.astype(np.int8),
        "sample_id": np.asarray([sample["sample_id"] for sample in samples]),
        "transition": np.asarray([sample["transition"] for sample in samples]),
        "prompt_sha256": np.asarray([sample["prompt_sha256"] for sample in samples]),
        "observation_fingerprint": np.asarray(
            [sample["observation_fingerprint"] for sample in samples]
        ),
        "old_chunk_sha256": np.asarray(
            [sample["old_chunk_sha256"] for sample in samples]
        ),
    }
    save_npz_immutable(feature_path, arrays)

    model_path = checkpoint_dir / "model.safetensors"
    manifest = {
        "purpose": "Frozen pi0.5 offline replan-router features",
        "feature_type": feature_type,
        "feature_definition": (
            "Per-camera masked mean of projected SigLIP tokens, concatenated in "
            "cam_high/cam_left_wrist/cam_right_wrist order."
            if feature_type == "vision_encoder"
            else (
                "Masked mean of final contextual PaliGemma prefix hidden states "
                "at valid language-token positions."
                if feature_type == "vlm_hidden"
                else "Frozen pi0.5 action-expert final-layer action-token hidden "
                "states at flow time tau=0, mean-pooled over [old_chunk_cursor:H)."
            )
        ),
        "feature_shape": list(visual_features.shape),
        "feature_dtype": str(visual_features.dtype),
        "old_action_chunk_shape": list(old_chunks.shape),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_model_path": str(model_path.resolve()),
        "checkpoint_model_sha256": sha256_file(model_path),
        "server_config": server_config,
        "pi0_frozen": True,
        "torch_inference_mode": True,
        "deterministic_torch": True,
        "horizon": horizon,
        "action_dim": action_dim,
        "dataset": dataset_report,
        "split": split_report,
        "samples": [
            {
                key: sample[key]
                for key in (
                    "sample_id",
                    "task",
                    "scene_seed",
                    "timestep",
                    "instruction",
                    "prompt_sha256",
                    "observation_path",
                    "observation_fingerprint",
                    "trace_path",
                    "old_chunk_cursor",
                    "cursor_ratio",
                    "remaining_ratio",
                    "time_since_plan",
                    "distance_to_next_r0",
                    "last_replan_distance",
                    "natural_replan_interval",
                    "old_chunk_sha256",
                    "keep_success",
                    "replan_success",
                    "raw_delta",
                    "router_label",
                    "transition",
                    "source_eval_root",
                )
            }
            for sample in samples
        ],
    }
    write_json_immutable(manifest_path, manifest)
    del model, policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feature_path, manifest_path


class OutcomeRouter(nn.Module):
    """Two-layer outcome router with an optional two-layer action encoder."""

    def __init__(
        self,
        visual_dim: int,
        horizon: int,
        action_dim: int,
        router_input: str,
        *,
        hidden_dim: int = 128,
        action_hidden_dim: int = 128,
        action_embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.router_input = router_input
        self.action_encoder: nn.Module | None
        if router_input == "vision_action":
            self.action_encoder = nn.Sequential(
                nn.Linear(horizon * action_dim, action_hidden_dim),
                nn.GELU(),
                nn.Linear(action_hidden_dim, action_embedding_dim),
                nn.GELU(),
            )
            router_dim = visual_dim + action_embedding_dim
        elif router_input == "vision":
            self.action_encoder = None
            router_dim = visual_dim
        else:
            raise ValueError(f"Unknown router input: {router_input}")
        self.router = nn.Sequential(
            nn.Linear(router_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, visual: torch.Tensor, old_action_chunk: torch.Tensor
    ) -> torch.Tensor:
        if self.action_encoder is not None:
            action = self.action_encoder(old_action_chunk.flatten(start_dim=1))
            visual = torch.cat([visual, action], dim=-1)
        return self.router(visual)


class FeatureAblationRouter(nn.Module):
    """The fixed two-layer router with optional fixed-shape action encoding."""

    def __init__(
        self,
        direct_feature_dims: dict[str, int],
        *,
        action_mode: str,
        horizon: int,
        action_dim: int,
        action_feature_dim: int | None = None,
        hidden_dim: int = 128,
        action_hidden_dim: int = 128,
        action_embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.direct_feature_names = tuple(direct_feature_dims)
        self.direct_feature_encoders = nn.ModuleDict()
        self.action_mode = action_mode
        self.action_encoder: nn.Module | None = None
        router_dim = 0
        for name, dimension in direct_feature_dims.items():
            if name == "action_expert_hidden_tail":
                self.direct_feature_encoders[name] = nn.Sequential(
                    nn.Linear(dimension, action_embedding_dim),
                    nn.GELU(),
                )
                router_dim += action_embedding_dim
            else:
                router_dim += dimension
        if action_mode in ("future_action_tail", "full_action_chunk"):
            action_feature_dim = horizon * action_dim
        elif action_mode == "eef_trajectory":
            if action_feature_dim is None or action_feature_dim < 1:
                raise ValueError("EEF trajectory requires a positive feature dimension")
        if action_mode in (
            "future_action_tail",
            "full_action_chunk",
            "eef_trajectory",
        ):
            self.action_encoder = nn.Sequential(
                nn.Linear(action_feature_dim, action_hidden_dim),
                nn.GELU(),
                nn.Linear(action_hidden_dim, action_embedding_dim),
                nn.GELU(),
            )
            router_dim += action_embedding_dim
        elif action_mode not in ("none", "action_statistics"):
            raise ValueError(f"Unknown action mode: {action_mode}")
        self.router = nn.Sequential(
            nn.Linear(router_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = [
            (
                self.direct_feature_encoders[name](features[name])
                if name in self.direct_feature_encoders
                else features[name]
            )
            for name in self.direct_feature_names
        ]
        if self.action_encoder is not None:
            action = features[self.action_mode].flatten(start_dim=1)
            inputs.append(self.action_encoder(action))
        return self.router(torch.cat(inputs, dim=-1))


def compute_action_statistics(
    future_action: np.ndarray,
    *,
    action_dim: int,
    gripper_indices: tuple[int, ...],
) -> np.ndarray:
    """Encode per-dimension tail moments/dynamics plus signed gripper change."""
    if future_action.ndim != 2 or future_action.shape[1] != action_dim:
        raise ValueError(f"Unexpected future action shape: {future_action.shape}")
    if not len(future_action):
        raise ValueError("Future action tail must contain at least one action")
    velocity = np.diff(future_action, axis=0)
    acceleration = np.diff(future_action, n=2, axis=0)
    max_velocity = (
        np.max(np.abs(velocity), axis=0)
        if len(velocity)
        else np.zeros(action_dim, dtype=np.float32)
    )
    max_acceleration = (
        np.max(np.abs(acceleration), axis=0)
        if len(acceleration)
        else np.zeros(action_dim, dtype=np.float32)
    )
    gripper_change = (
        future_action[-1, list(gripper_indices)]
        - future_action[0, list(gripper_indices)]
    )
    return np.concatenate(
        [
            future_action.mean(axis=0),
            future_action.std(axis=0),
            max_velocity,
            max_acceleration,
            gripper_change,
        ]
    ).astype(np.float32)


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _relative_rotation_vector(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = previous.T @ current
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    skew_vector = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return 0.5 * skew_vector
    sine = math.sin(angle)
    if abs(sine) < 1e-7:
        # Consecutive VLA targets are not expected to jump by pi, but keep a
        # deterministic eigenvector fallback for malformed chunks.
        eigenvalues, eigenvectors = np.linalg.eig(relative)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= max(np.linalg.norm(axis), 1e-12)
        return axis * angle
    return skew_vector * (angle / (2.0 * sine))


class AlohaOfflineForwardKinematics:
    """Exact SAPIEN FK using the same fixed ALOHA URDF as RoboTwin."""

    ARM_JOINT_NAMES = (
        tuple(f"fl_joint{index}" for index in range(1, 7)),
        tuple(f"fr_joint{index}" for index in range(1, 7)),
    )
    EEF_JOINT_NAMES = ("fl_joint6", "fr_joint6")
    ROOT_POSITION = (0.0, -0.65, 0.0)
    ROOT_QUATERNION_WXYZ = (0.707, 0.0, 0.0, 0.707)
    GLOBAL_TRANSFORM = np.diag([1.0, -1.0, -1.0])

    def __init__(self, urdf_path: Path):
        if not urdf_path.exists():
            raise FileNotFoundError(f"ALOHA URDF not found: {urdf_path}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import sapien
        except ImportError as exc:
            raise RuntimeError(
                "EEF action modeling requires SAPIEN in the offline Python environment."
            ) from exc

        self.urdf_path = urdf_path.resolve()
        self._sapien = sapien
        self._scene = sapien.Scene()
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robot = loader.load(str(self.urdf_path))
        if self._robot is None:
            raise RuntimeError(f"Failed to load ALOHA URDF: {self.urdf_path}")
        self._robot.set_root_pose(
            sapien.Pose(self.ROOT_POSITION, self.ROOT_QUATERNION_WXYZ)
        )
        self._scene.step()

        qpos_offsets = {}
        offset = 0
        for joint in self._robot.get_active_joints():
            if int(joint.dof) != 1:
                raise ValueError(f"Unexpected joint DOF for {joint.name}: {joint.dof}")
            qpos_offsets[joint.name] = offset
            offset += int(joint.dof)
        self._arm_qpos_indices = np.asarray(
            [[qpos_offsets[name] for name in names] for names in self.ARM_JOINT_NAMES],
            dtype=np.int64,
        )
        joints = {joint.name: joint for joint in self._robot.get_joints()}
        self._eef_joints = tuple(joints[name] for name in self.EEF_JOINT_NAMES)
        self._base_qpos = self._robot.get_qpos().copy()

    def dual_eef_trajectory(
        self, dual_arm_qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dual_arm_qpos = np.asarray(dual_arm_qpos, dtype=np.float64)
        if dual_arm_qpos.ndim != 3 or dual_arm_qpos.shape[1:] != (2, 6):
            raise ValueError(
                f"Expected dual-arm qpos [T,2,6], got {dual_arm_qpos.shape}"
            )
        positions = np.empty((len(dual_arm_qpos), 2, 3), dtype=np.float64)
        rotations = np.empty((len(dual_arm_qpos), 2, 3, 3), dtype=np.float64)
        qpos = self._base_qpos.copy()
        for timestep, arm_targets in enumerate(dual_arm_qpos):
            for arm_index in range(2):
                qpos[self._arm_qpos_indices[arm_index]] = arm_targets[arm_index]
            self._robot.set_qpos(qpos)
            self._scene.step()
            for arm_index, joint in enumerate(self._eef_joints):
                pose = joint.global_pose
                positions[timestep, arm_index] = np.asarray(pose.p, dtype=np.float64)
                rotations[timestep, arm_index] = (
                    _quaternion_wxyz_to_matrix(np.asarray(pose.q, dtype=np.float64))
                    @ self.GLOBAL_TRANSFORM
                )
        return positions, rotations


def build_eef_action_descriptor(
    future_action: np.ndarray,
    current_state: np.ndarray,
    *,
    forward_kinematics: AlohaOfflineForwardKinematics,
    waypoint_count: int,
) -> tuple[np.ndarray, float]:
    """Delta/integrate qpos, run FK, and sample EEF waypoints/velocities."""
    future_action = np.asarray(future_action, dtype=np.float64)
    current_state = np.asarray(current_state, dtype=np.float64)
    if future_action.ndim != 2 or future_action.shape[1] != 14:
        raise ValueError(
            f"Expected ALOHA future action [T,14], got {future_action.shape}"
        )
    if current_state.shape != (14,):
        raise ValueError(
            f"Expected current ALOHA state [14], got {current_state.shape}"
        )
    if waypoint_count < 2 or len(future_action) < waypoint_count:
        raise ValueError(
            f"Need at least {waypoint_count} future actions, got {len(future_action)}"
        )

    action_with_origin = np.concatenate([current_state[None, :], future_action], axis=0)
    delta_action = np.diff(action_with_origin, axis=0)
    integrated_action = current_state[None, :] + np.cumsum(delta_action, axis=0)
    reconstruction_error = float(np.max(np.abs(integrated_action - future_action)))
    if reconstruction_error > 1e-6:
        raise AssertionError(
            f"Delta-action integration drifted by {reconstruction_error}"
        )

    arm_indices = np.asarray(
        [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]],
        dtype=np.int64,
    )
    joint_trajectory = np.empty((len(integrated_action) + 1, 2, 6), dtype=np.float64)
    for arm_index in range(2):
        joint_trajectory[0, arm_index] = current_state[arm_indices[arm_index]]
        joint_trajectory[1:, arm_index] = integrated_action[:, arm_indices[arm_index]]
    positions, rotations = forward_kinematics.dual_eef_trajectory(joint_trajectory)

    future_positions = positions[1:]
    future_rotations = rotations[1:]
    rotation_6d = np.concatenate(
        [future_rotations[..., :, 0], future_rotations[..., :, 1]], axis=-1
    )
    gripper_indices = np.asarray([6, 13], dtype=np.int64)
    waypoint_frames = np.concatenate(
        [
            future_positions[:, 0],
            rotation_6d[:, 0],
            integrated_action[:, gripper_indices[0:1]],
            future_positions[:, 1],
            rotation_6d[:, 1],
            integrated_action[:, gripper_indices[1:2]],
        ],
        axis=-1,
    )

    linear_velocity = np.diff(positions, axis=0)
    angular_velocity = np.empty_like(linear_velocity)
    for timestep in range(len(future_action)):
        for arm_index in range(2):
            angular_velocity[timestep, arm_index] = _relative_rotation_vector(
                rotations[timestep, arm_index],
                rotations[timestep + 1, arm_index],
            )
    gripper_velocity = np.diff(action_with_origin[:, gripper_indices], axis=0)
    velocity_frames = np.concatenate(
        [
            linear_velocity[:, 0],
            angular_velocity[:, 0],
            np.linalg.norm(linear_velocity[:, 0], axis=-1, keepdims=True),
            np.linalg.norm(angular_velocity[:, 0], axis=-1, keepdims=True),
            linear_velocity[:, 1],
            angular_velocity[:, 1],
            np.linalg.norm(linear_velocity[:, 1], axis=-1, keepdims=True),
            np.linalg.norm(angular_velocity[:, 1], axis=-1, keepdims=True),
            gripper_velocity,
        ],
        axis=-1,
    )

    waypoint_indices = np.rint(
        np.linspace(0, len(future_action) - 1, waypoint_count)
    ).astype(np.int64)
    descriptor = np.concatenate(
        [
            waypoint_frames[waypoint_indices].reshape(-1),
            velocity_frames[waypoint_indices].reshape(-1),
        ]
    ).astype(np.float32)
    expected_dimension = waypoint_count * (20 + 18)
    if descriptor.shape != (expected_dimension,):
        raise AssertionError(f"Unexpected EEF descriptor shape: {descriptor.shape}")
    return descriptor, reconstruction_error


def _standardize_matrix(
    values: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return (values - mean) / std, mean, std


def prepare_ablation_features(
    feature_path: Path,
    samples: list[dict[str, Any]],
    config: FeatureConfig,
    *,
    temporal_k: int,
    gripper_indices: tuple[int, ...],
    eef_urdf_path: Path,
    eef_waypoints: int,
) -> dict[str, Any]:
    with np.load(feature_path, allow_pickle=False) as archive:
        visual = np.asarray(archive["zv"], dtype=np.float32)
        old_chunks = np.asarray(archive["old_action_chunk"], dtype=np.float32)
        split_ids = np.asarray(archive["split"], dtype=np.int8)
        archive_sample_ids = np.asarray(archive["sample_id"]).astype(str)
        keep_success = np.asarray(archive["keep_success"], dtype=np.int64)
        replan_success = np.asarray(archive["replan_success"], dtype=np.int64)
        scene_seeds = np.asarray(archive["scene_seed"], dtype=np.int64)
        timesteps = np.asarray(archive["timestep"], dtype=np.int64)
        action_hidden_tail = (
            np.asarray(archive["action_expert_hidden_tail"], dtype=np.float32)
            if config.action_hidden_tail
            and "action_expert_hidden_tail" in archive.files
            else None
        )

    sample_ids = np.asarray([sample["sample_id"] for sample in samples])
    if len(samples) != len(visual) or not np.array_equal(
        archive_sample_ids, sample_ids
    ):
        raise ValueError("Feature archive order does not match the current dataset")
    horizon, action_dim = old_chunks.shape[1:]
    if any(index < 0 or index >= action_dim for index in gripper_indices):
        raise ValueError(
            f"Gripper indices {gripper_indices} are invalid for action_dim={action_dim}"
        )

    valid_mask = np.ones(len(samples), dtype=bool)
    history_indices = np.full(len(samples), -1, dtype=np.int64)
    if config.delta_vision:
        key_to_index = {
            (int(scene_seed), int(timestep)): index
            for index, (scene_seed, timestep) in enumerate(
                zip(scene_seeds, timesteps, strict=True)
            )
        }
        for index, (scene_seed, timestep) in enumerate(
            zip(scene_seeds, timesteps, strict=True)
        ):
            history_indices[index] = key_to_index.get(
                (int(scene_seed), int(timestep) - temporal_k), -1
            )
        valid_mask = history_indices >= 0

    eligible_indices = np.flatnonzero(valid_mask)
    train_indices = np.flatnonzero(valid_mask & (split_ids == SPLIT_TO_ID["train"]))
    if not len(train_indices):
        raise ValueError(f"No eligible training samples for {config.name}")

    raw_components: dict[str, np.ndarray] = {"vision": visual}
    definitions: dict[str, Any] = {
        "vision": (
            "Per-camera frozen pi0.5 vision-encoder token mean, concatenated "
            "in high/left-wrist/right-wrist order."
        )
    }
    if config.action_hidden_tail:
        if action_hidden_tail is None:
            raise ValueError(
                f"{config.name} requires action_expert_hidden_tail in "
                f"{feature_path}"
            )
        if action_hidden_tail.shape[0] != len(samples):
            raise ValueError(
                "Action-hidden feature/sample length mismatch: "
                f"{action_hidden_tail.shape}"
            )
        raw_components["action_expert_hidden_tail"] = action_hidden_tail
        definitions["action_expert_hidden_tail"] = {
            "model_component": "frozen pi0.5 Gemma action expert",
            "layer": "final transformer layer after final RMS normalization",
            "flow_time_tau": 0.0,
            "action_input": (
                "current old_action_chunk transformed back into pi0.5 "
                "normalized model action space using the current observation state"
            ),
            "token_range": "[old_chunk_cursor:H)",
            "pooling": "mean over remaining action-token hidden states",
            "encoding": (
                "Train-standardized pooled hidden -> Linear(D,64) -> GELU, "
                "then concatenate with the vision feature before the MLP router"
            ),
            "online_action_sampling": False,
        }
    if config.chunk_state:
        raw_components["chunk_state"] = np.asarray(
            [
                [
                    sample["cursor_ratio"],
                    sample["remaining_ratio"],
                    sample["time_since_plan"],
                    sample["distance_to_next_r0"],
                    sample["last_replan_distance"],
                ]
                for sample in samples
            ],
            dtype=np.float32,
        )
        definitions["chunk_state"] = {
            "fields": [
                "cursor_ratio=old_chunk_cursor/H",
                "remaining_ratio=(H-old_chunk_cursor)/H",
                "time_since_plan=old_chunk_cursor action steps",
                "distance_to_next_r0=absolute_r0_next_boundary-timestep",
                "last_replan_distance=timestep-last_prior_router_clock; "
                "episode origin is used when the strict-prefix trace has no "
                "prior router trigger",
            ],
            "dimension": 5,
        }
    if config.delta_vision:
        delta = np.zeros_like(visual)
        delta[valid_mask] = visual[valid_mask] - visual[history_indices[valid_mask]]
        raw_components["delta_vision"] = delta
        definitions["delta_vision"] = (
            f"z_visual(t)-z_visual(t-{temporal_k}); samples without an exact "
            "same-scene archived history observation are excluded."
        )

    future_tails = [
        old_chunks[index, int(samples[index]["old_chunk_cursor"]) :, :]
        for index in range(len(samples))
    ]
    if any(len(tail) < 1 for tail in future_tails):
        raise ValueError("A future action tail is empty")
    if config.action_mode == "action_statistics":
        raw_components["action_statistics"] = np.stack(
            [
                compute_action_statistics(
                    tail,
                    action_dim=action_dim,
                    gripper_indices=gripper_indices,
                )
                for tail in future_tails
            ],
            axis=0,
        )
        definitions["action_statistics"] = {
            "source": "future actions old_chunk[old_chunk_cursor:H]",
            "fields": [
                f"mean[{action_dim}]",
                f"std[{action_dim}]",
                f"max_abs_velocity[{action_dim}]",
                f"max_abs_acceleration[{action_dim}]",
                f"signed_gripper_change[{len(gripper_indices)}]",
            ],
            "gripper_indices": list(gripper_indices),
            "dimension": int(raw_components["action_statistics"].shape[1]),
        }
    elif config.action_mode == "eef_trajectory":
        if action_dim != 14:
            raise ValueError(
                "EEF trajectory modeling is defined for the ALOHA 14-D "
                f"action layout, got action_dim={action_dim}"
            )
        forward_kinematics = AlohaOfflineForwardKinematics(eef_urdf_path)
        descriptors = []
        integration_errors = []
        for index, tail in enumerate(future_tails):
            with np.load(
                samples[index]["observation_path"], allow_pickle=False
            ) as observation:
                current_state = np.asarray(observation["state"], dtype=np.float32)
            descriptor, integration_error = build_eef_action_descriptor(
                tail,
                current_state,
                forward_kinematics=forward_kinematics,
                waypoint_count=eef_waypoints,
            )
            descriptors.append(descriptor)
            integration_errors.append(integration_error)
        raw_components["eef_trajectory"] = np.stack(descriptors, axis=0)
        definitions["eef_trajectory"] = {
            "source": "future actions old_chunk[old_chunk_cursor:H]",
            "action_layout": (
                "[left_joint_qpos[6], left_gripper, right_joint_qpos[6], right_gripper]"
            ),
            "delta_action": (
                "First delta is future_action[0]-decision_observation_state; "
                "later deltas are consecutive future-target differences."
            ),
            "integration": (
                "decision_observation_state+cumsum(delta_action), with exact "
                "reconstruction asserted before FK."
            ),
            "forward_kinematics": {
                "engine": "SAPIEN offline articulation FK",
                "urdf": str(eef_urdf_path.resolve()),
                "urdf_sha256": sha256_file(eef_urdf_path),
                "eef_joints": list(AlohaOfflineForwardKinematics.EEF_JOINT_NAMES),
                "root_position": list(AlohaOfflineForwardKinematics.ROOT_POSITION),
                "root_quaternion_wxyz": list(
                    AlohaOfflineForwardKinematics.ROOT_QUATERNION_WXYZ
                ),
                "orientation": (
                    "R_joint6 @ diag(1,-1,-1), matching RoboTwin "
                    "Robot.get_{left,right}_ee_pose"
                ),
            },
            "waypoints": {
                "count": eef_waypoints,
                "selection": (
                    "nearest integer indices at uniform normalized times "
                    "from first to last future target"
                ),
                "per_waypoint": (
                    "left xyz + rotation-6D + gripper, then right xyz + "
                    "rotation-6D + gripper (20 values)"
                ),
            },
            "velocity_profile": {
                "count": eef_waypoints,
                "per_waypoint": (
                    "per arm signed xyz velocity, signed rotation-vector "
                    "velocity, linear/angular speed; plus two signed gripper "
                    "velocities (18 values)"
                ),
                "first_transition": (
                    "decision observation EEF/state to the first future target"
                ),
            },
            "descriptor_dimension": int(raw_components["eef_trajectory"].shape[1]),
            "encoding": (
                "Train-standardized descriptor -> Linear(D,128) -> GELU -> "
                "Linear(128,64) -> GELU."
            ),
            "max_integration_reconstruction_error": float(max(integration_errors)),
        }

    normalized: dict[str, np.ndarray] = {}
    normalization: dict[str, dict[str, np.ndarray]] = {}
    for name, values in raw_components.items():
        standardized, mean, std = _standardize_matrix(values, train_indices)
        normalized[name] = np.ascontiguousarray(standardized, dtype=np.float32)
        normalization[name] = {"mean": mean, "std": std}

    if config.action_mode == "future_action_tail":
        train_action_rows = np.concatenate(
            [future_tails[index] for index in train_indices], axis=0
        )
        action_mean = train_action_rows.mean(axis=0, dtype=np.float64).astype(
            np.float32
        )
        action_std = train_action_rows.std(axis=0, dtype=np.float64).astype(np.float32)
        action_std[action_std < 1e-6] = 1.0
        padded = np.zeros_like(old_chunks)
        for index, tail in enumerate(future_tails):
            padded[index, : len(tail)] = (tail - action_mean) / action_std
        normalized["future_action_tail"] = padded
        normalization["future_action_tail"] = {
            "mean": action_mean,
            "std": action_std,
        }
        definitions["future_action_tail"] = {
            "source": "old_chunk[old_chunk_cursor:H]",
            "encoding": (
                "Valid future actions are train-standardized, left-aligned in "
                "[H,action_dim], zero-padded, flattened, then passed through "
                "the fixed two-layer action MLP."
            ),
        }
    elif config.action_mode == "full_action_chunk":
        flattened_train = old_chunks[train_indices].reshape(-1, action_dim)
        action_mean = flattened_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        action_std = flattened_train.std(axis=0, dtype=np.float64).astype(np.float32)
        action_std[action_std < 1e-6] = 1.0
        normalized["full_action_chunk"] = np.ascontiguousarray(
            (old_chunks - action_mean[None, None, :]) / action_std[None, None, :],
            dtype=np.float32,
        )
        normalization["full_action_chunk"] = {
            "mean": action_mean,
            "std": action_std,
        }
        definitions["full_action_chunk"] = {
            "source": "complete old_chunk[0:H]",
            "encoding": "Train-standardized, flattened, two-layer action MLP.",
        }

    return {
        "features": {
            name: torch.from_numpy(values) for name, values in normalized.items()
        },
        "normalization": normalization,
        "definitions": definitions,
        "eligible_indices": eligible_indices,
        "valid_mask": valid_mask,
        "history_indices": history_indices,
        "split_ids": split_ids,
        "keep_success": keep_success,
        "replan_success": replan_success,
        "scene_seeds": scene_seeds,
        "timesteps": timesteps,
        "sample_ids": sample_ids,
        "horizon": horizon,
        "action_dim": action_dim,
    }


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_predictions(
    keep_success: np.ndarray,
    replan_success: np.ndarray,
    p_keep: np.ndarray,
    p_replan: np.ndarray,
    decision_lambda: float,
    decision_temperature: float,
) -> dict[str, Any]:
    strong_replan = (keep_success == 0) & (replan_success == 1)
    advantage = p_replan - p_keep
    predicted_replan = advantage > decision_lambda
    decision_probability = 1.0 / (
        1.0 + np.exp(-(advantage - decision_lambda) / decision_temperature)
    )

    tp = int(np.sum(predicted_replan & strong_replan))
    tn = int(np.sum(~predicted_replan & ~strong_replan))
    fp = int(np.sum(predicted_replan & ~strong_replan))
    fn = int(np.sum(~predicted_replan & strong_replan))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    selected_success = np.where(predicted_replan, replan_success, keep_success).astype(
        np.float64
    )
    selected_utility = (
        selected_success - predicted_replan.astype(np.float64) * decision_lambda
    )

    transitions = np.full(len(keep_success), "both_success", dtype="<U16")
    transitions[(keep_success == 0) & (replan_success == 1)] = "rescue"
    transitions[(keep_success == 1) & (replan_success == 0)] = "harm"
    by_transition = {}
    for transition in ("rescue", "both_success", "harm"):
        mask = transitions == transition
        by_transition[transition] = {
            "count": int(mask.sum()),
            "predicted_replan": int(predicted_replan[mask].sum()),
            "predicted_replan_rate": (
                float(predicted_replan[mask].mean()) if mask.any() else None
            ),
            "mean_advantage_score": (
                float(advantage[mask].mean()) if mask.any() else None
            ),
        }

    always_keep_utility = float(np.mean(keep_success))
    always_replan_utility = float(np.mean(replan_success) - decision_lambda)
    oracle_replan = (replan_success - keep_success) > decision_lambda
    oracle_utility = float(
        np.mean(
            np.where(oracle_replan, replan_success, keep_success)
            - oracle_replan.astype(np.float64) * decision_lambda
        )
    )
    return {
        "sample_count": len(keep_success),
        "lambda": decision_lambda,
        "roc_auc": binary_roc_auc(strong_replan, advantage),
        "accuracy": safe_divide(tp + tn, len(keep_success)),
        "f1": f1,
        "precision_replan": precision,
        "recall_replan": recall,
        "strong_replan_recall": recall,
        "false_trigger_rate": safe_divide(fp, fp + tn),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "mean_decision_probability": float(decision_probability.mean()),
        "mean_advantage_score": float(advantage.mean()),
        "counterfactual_success_rate": float(selected_success.mean()),
        "counterfactual_utility": float(selected_utility.mean()),
        "baselines": {
            "always_keep_utility": always_keep_utility,
            "always_replan_utility": always_replan_utility,
            "oracle_utility": oracle_utility,
        },
        "outcome_heads": {
            "keep_success_roc_auc": binary_roc_auc(keep_success, p_keep),
            "replan_success_roc_auc": binary_roc_auc(replan_success, p_replan),
            "keep_success_accuracy_at_0_5": float(
                np.mean((p_keep > 0.5) == keep_success)
            ),
            "replan_success_accuracy_at_0_5": float(
                np.mean((p_replan > 0.5) == replan_success)
            ),
        },
        "by_transition": by_transition,
    }


def predict(
    model: OutcomeRouter,
    visual: torch.Tensor,
    action: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    keep_probabilities = []
    replan_probabilities = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            logits = model(
                visual[batch_indices].to(device),
                action[batch_indices].to(device),
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()
            keep_probabilities.append(probabilities[:, 0])
            replan_probabilities.append(probabilities[:, 1])
    return (
        np.concatenate(keep_probabilities),
        np.concatenate(replan_probabilities),
    )


def predict_feature_ablation(
    model: FeatureAblationRouter,
    features: dict[str, torch.Tensor],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    keep_probabilities = []
    replan_probabilities = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = {
                name: values[batch_indices].to(device)
                for name, values in features.items()
            }
            probabilities = torch.sigmoid(model(batch)).cpu().numpy()
            keep_probabilities.append(probabilities[:, 0])
            replan_probabilities.append(probabilities[:, 1])
    return (
        np.concatenate(keep_probabilities),
        np.concatenate(replan_probabilities),
    )


def _numpy_tree_to_torch(tree: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    return {
        name: {
            statistic: torch.from_numpy(value)
            for statistic, value in statistics.items()
        }
        for name, statistics in tree.items()
    }


def train_feature_ablation_config(
    prepared: dict[str, Any],
    *,
    config: FeatureConfig,
    feature_path: Path,
    source_manifest_path: Path,
    dataset_report: dict[str, Any],
    split_report: dict[str, Any],
    output_dir: Path,
    temporal_k: int,
    decision_lambda: float,
    decision_temperature: float,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[Path, Path, Path]:
    config_path = output_dir / "configs" / f"{config.name}_feature_manifest.json"
    checkpoint_path = output_dir / "checkpoints" / f"{config.name}.pt"
    evaluation_path = output_dir / "evaluations" / f"{config.name}.json"
    log_path = output_dir / "logs" / f"{config.name}_training_log.json"
    existing = [
        path for path in (checkpoint_path, evaluation_path, log_path) if path.exists()
    ]
    if existing:
        if len(existing) != 3 or not config_path.exists():
            raise FileExistsError(
                f"Incomplete immutable run for {config.name}: {existing}"
            )
        evaluation = json.loads(evaluation_path.read_text())
        existing_feature_manifest = json.loads(config_path.read_text())
        if (
            evaluation.get("feature_config") != config.name
            or evaluation["dataset"]["dataset_fingerprint"]
            != dataset_report["dataset_fingerprint"]
            or existing_feature_manifest.get("definitions") != prepared["definitions"]
            or int(evaluation["seed"]) != seed
            or float(evaluation["lambda"]) != decision_lambda
            or float(evaluation["decision_temperature"]) != decision_temperature
            or evaluation["optimization"]
            != {
                "loss": ("unweighted BCEWithLogits over keep/replan outcome heads"),
                "optimizer": "AdamW",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_epochs": epochs,
                "early_stopping_patience": patience,
                "selection": ("validation router ROC-AUC, then validation utility"),
            }
        ):
            raise FileExistsError(
                f"Existing {config.name} run has incompatible provenance"
            )
        logging.info("Reusing completed immutable run %s", config.name)
        return checkpoint_path, evaluation_path, config_path

    features: dict[str, torch.Tensor] = prepared["features"]
    split_ids = prepared["split_ids"]
    valid_mask = prepared["valid_mask"]
    keep_success = prepared["keep_success"]
    replan_success = prepared["replan_success"]
    scene_seeds = prepared["scene_seeds"]
    timesteps = prepared["timesteps"]
    sample_ids = prepared["sample_ids"]
    history_indices = prepared["history_indices"]
    split_indices = {
        name: np.flatnonzero(valid_mask & (split_ids == SPLIT_TO_ID[name]))
        for name in SPLIT_NAMES
    }
    if not all(len(indices) for indices in split_indices.values()):
        raise ValueError(f"An eligible split is empty for {config.name}")
    for split_name, indices in split_indices.items():
        labels = (keep_success[indices] == 0) & (replan_success[indices] == 1)
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{config.name} {split_name} lacks both decision classes")

    action_tensor_names = {
        "future_action_tail",
        "full_action_chunk",
        "eef_trajectory",
    }
    direct_feature_dims = {
        name: int(values.shape[1])
        for name, values in features.items()
        if name not in action_tensor_names
    }
    model = FeatureAblationRouter(
        direct_feature_dims,
        action_mode=config.action_mode,
        horizon=prepared["horizon"],
        action_dim=prepared["action_dim"],
        action_feature_dim=(
            int(features[config.action_mode][0].numel())
            if config.action_mode in action_tensor_names
            else None
        ),
    ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    eligible_by_split = {}
    for split_name, indices in split_indices.items():
        eligible_by_split[split_name] = {
            "sample_count": int(len(indices)),
            "scene_count": int(len(set(scene_seeds[indices].tolist()))),
            "strong_replan_count": int(
                np.sum((keep_success[indices] == 0) & (replan_success[indices] == 1))
            ),
            "sample_ids": sample_ids[indices].tolist(),
        }
    source_manifest = json.loads(source_manifest_path.read_text())
    split_identity = sha256_json(split_report)
    feature_manifest = {
        "purpose": "Offline replan-router feature ablation input manifest",
        "feature_config": config.name,
        "modules": list(config.modules),
        "definitions": prepared["definitions"],
        "dataset_fingerprint": dataset_report["dataset_fingerprint"],
        "split_identity_sha256": split_identity,
        "split": split_report,
        "eligible_by_split": eligible_by_split,
        "sample_count_before_temporal_filter": int(len(valid_mask)),
        "sample_count_after_temporal_filter": int(valid_mask.sum()),
        "temporal_history": {
            "k_action_steps": temporal_k,
            "missing_history_policy": "skip_sample",
            "eligible_history_pairs": [
                {
                    "sample_id": str(sample_ids[index]),
                    "history_sample_id": str(sample_ids[history_indices[index]]),
                }
                for index in prepared["eligible_indices"]
                if config.delta_vision
            ],
        },
        "source_feature_archive": str(feature_path.resolve()),
        "source_feature_archive_sha256": sha256_file(feature_path),
        "source_feature_manifest": str(source_manifest_path.resolve()),
        "source_feature_manifest_sha256": sha256_file(source_manifest_path),
        "source_feature_type": source_manifest["feature_type"],
        "pi0_frozen": True,
        "online_control_enabled": False,
        "normalization_fit_split": "train",
    }
    write_json_immutable(config_path, feature_manifest)

    outcomes = torch.from_numpy(
        np.stack([keep_success, replan_success], axis=1).astype(np.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    best_state = None
    best_epoch = None
    best_val_auc = -math.inf
    best_val_utility = -math.inf
    stale_epochs = 0
    history = []
    train_indices = split_indices["train"]

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=generator).numpy()
        ]
        epoch_losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch = {
                name: values[indices].to(device) for name, values in features.items()
            }
            logits = model(batch)
            target = outcomes[indices].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_indices = split_indices["val"]
        val_p_keep, val_p_replan = predict_feature_ablation(
            model, features, val_indices, device, batch_size
        )
        val_metrics = evaluate_predictions(
            keep_success[val_indices],
            replan_success[val_indices],
            val_p_keep,
            val_p_replan,
            decision_lambda,
            decision_temperature,
        )
        val_auc = (
            float(val_metrics["roc_auc"])
            if val_metrics["roc_auc"] is not None
            else -math.inf
        )
        val_utility = float(val_metrics["counterfactual_utility"])
        improved = val_auc > best_val_auc + 1e-12 or (
            abs(val_auc - best_val_auc) <= 1e-12
            and val_utility > best_val_utility + 1e-12
        )
        if improved:
            best_val_auc = val_auc
            best_val_utility = val_utility
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_bce": float(np.mean(epoch_losses)),
                "val_roc_auc": val_metrics["roc_auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_strong_replan_recall": val_metrics["strong_replan_recall"],
                "val_false_trigger_rate": val_metrics["false_trigger_rate"],
                "val_counterfactual_utility": val_utility,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or improved:
            logging.info(
                "%s epoch=%d loss=%.5f val_auc=%s val_f1=%.3f",
                config.name,
                epoch,
                history[-1]["train_bce"],
                (
                    f"{val_metrics['roc_auc']:.4f}"
                    if val_metrics["roc_auc"] is not None
                    else "n/a"
                ),
                val_metrics["f1"],
            )
        if stale_epochs >= patience:
            logging.info("%s early stopping at epoch %d", config.name, epoch)
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError(f"Training did not produce a checkpoint for {config.name}")
    model.load_state_dict(best_state)

    metrics_by_split = {}
    predictions_by_split = {}
    threshold_sweep = {}
    for split_name, indices in split_indices.items():
        p_keep, p_replan = predict_feature_ablation(
            model, features, indices, device, batch_size
        )
        metrics_by_split[split_name] = evaluate_predictions(
            keep_success[indices],
            replan_success[indices],
            p_keep,
            p_replan,
            decision_lambda,
            decision_temperature,
        )
        threshold_sweep[split_name] = [
            {
                "threshold": threshold,
                **evaluate_predictions(
                    keep_success[indices],
                    replan_success[indices],
                    p_keep,
                    p_replan,
                    threshold,
                    decision_temperature,
                ),
            }
            for threshold in THRESHOLD_SWEEP
        ]
        advantage = p_replan - p_keep
        predictions_by_split[split_name] = [
            {
                "sample_id": str(sample_ids[index]),
                "scene_seed": int(scene_seeds[index]),
                "timestep": int(timesteps[index]),
                "keep_success": int(keep_success[index]),
                "replan_success": int(replan_success[index]),
                "p_keep": float(p_keep[offset]),
                "p_replan": float(p_replan[offset]),
                "advantage": float(advantage[offset]),
                "predicted_replan": bool(advantage[offset] > decision_lambda),
            }
            for offset, index in enumerate(indices)
        ]

    checkpoint_payload = {
        "model_state_dict": best_state,
        "architecture": {
            "class": "FeatureAblationRouter",
            "direct_feature_dims": direct_feature_dims,
            "horizon": int(prepared["horizon"]),
            "action_dim": int(prepared["action_dim"]),
            "action_mode": config.action_mode,
            "action_feature_dim": (
                int(features[config.action_mode][0].numel())
                if config.action_mode in action_tensor_names
                else None
            ),
            "hidden_dim": 128,
            "action_hidden_dim": 128,
            "action_embedding_dim": 64,
            "dropout": 0.1,
            "output_logits": ["keep_success", "replan_success"],
            "decision": "sigmoid(replan_logit)-sigmoid(keep_logit) > lambda",
            "trainable_parameters": trainable_parameters,
        },
        "normalization": _numpy_tree_to_torch(prepared["normalization"]),
        "feature_config": config.name,
        "feature_manifest": str(config_path.resolve()),
        "feature_manifest_sha256": sha256_file(config_path),
        "dataset_fingerprint": dataset_report["dataset_fingerprint"],
        "split_identity_sha256": split_identity,
        "lambda": decision_lambda,
        "best_epoch": best_epoch,
        "seed": seed,
    }
    save_torch_immutable(checkpoint_path, checkpoint_payload)
    write_json_immutable(
        log_path,
        {
            "feature_config": config.name,
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "history": history,
        },
    )

    evaluation = {
        "purpose": "Offline replan-router feature ablation",
        "task": dataset_report["task"],
        "feature_config": config.name,
        "feature_modules": list(config.modules),
        "seed": seed,
        "lambda": decision_lambda,
        "decision_temperature": decision_temperature,
        "trigger_score": "p_replan-p_keep",
        "both_failure_excluded": True,
        "pi0_frozen": True,
        "online_control_enabled": False,
        "trainable_parameters": trainable_parameters,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "optimization": {
            "loss": "unweighted BCEWithLogits over keep/replan outcome heads",
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "early_stopping_patience": patience,
            "selection": "validation router ROC-AUC, then validation utility",
        },
        "dataset": dataset_report,
        "split": split_report,
        "split_identity_sha256": split_identity,
        "eligible_by_split": eligible_by_split,
        "metrics": metrics_by_split,
        "threshold_sweep": threshold_sweep,
        "test_predictions": predictions_by_split["test"],
        "artifacts": {
            "feature_manifest": str(config_path.resolve()),
            "router_checkpoint": str(checkpoint_path.resolve()),
            "training_log": str(log_path.resolve()),
        },
    }
    write_json_immutable(evaluation_path, evaluation)
    return checkpoint_path, evaluation_path, config_path


def train_router(
    feature_path: Path,
    manifest_path: Path,
    *,
    output_dir: Path,
    feature_type: str,
    router_input: str,
    decision_lambda: float,
    decision_temperature: float,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[Path, Path]:
    checkpoint_path = output_dir / f"router_{feature_type}_{router_input}.pt"
    evaluation_path = output_dir / f"evaluation_{feature_type}_{router_input}.json"
    if checkpoint_path.exists() or evaluation_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing router run: "
            f"{checkpoint_path}, {evaluation_path}"
        )

    with np.load(feature_path, allow_pickle=False) as archive:
        visual_np = np.asarray(archive["zv"], dtype=np.float32)
        action_np = np.asarray(archive["old_action_chunk"], dtype=np.float32)
        keep_success = np.asarray(archive["keep_success"], dtype=np.int64)
        replan_success = np.asarray(archive["replan_success"], dtype=np.int64)
        split_ids = np.asarray(archive["split"], dtype=np.int8)
        scene_seeds = np.asarray(archive["scene_seed"], dtype=np.int64)
        timesteps = np.asarray(archive["timestep"], dtype=np.int16)
        sample_ids = np.asarray(archive["sample_id"]).astype(str)

    train_indices = np.flatnonzero(split_ids == SPLIT_TO_ID["train"])
    val_indices = np.flatnonzero(split_ids == SPLIT_TO_ID["val"])
    test_indices = np.flatnonzero(split_ids == SPLIT_TO_ID["test"])
    if not all(len(indices) for indices in (train_indices, val_indices, test_indices)):
        raise ValueError("A dataset split is empty")
    for indices in (train_indices, val_indices, test_indices):
        labels = (keep_success[indices] == 0) & (replan_success[indices] == 1)
        if len(np.unique(labels)) != 2:
            raise ValueError("Each split must contain keep and strong-replan labels")

    visual_mean = (
        visual_np[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    )
    visual_std = (
        visual_np[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    )
    visual_std[visual_std < 1e-6] = 1.0
    visual_np = (visual_np - visual_mean) / visual_std

    flattened_train_actions = action_np[train_indices].reshape(-1, action_np.shape[-1])
    action_mean = flattened_train_actions.mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    action_std = flattened_train_actions.std(axis=0, dtype=np.float64).astype(
        np.float32
    )
    action_std[action_std < 1e-6] = 1.0
    action_np = (action_np - action_mean[None, None, :]) / action_std[None, None, :]

    visual = torch.from_numpy(np.ascontiguousarray(visual_np))
    action = torch.from_numpy(np.ascontiguousarray(action_np))
    outcomes = torch.from_numpy(
        np.stack([keep_success, replan_success], axis=1).astype(np.float32)
    )

    model = OutcomeRouter(
        visual_dim=visual.shape[1],
        horizon=action.shape[1],
        action_dim=action.shape[2],
        router_input=router_input,
    ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters >= 1_000_000:
        raise ValueError(f"Router exceeds 1M parameters: {trainable_parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    best_state = None
    best_epoch = None
    best_val_auc = -math.inf
    best_val_utility = -math.inf
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=generator).numpy()
        ]
        epoch_losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            logits = model(visual[indices].to(device), action[indices].to(device))
            target = outcomes[indices].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_p_keep, val_p_replan = predict(
            model, visual, action, val_indices, device, batch_size
        )
        val_metrics = evaluate_predictions(
            keep_success[val_indices],
            replan_success[val_indices],
            val_p_keep,
            val_p_replan,
            decision_lambda,
            decision_temperature,
        )
        val_auc = (
            float(val_metrics["roc_auc"])
            if val_metrics["roc_auc"] is not None
            else -math.inf
        )
        val_utility = float(val_metrics["counterfactual_utility"])
        improved = val_auc > best_val_auc + 1e-12 or (
            abs(val_auc - best_val_auc) <= 1e-12
            and val_utility > best_val_utility + 1e-12
        )
        if improved:
            best_val_auc = val_auc
            best_val_utility = val_utility
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_bce": float(np.mean(epoch_losses)),
                "val_roc_auc": val_metrics["roc_auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_strong_replan_recall": val_metrics["strong_replan_recall"],
                "val_counterfactual_utility": val_utility,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or improved:
            logging.info(
                "epoch=%d loss=%.5f val_auc=%s val_f1=%.3f val_recall=%.3f",
                epoch,
                history[-1]["train_bce"],
                (
                    f"{val_metrics['roc_auc']:.4f}"
                    if val_metrics["roc_auc"] is not None
                    else "n/a"
                ),
                val_metrics["f1"],
                val_metrics["strong_replan_recall"],
            )
        if stale_epochs >= patience:
            logging.info("Early stopping at epoch %d", epoch)
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    metrics_by_split = {}
    predictions_by_split = {}
    for split_name, indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        p_keep, p_replan = predict(model, visual, action, indices, device, batch_size)
        metrics_by_split[split_name] = evaluate_predictions(
            keep_success[indices],
            replan_success[indices],
            p_keep,
            p_replan,
            decision_lambda,
            decision_temperature,
        )
        advantage = p_replan - p_keep
        predictions_by_split[split_name] = [
            {
                "sample_id": sample_ids[index],
                "scene_seed": int(scene_seeds[index]),
                "timestep": int(timesteps[index]),
                "keep_success": int(keep_success[index]),
                "replan_success": int(replan_success[index]),
                "p_keep": float(p_keep[offset]),
                "p_replan": float(p_replan[offset]),
                "advantage": float(advantage[offset]),
                "predicted_replan": bool(advantage[offset] > decision_lambda),
            }
            for offset, index in enumerate(indices)
        ]

    feature_manifest = json.loads(manifest_path.read_text())
    checkpoint_payload = {
        "model_state_dict": best_state,
        "architecture": {
            "class": "OutcomeRouter",
            "visual_dim": int(visual.shape[1]),
            "horizon": int(action.shape[1]),
            "action_dim": int(action.shape[2]),
            "router_input": router_input,
            "hidden_dim": 128,
            "action_hidden_dim": 128,
            "action_embedding_dim": 64,
            "dropout": 0.1,
            "output_logits": ["keep_success", "replan_success"],
            "decision": "sigmoid(replan_logit)-sigmoid(keep_logit) > lambda",
            "trainable_parameters": trainable_parameters,
        },
        "normalization": {
            "visual_mean": torch.from_numpy(visual_mean),
            "visual_std": torch.from_numpy(visual_std),
            "action_mean": torch.from_numpy(action_mean),
            "action_std": torch.from_numpy(action_std),
        },
        "feature_type": feature_type,
        "feature_archive": str(feature_path.resolve()),
        "feature_archive_sha256": sha256_file(feature_path),
        "dataset_fingerprint": feature_manifest["dataset"]["dataset_fingerprint"],
        "split": feature_manifest["split"],
        "lambda": decision_lambda,
        "best_epoch": best_epoch,
        "seed": seed,
    }
    save_torch_immutable(checkpoint_path, checkpoint_payload)

    evaluation = {
        "purpose": "Offline event-triggered replan-router feasibility test",
        "task": feature_manifest["dataset"]["task"],
        "feature_type": feature_type,
        "router_input": router_input,
        "lambda": decision_lambda,
        "both_failure_excluded": True,
        "pi0_frozen": True,
        "online_control_enabled": False,
        "trainable_parameters": trainable_parameters,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "optimization": {
            "loss": "unweighted BCEWithLogits over keep/replan outcome heads",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "early_stopping_patience": patience,
            "selection": "validation router ROC-AUC, then validation utility",
        },
        "dataset": feature_manifest["dataset"],
        "split": feature_manifest["split"],
        "metrics": metrics_by_split,
        "training_history": history,
        "test_predictions": predictions_by_split["test"],
        "artifacts": {
            "feature_npz": str(feature_path.resolve()),
            "feature_manifest": str(manifest_path.resolve()),
            "router_checkpoint": str(checkpoint_path.resolve()),
        },
    }
    write_json_immutable(evaluation_path, evaluation)
    return checkpoint_path, evaluation_path


def resolve_ablation_feature_source(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    dataset_report: dict[str, Any],
    split_ids: np.ndarray,
    split_report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    if bool(args.base_feature_archive) != bool(args.base_feature_manifest):
        raise ValueError(
            "--base-feature-archive and --base-feature-manifest must be used together"
        )
    if args.base_feature_archive:
        candidates = [
            (
                args.base_feature_archive.resolve(),
                args.base_feature_manifest.resolve(),
            )
        ]
    else:
        candidates = [
            (
                output_dir / f"features_{args.feature_type}.npz",
                output_dir / f"features_{args.feature_type}_manifest.json",
            )
        ]
        if args.feature_type == "vision_encoder":
            minimal_dir = PROJECT_ROOT / "temp/outputs/replan_router_minimal_validation"
            candidates.append(
                (
                    minimal_dir / "features_vision_encoder.npz",
                    minimal_dir / "features_vision_encoder_manifest.json",
                )
            )

    for feature_path, manifest_path in candidates:
        if not feature_path.exists() and not manifest_path.exists():
            continue
        if not feature_path.exists() or not manifest_path.exists():
            raise FileExistsError(
                f"Incomplete frozen feature cache: {feature_path}, {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("feature_type") != args.feature_type
            or manifest["dataset"]["dataset_fingerprint"]
            != dataset_report["dataset_fingerprint"]
            or sha256_json(manifest["split"]) != sha256_json(split_report)
        ):
            if args.base_feature_archive:
                raise ValueError(
                    f"Explicit frozen feature cache is incompatible: {feature_path}"
                )
            logging.info("Ignoring incompatible feature cache %s", feature_path)
            continue
        with np.load(feature_path, allow_pickle=False) as archive:
            if not np.array_equal(
                np.asarray(archive["split"], dtype=np.int8), split_ids
            ):
                raise ValueError(f"Cached split IDs differ: {feature_path}")
        logging.info("Reusing frozen feature cache %s", feature_path)
        return feature_path, manifest_path

    if args.train_only:
        raise FileNotFoundError(
            "--train-only requested, but no compatible frozen feature cache exists"
        )
    return extract_features(
        samples,
        dataset_report,
        split_ids,
        split_report,
        feature_type=args.feature_type,
        output_dir=output_dir,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        server_config=args.server_config,
        horizon=args.horizon,
        action_dim=args.action_dim,
        batch_size=args.extract_batch_size,
        device=torch.device(args.extract_device),
    )


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_feature_ablation_summary(
    output_dir: Path,
    dataset_report: dict[str, Any],
    split_report: dict[str, Any],
) -> tuple[Path, Path]:
    def evaluation_row(config_name: str) -> dict[str, Any] | None:
        evaluation_path = output_dir / "evaluations" / f"{config_name}.json"
        if not evaluation_path.exists():
            return None
        evaluation = json.loads(evaluation_path.read_text())
        test = evaluation["metrics"]["test"]
        return {
            "feature_config": config_name,
            "feature_modules": evaluation["feature_modules"],
            "parameters": evaluation["trainable_parameters"],
            "test_sample_count": test["sample_count"],
            "roc_auc": test["roc_auc"],
            "accuracy": test["accuracy"],
            "f1": test["f1"],
            "strong_replan_recall": test["strong_replan_recall"],
            "false_trigger_rate": test["false_trigger_rate"],
            "precision_replan": test["precision_replan"],
            "eligible_by_split": evaluation["eligible_by_split"],
            "test_threshold_sweep": evaluation["threshold_sweep"]["test"],
            "evaluation": str(evaluation_path.resolve()),
        }

    rows = [
        row
        for config_name in FEATURE_ABLATION_ORDER
        if (row := evaluation_row(config_name)) is not None
    ]
    additional_rows = [
        row
        for config_name in ADDITIONAL_ACTION_MODELING_ORDER
        if (row := evaluation_row(config_name)) is not None
    ]
    aggregate_path = output_dir / "evaluation_feature_ablation.json"
    aggregate = {
        "purpose": "V0-V7 offline replan-router feature ablation summary",
        "task": dataset_report["task"],
        "dataset_fingerprint": dataset_report["dataset_fingerprint"],
        "split_identity_sha256": sha256_json(split_report),
        "thresholds": list(THRESHOLD_SWEEP),
        "headline_split": "test",
        "experiments": rows,
        "additional_action_modeling_experiments": additional_rows,
    }
    write_json_atomic(aggregate_path, aggregate)

    summary_path = output_dir / "feature_ablation_summary.md"
    lines = [
        "|Feature|Params|AUC|F1|Strong Recall|False Trigger|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        feature = f"{row['feature_config']}: " + " + ".join(row["feature_modules"])
        lines.append(
            f"|{feature}|{row['parameters']:,}|"
            f"{_format_metric(row['roc_auc'])}|"
            f"{_format_metric(row['f1'])}|"
            f"{_format_metric(row['strong_replan_recall'])}|"
            f"{_format_metric(row['false_trigger_rate'])}|"
        )
    if additional_rows:
        lines.extend(
            [
                "",
                "Additional action modeling:",
                "",
                "|Feature|Params|AUC|F1|Strong Recall|False Trigger|",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in additional_rows:
            feature = f"{row['feature_config']}: " + " + ".join(row["feature_modules"])
            lines.append(
                f"|{feature}|{row['parameters']:,}|"
                f"{_format_metric(row['roc_auc'])}|"
                f"{_format_metric(row['f1'])}|"
                f"{_format_metric(row['strong_replan_recall'])}|"
                f"{_format_metric(row['false_trigger_rate'])}|"
            )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(summary_path)
    return aggregate_path, summary_path


def run_feature_ablation(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    dataset_report: dict[str, Any],
    split_ids: np.ndarray,
    split_report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    if args.feature_type != "vision_encoder":
        raise ValueError(
            "V0-V7 are defined with the frozen pi0.5 vision_encoder feature"
        )
    if args.temporal_k < 1:
        raise ValueError("--temporal-k must be positive")
    if args.eef_waypoints < 2:
        raise ValueError("--eef-waypoints must be at least 2")
    try:
        gripper_indices = tuple(
            int(value.strip())
            for value in args.gripper_indices.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--gripper-indices must be comma-separated integers") from exc
    if not gripper_indices:
        raise ValueError("--gripper-indices cannot be empty")

    for child in ("configs", "checkpoints", "evaluations", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    dataset_manifest = {
        "dataset": dataset_report,
        "split": split_report,
        "split_identity_sha256": sha256_json(split_report),
        "data_roots": [str(path.resolve()) for path in args.data_root]
        if args.data_root
        else [str(path) for path in discover_data_roots(args.task)],
        "horizon": args.horizon,
        "action_dim": args.action_dim,
        "lambda_does_not_change_labels": True,
        "router_targets": ["keep_success", "replan_success"],
        "both_failure_excluded": True,
    }
    write_json_immutable(
        output_dir / "configs" / "dataset_manifest.json",
        dataset_manifest,
    )
    if args.dataset_only:
        return write_feature_ablation_summary(output_dir, dataset_report, split_report)

    feature_path, source_manifest_path = resolve_ablation_feature_source(
        args,
        samples,
        dataset_report,
        split_ids,
        split_report,
        output_dir,
    )
    if args.extract_only:
        print(feature_path)
        return write_feature_ablation_summary(output_dir, dataset_report, split_report)

    config_names = (
        FEATURE_ABLATION_ORDER if args.run_feature_ablation else (args.feature_config,)
    )
    for config_name in config_names:
        config = FEATURE_CONFIGS[config_name]
        configure_determinism(args.train_seed)
        prepared = prepare_ablation_features(
            feature_path,
            samples,
            config,
            temporal_k=args.temporal_k,
            gripper_indices=gripper_indices,
            eef_urdf_path=args.aloha_urdf.resolve(),
            eef_waypoints=args.eef_waypoints,
        )
        logging.info(
            "%s modules=%s eligible=%d/%d",
            config.name,
            "+".join(config.modules),
            int(prepared["valid_mask"].sum()),
            len(samples),
        )
        train_feature_ablation_config(
            prepared,
            config=config,
            feature_path=feature_path,
            source_manifest_path=source_manifest_path,
            dataset_report=dataset_report,
            split_report=split_report,
            output_dir=output_dir,
            temporal_k=args.temporal_k,
            decision_lambda=args.decision_lambda,
            decision_temperature=args.decision_temperature,
            device=torch.device(args.train_device),
            seed=args.train_seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    return write_feature_ablation_summary(output_dir, dataset_report, split_report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen pi0.5 features and train an offline shared-prefix "
            "counterfactual replan router."
        )
    )
    parser.add_argument("--task", default="move_playingcard_away")
    parser.add_argument(
        "--feature-type",
        "--feature_type",
        dest="feature_type",
        choices=(
            "vision_encoder",
            "vlm_hidden",
            "action_expert_hidden_tail",
        ),
        default="vision_encoder",
    )
    parser.add_argument(
        "--feature-config",
        "--feature_config",
        dest="feature_config",
        choices=tuple(FEATURE_CONFIGS),
        help=(
            "Run one modular input configuration. V0-V7 are the prescribed "
            "ablation; C3 is the full-action-chunk diagnostic."
        ),
    )
    parser.add_argument(
        "--run-feature-ablation",
        "--run_feature_ablation",
        dest="run_feature_ablation",
        action="store_true",
        help="Run V0 through V7 sequentially with one immutable shared split.",
    )
    parser.add_argument(
        "--router-input",
        "--router_input",
        dest="router_input",
        choices=("vision", "vision_action"),
        default="vision_action",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        type=Path,
        help=(
            "Paired-eval root containing paired_summary.json. Repeat for "
            "multiple roots. Move Playingcard Away defaults to its three "
            "validated absolute-cadence roots."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--base-feature-archive",
        type=Path,
        help="Optional compatible immutable frozen-feature NPZ to reuse.",
    )
    parser.add_argument(
        "--base-feature-manifest",
        type=Path,
        help="Manifest paired with --base-feature-archive.",
    )
    parser.add_argument("--server-config", default=DEFAULT_SERVER_CONFIG)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--lambda", dest="decision_lambda", type=float, default=0.05)
    parser.add_argument("--decision-temperature", type=float, default=0.10)
    parser.add_argument("--temporal-k", type=int, default=DEFAULT_TEMPORAL_K)
    parser.add_argument(
        "--eef-waypoints",
        type=int,
        default=DEFAULT_EEF_WAYPOINTS,
        help="Uniform EEF waypoints/velocity samples for the E1 action model.",
    )
    parser.add_argument(
        "--aloha-urdf",
        type=Path,
        default=DEFAULT_ALOHA_URDF,
        help="ALOHA URDF used by offline SAPIEN FK for E1.",
    )
    parser.add_argument(
        "--gripper-indices",
        default="6,13",
        help="Zero-based action dimensions used for signed gripper change.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--extract-device", default="cuda:0")
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--extract-batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Validate paired data and write the dataset/split manifest only.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Stop after writing the immutable frozen-feature archive.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Require and reuse an existing feature archive without loading pi0.5.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    args = build_parser().parse_args()
    if not 0.0 < args.decision_lambda < 1.0:
        raise ValueError("--lambda must be strictly between 0 and 1")
    if args.decision_temperature <= 0:
        raise ValueError("--decision-temperature must be positive")
    if args.dataset_only and (args.extract_only or args.train_only):
        raise ValueError("--dataset-only cannot be combined with other mode flags")
    if args.extract_only and args.train_only:
        raise ValueError("--extract-only and --train-only are mutually exclusive")
    if args.run_feature_ablation and args.feature_config:
        raise ValueError(
            "--run-feature-ablation cannot be combined with --feature-config"
        )
    if (
        min(args.extract_batch_size, args.train_batch_size, args.epochs, args.patience)
        < 1
    ):
        raise ValueError("Batch sizes, epochs, and patience must be positive")

    configure_determinism(args.train_seed)
    roots = (
        [path.resolve() for path in args.data_root]
        if args.data_root
        else discover_data_roots(args.task)
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (
            default_feature_ablation_dir()
            if args.run_feature_ablation or args.feature_config
            else default_output_dir(args.task)
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, dataset_report = load_decision_samples(
        args.task,
        roots,
        expected_horizon=args.horizon,
        expected_action_dim=args.action_dim,
    )
    split_ids, split_report = assign_scene_splits(samples, args.split_seed)
    if args.run_feature_ablation or args.feature_config:
        aggregate_path, summary_path = run_feature_ablation(
            args,
            samples,
            dataset_report,
            split_ids,
            split_report,
            output_dir,
        )
        print(aggregate_path)
        print(summary_path)
        return

    dataset_manifest_path = output_dir / "dataset_manifest.json"
    dataset_manifest = {
        "dataset": dataset_report,
        "split": split_report,
        "data_roots": [str(root) for root in roots],
        "horizon": args.horizon,
        "action_dim": args.action_dim,
        "lambda_does_not_change_labels": True,
        "router_label": "keep_success == 0 and replan_success == 1",
        "both_failure_excluded": True,
    }
    write_json_immutable(dataset_manifest_path, dataset_manifest)
    logging.info(
        "Eligible dataset: %d samples, %d scenes, transitions=%s",
        dataset_report["sample_count"],
        dataset_report["scene_count"],
        dataset_report["transition_counts"],
    )
    if args.dataset_only:
        print(dataset_manifest_path)
        return

    feature_path = output_dir / f"features_{args.feature_type}.npz"
    feature_manifest_path = output_dir / f"features_{args.feature_type}_manifest.json"
    if args.train_only:
        if not feature_path.exists() or not feature_manifest_path.exists():
            raise FileNotFoundError(
                f"--train-only requires {feature_path} and {feature_manifest_path}"
            )
    else:
        feature_path, feature_manifest_path = extract_features(
            samples,
            dataset_report,
            split_ids,
            split_report,
            feature_type=args.feature_type,
            output_dir=output_dir,
            checkpoint_dir=args.checkpoint_dir.resolve(),
            server_config=args.server_config,
            horizon=args.horizon,
            action_dim=args.action_dim,
            batch_size=args.extract_batch_size,
            device=torch.device(args.extract_device),
        )
    if args.extract_only:
        print(feature_path)
        return

    configure_determinism(args.train_seed)
    checkpoint_path, evaluation_path = train_router(
        feature_path,
        feature_manifest_path,
        output_dir=output_dir,
        feature_type=args.feature_type,
        router_input=args.router_input,
        decision_lambda=args.decision_lambda,
        decision_temperature=args.decision_temperature,
        device=torch.device(args.train_device),
        seed=args.train_seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(checkpoint_path)
    print(evaluation_path)


if __name__ == "__main__":
    main()
