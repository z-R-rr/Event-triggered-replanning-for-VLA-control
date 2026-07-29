"""Fair feature-selection training with scene-grouped OOF evaluation.

Result 1 is computed from out-of-fold predictions covering the same 348
counterfactual samples for every candidate feature. The online checkpoint is
then retrained on all 348 samples for the median best epoch selected by the
OOF folds.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import train_replan_router as base


PRIMARY_FEATURE_CONFIGS = ("V0", "V1", "V2", "V3", "V5", "E1", "E2")
EXTENSION_FEATURE_CONFIGS = ("Z1",)
EXPECTED_TRANSITIONS = {"rescue": 56, "harm": 16, "both_success": 276}


def scene_strata(samples: list[dict[str, Any]]) -> dict[int, str]:
    by_scene: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_scene.setdefault(int(sample["scene_seed"]), []).append(sample)
    strata = {}
    for scene_seed, scene_samples in by_scene.items():
        transitions = {sample["transition"] for sample in scene_samples}
        if "rescue" in transitions:
            strata[scene_seed] = "has_rescue"
        elif "harm" in transitions:
            strata[scene_seed] = "has_harm"
        else:
            strata[scene_seed] = "both_success_only"
    return strata


def build_scene_grouped_folds(
    samples: list[dict[str, Any]], folds: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    strata = scene_strata(samples)
    scenes_by_stratum: dict[str, list[int]] = {}
    for scene_seed, stratum in strata.items():
        scenes_by_stratum.setdefault(stratum, []).append(scene_seed)

    scene_to_fold: dict[int, int] = {}
    stratum_report = {}
    for stratum_index, (stratum, scenes) in enumerate(
        sorted(scenes_by_stratum.items())
    ):
        shuffled = sorted(scenes)
        random.Random(seed + 1009 * (stratum_index + 1)).shuffle(shuffled)
        fold_scenes = [[] for _ in range(folds)]
        for index, scene_seed in enumerate(shuffled):
            fold = index % folds
            scene_to_fold[scene_seed] = fold
            fold_scenes[fold].append(scene_seed)
        stratum_report[stratum] = {
            "scene_count": len(shuffled),
            "fold_scene_counts": [len(values) for values in fold_scenes],
            "fold_scene_seeds": fold_scenes,
        }

    sample_folds = np.asarray(
        [scene_to_fold[int(sample["scene_seed"])] for sample in samples],
        dtype=np.int8,
    )
    fold_reports = []
    for outer_fold in range(folds):
        test_scenes = {
            scene_seed
            for scene_seed, fold in scene_to_fold.items()
            if fold == outer_fold
        }
        remaining_by_stratum: dict[str, list[int]] = {}
        for scene_seed, stratum in strata.items():
            if scene_seed not in test_scenes:
                remaining_by_stratum.setdefault(stratum, []).append(scene_seed)
        val_scenes: set[int] = set()
        for stratum_index, (stratum, scenes) in enumerate(
            sorted(remaining_by_stratum.items())
        ):
            shuffled = sorted(scenes)
            random.Random(
                seed + 100_003 * (outer_fold + 1) + 997 * (stratum_index + 1)
            ).shuffle(shuffled)
            val_count = max(1, int(round(0.15 * len(shuffled))))
            val_scenes.update(shuffled[:val_count])
        train_scenes = set(strata) - test_scenes - val_scenes
        scene_split = {
            scene_seed: (
                "test"
                if scene_seed in test_scenes
                else "val"
                if scene_seed in val_scenes
                else "train"
            )
            for scene_seed in strata
        }
        sample_counts = {
            name: sum(
                scene_split[int(sample["scene_seed"])] == name for sample in samples
            )
            for name in base.SPLIT_NAMES
        }
        transition_counts = {
            name: {
                transition: sum(
                    scene_split[int(sample["scene_seed"])] == name
                    and sample["transition"] == transition
                    for sample in samples
                )
                for transition in EXPECTED_TRANSITIONS
            }
            for name in base.SPLIT_NAMES
        }
        fold_reports.append(
            {
                "fold": outer_fold,
                "scene_counts": {
                    "train": len(train_scenes),
                    "val": len(val_scenes),
                    "test": len(test_scenes),
                },
                "sample_counts": sample_counts,
                "transition_counts": transition_counts,
                "scene_splits": {
                    name: sorted(
                        scene_seed
                        for scene_seed, split_name in scene_split.items()
                        if split_name == name
                    )
                    for name in base.SPLIT_NAMES
                },
            }
        )
    report = {
        "method": (
            f"{folds}-fold deterministic scene-grouped stratified OOF; "
            "15% scene-grouped validation inside each outer training pool"
        ),
        "seed": seed,
        "folds": folds,
        "strata": stratum_report,
        "fold_reports": fold_reports,
    }
    return sample_folds, report


def split_ids_for_fold(
    samples: list[dict[str, Any]], fold_report: dict[str, Any]
) -> np.ndarray:
    scene_to_split = {
        scene_seed: base.SPLIT_TO_ID[split_name]
        for split_name, scene_seeds in fold_report["scene_splits"].items()
        for scene_seed in scene_seeds
    }
    return np.asarray(
        [scene_to_split[int(sample["scene_seed"])] for sample in samples],
        dtype=np.int8,
    )


def write_split_archive(
    base_archive: Path,
    output_path: Path,
    split_ids: np.ndarray,
    *,
    dataset_fingerprint: str,
    protocol: dict[str, Any],
    z1_feature_archive: Path | None = None,
) -> Path:
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    if output_path.exists() and manifest_path.exists():
        with np.load(output_path, allow_pickle=False) as archive:
            if not np.array_equal(np.asarray(archive["split"]), split_ids):
                raise FileExistsError(f"Incompatible split archive: {output_path}")
        return manifest_path
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Incomplete split archive: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(base_archive, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    if z1_feature_archive is not None:
        with np.load(z1_feature_archive, allow_pickle=False) as source:
            if not np.array_equal(
                np.asarray(source["sample_id"]).astype(str),
                np.asarray(arrays["sample_id"]).astype(str),
            ):
                raise ValueError(
                    "Z1 action-hidden archive sample order does not match vision"
                )
            arrays["action_expert_hidden_tail"] = np.asarray(
                source["zv"], dtype=np.float32
            )
    if len(split_ids) != len(arrays["split"]):
        raise ValueError("Split length does not match frozen feature archive")
    arrays["split"] = split_ids
    np.savez_compressed(output_path, **arrays)
    base.write_json_immutable(
        manifest_path,
        {
            "purpose": "Scene-grouped feature-selection split archive",
            "feature_type": "vision_encoder",
            "dataset_fingerprint": dataset_fingerprint,
            "protocol": protocol,
            "source_feature_archive": str(base_archive.resolve()),
            "source_feature_archive_sha256": base.sha256_file(base_archive),
            "split_archive": str(output_path.resolve()),
            "split_archive_sha256": base.sha256_file(output_path),
            "z1_feature_archive": (
                str(z1_feature_archive.resolve())
                if z1_feature_archive is not None
                else None
            ),
            "z1_feature_archive_sha256": (
                base.sha256_file(z1_feature_archive)
                if z1_feature_archive is not None
                else None
            ),
        },
    )
    return manifest_path


def build_model(
    prepared: dict[str, Any], config: base.FeatureConfig, device: torch.device
) -> tuple[base.FeatureAblationRouter, dict[str, int], int | None, int]:
    action_tensor_names = {
        "future_action_tail",
        "full_action_chunk",
        "eef_trajectory",
    }
    direct_feature_dims = {
        name: int(values.shape[1])
        for name, values in prepared["features"].items()
        if name not in action_tensor_names
    }
    action_feature_dim = (
        int(prepared["features"][config.action_mode][0].numel())
        if config.action_mode in action_tensor_names
        else None
    )
    model = base.FeatureAblationRouter(
        direct_feature_dims,
        action_mode=config.action_mode,
        horizon=prepared["horizon"],
        action_dim=prepared["action_dim"],
        action_feature_dim=action_feature_dim,
    ).to(device)
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return model, direct_feature_dims, action_feature_dim, parameters


def train_epoch(
    model: base.FeatureAblationRouter,
    features: dict[str, torch.Tensor],
    outcomes: torch.Tensor,
    indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
    batch_size: int,
) -> float:
    model.train()
    permutation = indices[
        torch.randperm(len(indices), generator=generator).numpy()
    ]
    losses = []
    for start in range(0, len(permutation), batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch = {
            name: values[batch_indices].to(device)
            for name, values in features.items()
        }
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(
            logits, outcomes[batch_indices].to(device)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def fit_fold(
    prepared: dict[str, Any],
    config: base.FeatureConfig,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    decision_lambda: float,
    decision_temperature: float,
) -> dict[str, Any]:
    base.configure_determinism(seed)
    model, _, _, parameters = build_model(prepared, config, device)
    split_ids = prepared["split_ids"]
    valid_mask = prepared["valid_mask"]
    indices = {
        name: np.flatnonzero(valid_mask & (split_ids == base.SPLIT_TO_ID[name]))
        for name in base.SPLIT_NAMES
    }
    if not all(len(values) for values in indices.values()):
        raise ValueError(f"Empty OOF split for {config.name}")
    keep = prepared["keep_success"]
    replan = prepared["replan_success"]
    for name, values in indices.items():
        strong = (keep[values] == 0) & (replan[values] == 1)
        if len(np.unique(strong)) != 2:
            raise ValueError(f"{config.name} fold {name} lacks both router classes")
    outcomes = torch.from_numpy(
        np.stack([keep, replan], axis=1).astype(np.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_epoch = 0
    best_auc = -math.inf
    best_utility = -math.inf
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model,
            prepared["features"],
            outcomes,
            indices["train"],
            optimizer,
            generator,
            device,
            batch_size,
        )
        p_keep, p_replan = base.predict_feature_ablation(
            model,
            prepared["features"],
            indices["val"],
            device,
            batch_size,
        )
        metrics = base.evaluate_predictions(
            keep[indices["val"]],
            replan[indices["val"]],
            p_keep,
            p_replan,
            decision_lambda,
            decision_temperature,
        )
        auc = (
            float(metrics["roc_auc"])
            if metrics["roc_auc"] is not None
            else -math.inf
        )
        utility = float(metrics["counterfactual_utility"])
        improved = auc > best_auc + 1e-12 or (
            abs(auc - best_auc) <= 1e-12 and utility > best_utility + 1e-12
        )
        if improved:
            best_auc = auc
            best_utility = utility
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_bce": loss,
                "val_roc_auc": metrics["roc_auc"],
                "val_f1": metrics["f1"],
                "val_strong_replan_recall": metrics["strong_replan_recall"],
                "val_false_trigger_rate": metrics["false_trigger_rate"],
                "val_counterfactual_utility": utility,
            }
        )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"No OOF checkpoint produced for {config.name}")
    model.load_state_dict(best_state)
    p_keep, p_replan = base.predict_feature_ablation(
        model,
        prepared["features"],
        indices["test"],
        device,
        batch_size,
    )
    return {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "trainable_parameters": parameters,
        "test_indices": indices["test"],
        "p_keep": p_keep,
        "p_replan": p_replan,
        "history": history,
    }


def fit_all_samples(
    prepared: dict[str, Any],
    config: base.FeatureConfig,
    *,
    fixed_epochs: int,
    device: torch.device,
    seed: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[float]]:
    base.configure_determinism(seed)
    model, direct_dims, action_feature_dim, parameters = build_model(
        prepared, config, device
    )
    indices = np.flatnonzero(prepared["valid_mask"])
    if len(indices) != len(prepared["valid_mask"]):
        raise ValueError(
            f"{config.name} is not eligible on all fixed feature-selection samples"
        )
    outcomes = torch.from_numpy(
        np.stack(
            [prepared["keep_success"], prepared["replan_success"]], axis=1
        ).astype(np.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(fixed_epochs):
        losses.append(
            train_epoch(
                model,
                prepared["features"],
                outcomes,
                indices,
                optimizer,
                generator,
                device,
                batch_size,
            )
        )
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    architecture = {
        "class": "FeatureAblationRouter",
        "direct_feature_dims": direct_dims,
        "horizon": int(prepared["horizon"]),
        "action_dim": int(prepared["action_dim"]),
        "action_mode": config.action_mode,
        "action_feature_dim": action_feature_dim,
        "hidden_dim": 128,
        "action_hidden_dim": 128,
        "action_embedding_dim": 64,
        "dropout": 0.1,
        "output_logits": ["keep_success", "replan_success"],
        "decision": "sigmoid(replan_logit)-sigmoid(keep_logit) > lambda",
        "trainable_parameters": parameters,
    }
    return state, architecture, losses


def transition_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    return {
        transition: sum(sample["transition"] == transition for sample in samples)
        for transition in EXPECTED_TRANSITIONS
    }


def train_config(
    args: argparse.Namespace,
    *,
    config: base.FeatureConfig,
    samples: list[dict[str, Any]],
    dataset_report: dict[str, Any],
    fold_report: dict[str, Any],
    fold_archives: list[tuple[Path, Path]],
    all_archive: tuple[Path, Path],
    output_dir: Path,
) -> dict[str, Any]:
    evaluation_path = output_dir / "evaluations" / f"{config.name}.json"
    checkpoint_path = output_dir / "checkpoints" / f"{config.name}.pt"
    feature_manifest_path = (
        output_dir / "configs" / f"{config.name}_feature_manifest.json"
    )
    log_path = output_dir / "logs" / f"{config.name}_training_log.json"
    if all(
        path.exists()
        for path in (
            evaluation_path,
            checkpoint_path,
            feature_manifest_path,
            log_path,
        )
    ):
        payload = json.loads(evaluation_path.read_text())
        if (
            payload["dataset"]["dataset_fingerprint"]
            != dataset_report["dataset_fingerprint"]
            or payload["feature_config"] != config.name
            or payload["protocol"]["fold_identity_sha256"]
            != base.sha256_json(fold_report)
        ):
            raise FileExistsError(f"Incompatible existing run: {config.name}")
        return payload
    if any(
        path.exists()
        for path in (
            evaluation_path,
            checkpoint_path,
            feature_manifest_path,
            log_path,
        )
    ):
        raise FileExistsError(f"Incomplete existing run: {config.name}")

    gripper_indices = tuple(
        int(value.strip()) for value in args.gripper_indices.split(",")
    )
    oof_keep = np.full(len(samples), np.nan, dtype=np.float32)
    oof_replan = np.full(len(samples), np.nan, dtype=np.float32)
    fold_summaries = []
    fold_histories = []
    trainable_parameters = None
    for fold, ((archive, _), report) in enumerate(
        zip(fold_archives, fold_report["fold_reports"], strict=True)
    ):
        prepared = base.prepare_ablation_features(
            archive,
            samples,
            config,
            temporal_k=args.temporal_k,
            gripper_indices=gripper_indices,
            eef_urdf_path=args.aloha_urdf.resolve(),
            eef_waypoints=args.eef_waypoints,
        )
        if int(prepared["valid_mask"].sum()) != len(samples):
            raise ValueError(
                f"{config.name} does not cover the common 348-sample cohort"
            )
        result = fit_fold(
            prepared,
            config,
            device=torch.device(args.train_device),
            seed=args.train_seed + fold,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            decision_lambda=args.decision_lambda,
            decision_temperature=args.decision_temperature,
        )
        test_indices = result["test_indices"]
        if np.any(~np.isnan(oof_keep[test_indices])):
            raise AssertionError("OOF test samples overlap")
        oof_keep[test_indices] = result["p_keep"]
        oof_replan[test_indices] = result["p_replan"]
        trainable_parameters = result["trainable_parameters"]
        fold_summaries.append(
            {
                "fold": fold,
                "best_epoch": result["best_epoch"],
                "epochs_ran": result["epochs_ran"],
                "trainable_parameters": result["trainable_parameters"],
                "scene_counts": report["scene_counts"],
                "sample_counts": report["sample_counts"],
                "transition_counts": report["transition_counts"],
            }
        )
        fold_histories.append(
            {"fold": fold, "history": result["history"]}
        )
    if np.any(np.isnan(oof_keep)) or np.any(np.isnan(oof_replan)):
        raise AssertionError("OOF predictions do not cover every sample exactly once")

    keep = np.asarray([sample["keep_success"] for sample in samples], dtype=np.int64)
    replan = np.asarray(
        [sample["replan_success"] for sample in samples], dtype=np.int64
    )
    oof_metrics = base.evaluate_predictions(
        keep,
        replan,
        oof_keep,
        oof_replan,
        args.decision_lambda,
        args.decision_temperature,
    )
    threshold_sweep = [
        {
            "threshold": threshold,
            **base.evaluate_predictions(
                keep,
                replan,
                oof_keep,
                oof_replan,
                threshold,
                args.decision_temperature,
            ),
        }
        for threshold in base.THRESHOLD_SWEEP
    ]
    fixed_epochs = max(
        1,
        int(
            round(
                float(
                    np.median(
                        [summary["best_epoch"] for summary in fold_summaries]
                    )
                )
            )
        ),
    )

    all_prepared = base.prepare_ablation_features(
        all_archive[0],
        samples,
        config,
        temporal_k=args.temporal_k,
        gripper_indices=gripper_indices,
        eef_urdf_path=args.aloha_urdf.resolve(),
        eef_waypoints=args.eef_waypoints,
    )
    final_state, architecture, full_losses = fit_all_samples(
        all_prepared,
        config,
        fixed_epochs=fixed_epochs,
        device=torch.device(args.train_device),
        seed=args.train_seed,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    feature_manifest = {
        "purpose": "Full-cohort online checkpoint feature manifest",
        "feature_config": config.name,
        "modules": list(config.modules),
        "definitions": all_prepared["definitions"],
        "dataset_fingerprint": dataset_report["dataset_fingerprint"],
        "sample_count": len(samples),
        "transition_counts": transition_counts(samples),
        "normalization_fit": "all 348 samples for final online checkpoint",
        "oof_fold_identity_sha256": base.sha256_json(fold_report),
        "source_feature_archive": str(all_archive[0].resolve()),
        "source_feature_archive_sha256": base.sha256_file(all_archive[0]),
        "source_feature_type": "vision_encoder",
        "pi0_frozen": True,
        "online_control_enabled": False,
    }
    base.write_json_immutable(feature_manifest_path, feature_manifest)
    checkpoint_payload = {
        "model_state_dict": final_state,
        "architecture": architecture,
        "normalization": base._numpy_tree_to_torch(
            all_prepared["normalization"]
        ),
        "feature_config": config.name,
        "feature_manifest": str(feature_manifest_path.resolve()),
        "feature_manifest_sha256": base.sha256_file(feature_manifest_path),
        "dataset_fingerprint": dataset_report["dataset_fingerprint"],
        "split_identity_sha256": base.sha256_json(fold_report),
        "training_sample_count": len(samples),
        "training_transition_counts": transition_counts(samples),
        "lambda": args.decision_lambda,
        "best_epoch": fixed_epochs,
        "seed": args.train_seed,
    }
    base.save_torch_immutable(checkpoint_path, checkpoint_payload)
    base.write_json_immutable(
        log_path,
        {
            "feature_config": config.name,
            "folds": fold_histories,
            "full_training": {
                "fixed_epochs": fixed_epochs,
                "epoch_bce": full_losses,
            },
        },
    )
    evaluation = {
        "purpose": "Result 1: common-cohort scene-grouped OOF feature evaluation",
        "task": dataset_report["task"],
        "feature_config": config.name,
        "feature_modules": list(config.modules),
        "trainable_parameters": trainable_parameters,
        "lambda": args.decision_lambda,
        "dataset": dataset_report,
        "protocol": {
            "method": fold_report["method"],
            "fold_identity_sha256": base.sha256_json(fold_report),
            "all_samples_predicted_out_of_fold_once": True,
            "final_online_checkpoint_training": (
                "all 348 samples for median OOF-best epoch"
            ),
            "final_training_sample_count": len(samples),
            "final_training_transition_counts": transition_counts(samples),
            "final_training_epochs": fixed_epochs,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.train_batch_size,
        },
        "fold_summaries": fold_summaries,
        "metrics": {"oof": oof_metrics},
        "threshold_sweep": threshold_sweep,
        "oof_predictions": [
            {
                "sample_id": samples[index]["sample_id"],
                "scene_seed": int(samples[index]["scene_seed"]),
                "timestep": int(samples[index]["timestep"]),
                "transition": samples[index]["transition"],
                "keep_success": int(keep[index]),
                "replan_success": int(replan[index]),
                "p_keep": float(oof_keep[index]),
                "p_replan": float(oof_replan[index]),
                "advantage": float(oof_replan[index] - oof_keep[index]),
            }
            for index in range(len(samples))
        ],
        "artifacts": {
            "feature_manifest": str(feature_manifest_path.resolve()),
            "online_checkpoint": str(checkpoint_path.resolve()),
            "training_log": str(log_path.resolve()),
        },
    }
    base.write_json_immutable(evaluation_path, evaluation)
    return evaluation


def write_summary(
    output_dir: Path,
    dataset_report: dict[str, Any],
    fold_report: dict[str, Any],
    configs: tuple[str, ...],
) -> None:
    rows = []
    for name in configs:
        path = output_dir / "evaluations" / f"{name}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        metrics = payload["metrics"]["oof"]
        rows.append(
            {
                "feature_config": name,
                "feature_modules": payload["feature_modules"],
                "parameters": payload["trainable_parameters"],
                "sample_count": metrics["sample_count"],
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "strong_replan_recall": metrics["strong_replan_recall"],
                "false_trigger_rate": metrics["false_trigger_rate"],
                "precision_replan": metrics["precision_replan"],
            }
        )
    if len(rows) != len(configs):
        logging.info(
            "Result 1 summary pending: %d/%d feature configs complete",
            len(rows),
            len(configs),
        )
        return
    base.write_json_immutable(
        output_dir / "evaluation_feature_selection_oof.json",
        {
            "purpose": "Result 1: fair common-cohort input-feature comparison",
            "dataset": {
                "fingerprint": dataset_report["dataset_fingerprint"],
                "sample_count": dataset_report["sample_count"],
                "scene_count": dataset_report["scene_count"],
                "transition_counts": transition_counts_from_report(dataset_report),
            },
            "fold_identity_sha256": base.sha256_json(fold_report),
            "experiments": rows,
        },
    )
    lines = [
        "|Feature|Params|OOF N|AUC|F1|Strong Recall|False Trigger|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['feature_config']}|{row['parameters']}|"
            f"{row['sample_count']}|{row['roc_auc']:.4f}|{row['f1']:.4f}|"
            f"{row['strong_replan_recall']:.4f}|"
            f"{row['false_trigger_rate']:.4f}|"
        )
    summary_path = output_dir / "feature_selection_oof_summary.md"
    text = "\n".join(lines) + "\n"
    if summary_path.exists() and summary_path.read_text() != text:
        raise FileExistsError(f"Incompatible summary: {summary_path}")
    if not summary_path.exists():
        summary_path.write_text(text)


def transition_counts_from_report(dataset_report: dict[str, Any]) -> dict[str, int]:
    counts = dataset_report["transition_counts"]
    return {name: int(counts[name]) for name in EXPECTED_TRANSITIONS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="move_playingcard_away")
    parser.add_argument("--data-root", action="append", type=Path, required=True)
    parser.add_argument("--base-feature-archive", type=Path, required=True)
    parser.add_argument("--base-feature-manifest", type=Path, required=True)
    parser.add_argument(
        "--z1-feature-archive",
        type=Path,
        help=(
            "Frozen action_expert_hidden_tail archive produced by "
            "train_replan_router.py; required when Z1 is requested."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-config",
        action="append",
        choices=(*PRIMARY_FEATURE_CONFIGS, *EXTENSION_FEATURE_CONFIGS),
        help="Repeat for a subset; default runs all primary candidates.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--lambda", dest="decision_lambda", type=float, default=0.05)
    parser.add_argument("--decision-temperature", type=float, default=0.10)
    parser.add_argument("--temporal-k", type=int, default=5)
    parser.add_argument("--eef-waypoints", type=int, default=5)
    parser.add_argument("--aloha-urdf", type=Path, default=base.DEFAULT_ALOHA_URDF)
    parser.add_argument("--gripper-indices", default="6,13")
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write and validate the common scene folds without training.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    args = build_parser().parse_args()
    if args.folds < 3:
        raise ValueError("--folds must be at least 3")
    roots = [path.resolve() for path in args.data_root]
    samples, dataset_report = base.load_decision_samples(
        args.task,
        roots,
        expected_horizon=args.horizon,
        expected_action_dim=args.action_dim,
    )
    counts = transition_counts(samples)
    if counts != EXPECTED_TRANSITIONS:
        raise ValueError(
            f"Feature-selection dataset mismatch: {counts} != {EXPECTED_TRANSITIONS}"
        )
    selected = (
        tuple(args.feature_config)
        if args.feature_config
        else PRIMARY_FEATURE_CONFIGS
    )
    if "Z1" in selected and args.z1_feature_archive is None:
        raise ValueError("--z1-feature-archive is required for Z1")
    z1_feature_archive = (
        args.z1_feature_archive.resolve()
        if args.z1_feature_archive is not None
        else None
    )
    sample_folds, fold_report = build_scene_grouped_folds(
        samples, args.folds, args.split_seed
    )
    output_dir = args.output_dir.resolve()
    for child in ("configs", "checkpoints", "evaluations", "logs", "folds"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    dataset_manifest = {
        "purpose": "Common dataset and scene folds for input-feature selection",
        "dataset": dataset_report,
        "data_roots": [str(root) for root in roots],
        "horizon": args.horizon,
        "action_dim": args.action_dim,
        "both_failure_excluded": True,
        "expected_transition_counts": EXPECTED_TRANSITIONS,
        "fold_protocol": fold_report,
        "fold_identity_sha256": base.sha256_json(fold_report),
        "primary_feature_configs": list(PRIMARY_FEATURE_CONFIGS),
        "requested_feature_configs": list(selected),
        "excluded_noncomparable_configs": {
            "V4": "missing t-k history violates the fixed 348-sample cohort",
            "V6": "missing t-k history violates the fixed 348-sample cohort",
            "V7": "missing t-k history violates the fixed 348-sample cohort",
            "C3": "full-chunk diagnostic is outside the prescribed candidates",
        },
    }
    base.write_json_immutable(
        output_dir / "configs/dataset_oof_manifest.json", dataset_manifest
    )

    base_archive = args.base_feature_archive.resolve()
    base_manifest = json.loads(args.base_feature_manifest.read_text())
    if (
        base_manifest["dataset"]["dataset_fingerprint"]
        != dataset_report["dataset_fingerprint"]
    ):
        raise ValueError("Base frozen feature dataset mismatch")
    fold_archives = []
    for fold, report in enumerate(fold_report["fold_reports"]):
        split_ids = split_ids_for_fold(samples, report)
        archive = output_dir / "folds" / f"fold_{fold}_features.npz"
        manifest = write_split_archive(
            base_archive,
            archive,
            split_ids,
            dataset_fingerprint=dataset_report["dataset_fingerprint"],
            protocol=report,
            z1_feature_archive=z1_feature_archive,
        )
        fold_archives.append((archive, manifest))
    all_archive_path = output_dir / "folds/all_samples_features.npz"
    all_manifest_path = write_split_archive(
        base_archive,
        all_archive_path,
        np.full(len(samples), base.SPLIT_TO_ID["train"], dtype=np.int8),
        dataset_fingerprint=dataset_report["dataset_fingerprint"],
        protocol={
            "purpose": "All samples train split for final online checkpoint",
            "sample_count": len(samples),
            "transition_counts": counts,
        },
        z1_feature_archive=z1_feature_archive,
    )
    print(output_dir / "configs/dataset_oof_manifest.json")
    if args.prepare_only:
        return

    for name in selected:
        logging.info("Training common-cohort feature candidate %s", name)
        train_config(
            args,
            config=base.FEATURE_CONFIGS[name],
            samples=samples,
            dataset_report=dataset_report,
            fold_report=fold_report,
            fold_archives=fold_archives,
            all_archive=(all_archive_path, all_manifest_path),
            output_dir=output_dir,
        )
    write_summary(output_dir, dataset_report, fold_report, selected)
    print(output_dir / "evaluation_feature_selection_oof.json")


if __name__ == "__main__":
    main()
