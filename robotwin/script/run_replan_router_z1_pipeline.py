"""Run Z1 through the common OOF and balanced Online-40 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/replan_router_z1_evaluation_v3.json"
OPENPI_PYTHON = WORKSPACE / "openpi/.venv/bin/python"
ROBOTWIN_PYTHON = WORKSPACE / "RoboTwin/.venv/bin/python"


def project_path(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Hash mismatch: {path}\nexpected={expected}\nactual={actual}"
        )


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 3
        or payload.get("status") != "active_extension"
        or payload.get("feature", {}).get("name") != "Z1"
    ):
        raise ValueError("Expected active Z1 schema-v3 config")
    return payload


def output_root(config: dict[str, Any]) -> Path:
    return project_path(config["artifacts"]["output_root"])


def append_data_roots(command: list[str], config: dict[str, Any]) -> list[str]:
    for root in config["offline"]["data_roots"]:
        command.extend(["--data-root", str(project_path(root))])
    return command


def extract_command(config: dict[str, Any]) -> list[str]:
    artifacts = config["artifacts"]
    task = config["task"]
    return append_data_roots(
        [
            str(OPENPI_PYTHON),
            str(PROJECT_ROOT / "robotwin/script/train_replan_router.py"),
            "--task",
            task["name"],
            "--feature-type",
            "action_expert_hidden_tail",
            "--output-dir",
            str(output_root(config) / "features"),
            "--checkpoint-dir",
            str(Path(artifacts["pi05_checkpoint"]).resolve()),
            "--server-config",
            artifacts["server_config"],
            "--horizon",
            str(task["horizon"]),
            "--action-dim",
            str(task["action_dim"]),
            "--extract-device",
            "cuda:0",
            "--extract-batch-size",
            "4",
            "--extract-only",
        ],
        config,
    )


def offline_command(config: dict[str, Any]) -> list[str]:
    offline = config["offline"]
    task = config["task"]
    artifacts = config["artifacts"]
    return append_data_roots(
        [
            str(OPENPI_PYTHON),
            str(
                PROJECT_ROOT
                / "robotwin/script/train_replan_router_feature_selection.py"
            ),
            "--task",
            task["name"],
            "--base-feature-archive",
            str(project_path(artifacts["base_visual_archive"])),
            "--base-feature-manifest",
            str(project_path(artifacts["base_visual_manifest"])),
            "--z1-feature-archive",
            str(
                output_root(config)
                / "features/features_action_expert_hidden_tail.npz"
            ),
            "--output-dir",
            str(output_root(config) / "offline"),
            "--feature-config",
            "Z1",
            "--folds",
            str(offline["oof_folds"]),
            "--split-seed",
            str(offline["split_seed"]),
            "--train-seed",
            str(offline["train_seed"]),
            "--horizon",
            str(task["horizon"]),
            "--action-dim",
            str(task["action_dim"]),
            "--lambda",
            str(offline["lambda"]),
            "--train-device",
            "cuda:0",
            "--train-batch-size",
            str(offline["batch_size"]),
            "--epochs",
            str(offline["max_epochs"]),
            "--patience",
            str(offline["patience"]),
            "--learning-rate",
            str(offline["learning_rate"]),
            "--weight-decay",
            str(offline["weight_decay"]),
        ],
        config,
    )


def online_command(
    config: dict[str, Any], *, prepare_only: bool
) -> list[str]:
    online = config["online"]
    task = config["task"]
    artifacts = config["artifacts"]
    root = output_root(config)
    command = [
        str(ROBOTWIN_PYTHON),
        str(
            PROJECT_ROOT
            / "robotwin/script/run_pi05_online_router_feature_candidates_eval.py"
        ),
        "--output-dir",
        str(root / "online"),
        "--scene-manifest",
        str(project_path(online["seed_manifest"])),
        "--resolved-scene-manifest",
        str(project_path(online["resolved_manifest"])),
        "--scene-count",
        str(online["scene_count"]),
        "--task-name",
        task["name"],
        "--task-config",
        task["config"],
        "--horizon",
        str(task["horizon"]),
        "--r0",
        str(task["natural_replan_interval"]),
        "--query-interval",
        str(online["query_interval"]),
        "--query-max-action",
        str(online["query_max_action"]),
        "--router-max-replans",
        str(online["max_replans"]),
        "--router-lambda",
        str(online["lambda"]),
        "--gpus",
        online["gpus"],
        "--port-base",
        str(online["port_base"]),
        "--seed",
        str(online["policy_seed"]),
        "--feature-archive",
        str(project_path(artifacts["base_visual_archive"])),
        "--feature-manifest",
        str(project_path(artifacts["base_visual_manifest"])),
        "--candidate",
        f"Z1={root / 'offline/checkpoints/Z1.pt'}",
    ]
    if prepare_only:
        command.append("--prepare-only")
    return command


def validate(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["artifacts"]
    online = config["online"]
    offline = config["offline"]
    checks = {}
    require_hash(
        project_path(artifacts["base_visual_archive"]),
        artifacts["base_visual_archive_sha256"],
    )
    checks["base_visual_archive_hash"] = True
    require_hash(
        project_path(artifacts["base_visual_manifest"]),
        artifacts["base_visual_manifest_sha256"],
    )
    checks["base_visual_manifest_hash"] = True
    require_hash(
        project_path(online["seed_manifest"]),
        online["seed_manifest_sha256"],
    )
    checks["online_seed_manifest_hash"] = True
    require_hash(
        project_path(online["resolved_manifest"]),
        online["resolved_manifest_sha256"],
    )
    checks["online_resolved_manifest_hash"] = True
    seed_manifest = json.loads(project_path(online["seed_manifest"]).read_text())
    checks["online_cohort_identity"] = (
        seed_manifest["cohort_identity_sha256"]
        == online["cohort_identity_sha256"]
    )
    checks["online_scene_count"] = (
        len(seed_manifest["scene_seeds"]) == online["scene_count"]
    )
    checks["online_failure_count"] = (
        seed_manifest["control_outcome_strata"].count("control_failure")
        == online["confirmed_control_failure_count"]
    )
    checks["online_success_count"] = (
        seed_manifest["control_outcome_strata"].count("control_success")
        == online["confirmed_control_success_count"]
    )

    root = output_root(config)
    feature_path = root / "features/features_action_expert_hidden_tail.npz"
    extraction_status = "pending"
    if feature_path.exists():
        manifest = json.loads(
            (
                root
                / "features/features_action_expert_hidden_tail_manifest.json"
            ).read_text()
        )
        checks["z1_feature_dataset"] = (
            manifest["dataset"]["dataset_fingerprint"]
            == offline["dataset_fingerprint"]
        )
        checks["z1_feature_shape"] = manifest["feature_shape"] == [
            offline["sample_count"],
            config["feature"]["definition"]["raw_hidden_dimension"],
        ]
        extraction_status = "complete"

    evaluation_path = root / "offline/evaluations/Z1.json"
    checkpoint_path = root / "offline/checkpoints/Z1.pt"
    offline_status = "pending"
    if evaluation_path.exists() and checkpoint_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        checks["oof_sample_count"] = (
            evaluation["metrics"]["oof"]["sample_count"]
            == offline["sample_count"]
        )
        checks["oof_fold_identity"] = (
            evaluation["protocol"]["fold_identity_sha256"]
            == offline["fold_identity_sha256"]
        )
        checks["full_training_sample_count"] = (
            evaluation["protocol"]["final_training_sample_count"]
            == offline["sample_count"]
        )
        checks["router_parameters"] = (
            evaluation["trainable_parameters"]
            == config["feature"]["expected_router_parameters"]
        )
        offline_status = "complete"
    elif evaluation_path.exists() or checkpoint_path.exists():
        raise RuntimeError("Partial Z1 offline artifacts")

    summary_path = root / "online/online_feature_candidates_summary.json"
    online_status = "complete" if summary_path.exists() else "pending"
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Z1 contract failed: {failed}")
    return {
        "pipeline_id": config["pipeline_id"],
        "status": "valid",
        "checks_passed": len(checks),
        "feature_extraction": extraction_status,
        "offline_result_1": offline_status,
        "online_result_2": online_status,
        "output_root": str(root),
    }


def result_payload(config: dict[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    evaluation_path = root / "offline/evaluations/Z1.json"
    online_path = root / "online/online_feature_candidates_summary.json"
    payload: dict[str, Any] = {
        "pipeline_id": config["pipeline_id"],
        "feature": "Z1",
        "result_1": {"status": "pending"},
        "result_2": {"status": "pending"},
    }
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        metrics = evaluation["metrics"]["oof"]
        payload["result_1"] = {
            "status": "complete",
            "parameters": evaluation["trainable_parameters"],
            "sample_count": metrics["sample_count"],
            "roc_auc": metrics["roc_auc"],
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "strong_replan_recall": metrics["strong_replan_recall"],
            "false_trigger_rate": metrics["false_trigger_rate"],
            "precision_replan": metrics["precision_replan"],
        }
    if online_path.exists():
        summary = json.loads(online_path.read_text())
        candidate = summary["candidate_summaries"]["Z1"]
        failure = candidate["by_control_stratum"]["control_failure"]
        success = candidate["by_control_stratum"]["control_success"]
        control_matches = (
            failure["control_outcome_matches_selection"]
            + success["control_outcome_matches_selection"]
        )
        payload["result_2"] = {
            "status": "complete",
            "valid_pairs": candidate["valid_pairs"],
            "control_outcome_match_rate": (
                control_matches / candidate["valid_pairs"]
            ),
            "control_failure_router_success_rate": failure[
                "router_success_rate"
            ],
            "control_failure_rescues": failure["rescues"],
            "control_success_router_success_rate": success[
                "router_success_rate"
            ],
            "control_success_harms": success["harms"],
            "overall_router_success_rate": (
                candidate["router_successes"] / candidate["valid_pairs"]
            ),
            "trigger_rate": (
                candidate["triggered_pairs"] / candidate["valid_pairs"]
            ),
        }
    return payload


def write_results(config: dict[str, Any], payload: dict[str, Any]) -> None:
    if (
        payload["result_1"]["status"] != "complete"
        or payload["result_2"]["status"] != "complete"
    ):
        raise RuntimeError("Refusing to freeze incomplete Z1 results")
    path = output_root(config) / "z1_result_1_result_2.json"
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise FileExistsError(f"Incompatible result file: {path}")
    if not path.exists():
        path.write_text(encoded)


def run_command(command: list[str], execute: bool) -> None:
    print(shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate",
            "extract",
            "offline-train",
            "online-prepare",
            "online-run",
            "summarize",
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.stage == "validate":
        print(json.dumps(validate(config), indent=2))
    elif args.stage == "extract":
        run_command(extract_command(config), args.execute)
    elif args.stage == "offline-train":
        run_command(offline_command(config), args.execute)
    elif args.stage == "online-prepare":
        run_command(online_command(config, prepare_only=True), args.execute)
    elif args.stage == "online-run":
        run_command(online_command(config, prepare_only=False), args.execute)
    else:
        payload = result_payload(config)
        print(json.dumps(payload, indent=2))
        if args.execute:
            write_results(config, payload)


if __name__ == "__main__":
    main()
