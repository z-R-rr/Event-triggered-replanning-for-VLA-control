"""Versioned offline-to-online replan-router pipeline coordinator.

Expensive or state-changing stages only print their exact command unless
``--execute`` is provided. Validation and summaries are always read-only.
"""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/replan_router_pipeline_v1.json"
OPENPI_PYTHON = WORKSPACE / "openpi/.venv/bin/python"
ROBOTWIN_PYTHON = WORKSPACE / "RoboTwin/.venv/bin/python"


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported pipeline schema: {payload.get('schema_version')}")
    if payload.get("status") == "superseded":
        raise ValueError(
            "This pipeline is superseded; use "
            f"{payload.get('superseded_by', 'the active feature-selection config')}"
        )
    return payload


def project_path(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, expected_sha256: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(
                f"Artifact hash mismatch: {path}\n"
                f"expected={expected_sha256}\nactual={actual}"
            )


def immutable_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite incompatible file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def evaluation_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    metrics = payload["metrics"]["test"]
    return {
        "feature_config": payload["feature_config"],
        "feature_modules": payload["feature_modules"],
        "trainable_parameters": int(payload["trainable_parameters"]),
        "test_sample_count": int(metrics["sample_count"]),
        "roc_auc": float(metrics["roc_auc"]),
        "accuracy": float(metrics["accuracy"]),
        "f1": float(metrics["f1"]),
        "strong_replan_recall": float(metrics["strong_replan_recall"]),
        "false_trigger_rate": float(metrics["false_trigger_rate"]),
        "precision_replan": float(metrics["precision_replan"]),
        "dataset_fingerprint": payload["dataset"]["dataset_fingerprint"],
        "split_identity_sha256": payload["split_identity_sha256"],
        "evaluation_path": str(path),
    }


def active_offline_root(config: dict[str, Any]) -> Path:
    """Use a new run only after every configured evaluation/checkpoint exists."""
    run_root = project_path(config["artifacts"]["run_root"]) / "offline"
    artifact_pairs = [
        (
            run_root / "evaluations" / f"{name}.json",
            run_root / "checkpoints" / f"{name}.pt",
        )
        for name in config["training"]["feature_configs"]
    ]
    states = [
        evaluation.is_file() and checkpoint.is_file()
        for evaluation, checkpoint in artifact_pairs
    ]
    if all(states):
        return run_root
    if any(
        evaluation.exists() or checkpoint.exists()
        for evaluation, checkpoint in artifact_pairs
    ):
        raise RuntimeError(
            f"Partial offline run at {run_root}; finish all configured modules "
            "before shortlist or online stages"
        )
    return project_path(config["artifacts"]["reference_offline_root"])


def offline_rows(
    config: dict[str, Any], root: Path | None = None
) -> list[dict[str, Any]]:
    root = active_offline_root(config) if root is None else root
    rows = []
    for name in config["training"]["feature_configs"]:
        path = root / "evaluations" / f"{name}.json"
        require_file(path)
        rows.append(evaluation_payload(path))
    return rows


def compute_shortlist(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    rule = config["offline_evaluation"]
    diagnostic = set(rule["diagnostic_only_configs"])
    comparable = [
        row
        for row in rows
        if row["feature_config"] not in diagnostic
        and row["test_sample_count"] == rule["comparable_test_sample_count"]
        and row["trainable_parameters"] <= rule["maximum_trainable_parameters"]
    ]
    comparable.sort(
        key=lambda row: (
            -row["f1"],
            row["false_trigger_rate"],
            -row["roc_auc"],
            row["trainable_parameters"],
            row["feature_config"],
        )
    )
    baseline = rule["baseline_config"]
    challengers = [
        row["feature_config"]
        for row in comparable
        if row["feature_config"] != baseline
    ][: rule["challenger_count"]]
    shortlist = [baseline, *challengers]
    return shortlist, comparable


def validate(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["artifacts"]
    offline_root = project_path(artifacts["reference_offline_root"])
    dataset_path = offline_root / "configs/dataset_manifest.json"
    require_file(dataset_path, artifacts["reference_dataset_manifest_sha256"])
    dataset = json.loads(dataset_path.read_text())

    expected_dataset = config["dataset"]
    expected_split = config["split"]
    actual_dataset = dataset["dataset"]
    checks = {
        "dataset_fingerprint": (
            actual_dataset["dataset_fingerprint"]
            == expected_dataset["expected_fingerprint"]
        ),
        "dataset_sample_count": (
            actual_dataset["sample_count"] == expected_dataset["expected_sample_count"]
        ),
        "dataset_scene_count": (
            actual_dataset["scene_count"] == expected_dataset["expected_scene_count"]
        ),
        "both_failure_excluded": (
            dataset["both_failure_excluded"] is expected_dataset["both_failure_excluded"]
        ),
        "split_identity": (
            dataset["split_identity_sha256"]
            == expected_split["expected_identity_sha256"]
        ),
    }
    for split_name, expected in expected_split["expected_counts"].items():
        actual = dataset["split"]["splits"][split_name]
        checks[f"{split_name}_sample_count"] = (
            actual["sample_count"] == expected["samples"]
        )
        checks[f"{split_name}_scene_count"] = actual["scene_count"] == expected["scenes"]

    rows = offline_rows(config, offline_root)
    for row in rows:
        name = row["feature_config"]
        require_file(
            offline_root / "evaluations" / f"{name}.json",
            artifacts["reference_evaluations_sha256"][name],
        )
        checks[f"{name}_evaluation_hash"] = True
        checks[f"{name}_dataset"] = (
            row["dataset_fingerprint"] == expected_dataset["expected_fingerprint"]
        )
        checks[f"{name}_split"] = (
            row["split_identity_sha256"]
            == expected_split["expected_identity_sha256"]
        )

    shortlist, comparable = compute_shortlist(config, rows)
    checks["shortlist"] = shortlist == config["offline_evaluation"]["expected_shortlist"]
    for name, expected_hash in artifacts["reference_checkpoints_sha256"].items():
        path = offline_root / "checkpoints" / f"{name}.pt"
        require_file(path, expected_hash)
        checks[f"{name}_checkpoint_hash"] = True
    require_file(
        project_path(artifacts["base_feature_archive"]),
        artifacts["base_feature_archive_sha256"],
    )
    checks["base_feature_archive_hash"] = True
    require_file(
        project_path(artifacts["base_feature_manifest"]),
        artifacts["base_feature_manifest_sha256"],
    )
    checks["base_feature_manifest_hash"] = True

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Pipeline contract failed: {', '.join(failed)}")
    return {
        "pipeline_id": config["pipeline_id"],
        "status": "valid",
        "checks_passed": len(checks),
        "dataset_fingerprint": expected_dataset["expected_fingerprint"],
        "split_identity_sha256": expected_split["expected_identity_sha256"],
        "comparable_configs": [row["feature_config"] for row in comparable],
        "diagnostic_only_configs": config["offline_evaluation"][
            "diagnostic_only_configs"
        ],
        "online_shortlist": shortlist,
    }


def common_train_command(config: dict[str, Any], feature_config: str) -> list[str]:
    task = config["task"]
    training = config["training"]
    artifacts = config["artifacts"]
    run_root = project_path(artifacts["run_root"])
    command = [
        str(OPENPI_PYTHON),
        str(PROJECT_ROOT / "robotwin/script/train_replan_router.py"),
        "--task",
        task["name"],
        "--feature-config",
        feature_config,
        "--feature-type",
        training["feature_type"],
        "--output-dir",
        str(run_root / "offline"),
        "--horizon",
        str(task["horizon"]),
        "--action-dim",
        str(task["action_dim"]),
        "--lambda",
        str(config["offline_evaluation"]["threshold"]),
        "--temporal-k",
        str(training["temporal_k"]),
        "--eef-waypoints",
        str(training["eef_waypoints"]),
        "--gripper-indices",
        training["gripper_indices"],
        "--split-seed",
        str(config["split"]["seed"]),
        "--train-seed",
        str(training["train_seed"]),
        "--train-batch-size",
        str(training["batch_size"]),
        "--epochs",
        str(training["max_epochs"]),
        "--patience",
        str(training["early_stopping_patience"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--weight-decay",
        str(training["weight_decay"]),
    ]
    for root in config["dataset"]["data_roots"]:
        command.extend(["--data-root", str(project_path(root))])
    return command


def dataset_command(config: dict[str, Any]) -> list[str]:
    return [*common_train_command(config, "V0"), "--dataset-only"]


def offline_commands(config: dict[str, Any]) -> list[list[str]]:
    artifacts = config["artifacts"]
    shared = [
        "--train-only",
        "--base-feature-archive",
        str(project_path(artifacts["base_feature_archive"])),
        "--base-feature-manifest",
        str(project_path(artifacts["base_feature_manifest"])),
    ]
    return [
        [*common_train_command(config, name), *shared]
        for name in config["training"]["feature_configs"]
    ]


def scene_selection_command(config: dict[str, Any]) -> list[str]:
    online = config["online_evaluation"]
    run_root = project_path(config["artifacts"]["run_root"])
    return [
        str(ROBOTWIN_PYTHON),
        str(PROJECT_ROOT / "robotwin/script/select_robotwin_unseen_scenes.py"),
        "--output-dir",
        str(run_root / "online/inputs"),
        "--task-name",
        config["task"]["name"],
        "--task-config",
        config["task"]["config"],
        "--count",
        str(online["scene_count"]),
        "--candidate-start",
        str(online["candidate_start"]),
        "--policy-seed",
        str(online["policy_seed"]),
        "--instruction-type",
        online["instruction_type"],
    ]


def online_command(config: dict[str, Any], *, prepare_only: bool) -> list[str]:
    online = config["online_evaluation"]
    artifacts = config["artifacts"]
    run_root = project_path(artifacts["run_root"])
    inputs = run_root / "online/inputs"
    output = run_root / "online/eval_interval5_max1"
    offline_root = active_offline_root(config)
    rows = offline_rows(config)
    shortlist, _ = compute_shortlist(config, rows)
    command = [
        str(ROBOTWIN_PYTHON),
        str(
            PROJECT_ROOT
            / "robotwin/script/run_pi05_online_router_feature_candidates_eval.py"
        ),
        "--output-dir",
        str(output),
        "--scene-manifest",
        str(inputs / "unseen_seed_manifest.json"),
        "--resolved-scene-manifest",
        str(inputs / "unseen_resolved_episode_manifest.json"),
        "--scene-count",
        str(online["scene_count"]),
        "--task-name",
        config["task"]["name"],
        "--task-config",
        config["task"]["config"],
        "--horizon",
        str(config["task"]["horizon"]),
        "--r0",
        str(config["task"]["natural_replan_interval"]),
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
        str(project_path(artifacts["base_feature_archive"])),
        "--feature-manifest",
        str(project_path(artifacts["base_feature_manifest"])),
    ]
    for name in shortlist:
        command.extend(
            [
                "--candidate",
                f"{name}={offline_root / 'checkpoints' / f'{name}.pt'}",
            ]
        )
    if prepare_only:
        command.append("--prepare-only")
    return command


def run_commands(commands: list[list[str]], execute: bool) -> None:
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def shortlist_payload(config: dict[str, Any]) -> dict[str, Any]:
    rows = offline_rows(config)
    shortlist, comparable = compute_shortlist(config, rows)
    return {
        "pipeline_id": config["pipeline_id"],
        "dataset_fingerprint": config["dataset"]["expected_fingerprint"],
        "split_identity_sha256": config["split"]["expected_identity_sha256"],
        "selection_uses_online_results": False,
        "ranking": config["offline_evaluation"]["ranking"],
        "baseline": config["offline_evaluation"]["baseline_config"],
        "shortlist": shortlist,
        "ranked_comparable_configs": [
            {
                key: row[key]
                for key in (
                    "feature_config",
                    "feature_modules",
                    "trainable_parameters",
                    "test_sample_count",
                    "roc_auc",
                    "f1",
                    "strong_replan_recall",
                    "false_trigger_rate",
                )
            }
            for row in comparable
        ],
        "diagnostic_only_configs": config["offline_evaluation"][
            "diagnostic_only_configs"
        ],
    }


def summary_payload(config: dict[str, Any]) -> dict[str, Any]:
    rows = offline_rows(config)
    shortlist, _ = compute_shortlist(config, rows)
    run_root = project_path(config["artifacts"]["run_root"])
    online_path = (
        run_root
        / "online/eval_interval5_max1/online_feature_candidates_summary.json"
    )
    online_result: dict[str, Any] = {
        "status": "pending",
        "summary_path": str(online_path),
    }
    if online_path.exists():
        raw = json.loads(online_path.read_text())
        candidates = {}
        for name in shortlist:
            result = raw["candidate_summaries"][name]
            all_checks_pass = all(
                pair["valid_pair"] and all(pair["checks"].values())
                for pair in result["pairs"]
            )
            valid = result["valid_pairs"] == config["online_evaluation"]["scene_count"]
            promoted = (
                valid
                and all_checks_pass
                and result["router_successes"] > result["control_successes"]
                and result["rescues"] > result["harms"]
            )
            candidates[name] = {
                key: result[key]
                for key in (
                    "valid_pairs",
                    "triggered_pairs",
                    "control_successes",
                    "router_successes",
                    "rescues",
                    "harms",
                    "both_success",
                    "both_failure",
                )
            }
            candidates[name]["all_pair_checks_pass"] = all_checks_pass
            candidates[name]["promoted"] = promoted
        online_result = {
            "status": "complete",
            "summary_path": str(online_path),
            "candidates": candidates,
        }
    return {
        "pipeline_id": config["pipeline_id"],
        "dataset_fingerprint": config["dataset"]["expected_fingerprint"],
        "split_identity_sha256": config["split"]["expected_identity_sha256"],
        "offline": rows,
        "shortlist": shortlist,
        "online": online_result,
    }


def summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['pipeline_id']}",
        "",
        "|Feature|Params|N|AUC|F1|Strong Recall|False Trigger|Role|",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    shortlist = set(payload["shortlist"])
    for row in payload["offline"]:
        role = "online" if row["feature_config"] in shortlist else "offline"
        if row["test_sample_count"] != 52:
            role = "diagnostic"
        lines.append(
            f"|{row['feature_config']}|{row['trainable_parameters']}|"
            f"{row['test_sample_count']}|{row['roc_auc']:.4f}|{row['f1']:.4f}|"
            f"{row['strong_replan_recall']:.4f}|"
            f"{row['false_trigger_rate']:.4f}|{role}|"
        )
    lines.extend(["", f"Online shortlist: `{', '.join(payload['shortlist'])}`.", ""])
    if payload["online"]["status"] == "pending":
        lines.append("Online status: pending.")
    else:
        lines.extend(
            [
                "|Arm|Valid|Control success|Router success|Rescue|Harm|Promoted|",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for name, result in payload["online"]["candidates"].items():
            lines.append(
                f"|{name}|{result['valid_pairs']}|"
                f"{result['control_successes']}|{result['router_successes']}|"
                f"{result['rescues']}|{result['harms']}|"
                f"{'yes' if result['promoted'] else 'no'}|"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate",
            "dataset",
            "offline",
            "shortlist",
            "select-scenes",
            "online-prepare",
            "online-run",
            "summarize",
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run/write the requested stage; otherwise print commands/results.",
    )
    args = parser.parse_args()
    config = load_config(args.config.resolve())

    if args.stage == "validate":
        print(json.dumps(validate(config), indent=2))
        return
    if args.stage == "dataset":
        run_commands([dataset_command(config)], args.execute)
        return
    if args.stage == "offline":
        run_commands(offline_commands(config), args.execute)
        return
    if args.stage == "shortlist":
        payload = shortlist_payload(config)
        print(json.dumps(payload, indent=2))
        if args.execute:
            run_root = project_path(config["artifacts"]["run_root"])
            immutable_json(run_root / "configs/online_shortlist.json", payload)
        return
    if args.stage == "select-scenes":
        run_commands([scene_selection_command(config)], args.execute)
        return
    if args.stage == "online-prepare":
        run_commands([online_command(config, prepare_only=True)], args.execute)
        return
    if args.stage == "online-run":
        run_commands([online_command(config, prepare_only=False)], args.execute)
        return

    payload = summary_payload(config)
    markdown = summary_markdown(payload)
    print(markdown, end="")
    if args.execute:
        run_root = project_path(config["artifacts"]["run_root"])
        immutable_json(run_root / "pipeline_summary.json", payload)
        summary_path = run_root / "pipeline_summary.md"
        if summary_path.exists() and summary_path.read_text() != markdown:
            raise FileExistsError(
                f"Refusing to overwrite incompatible file: {summary_path}"
            )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(markdown)


if __name__ == "__main__":
    main()
