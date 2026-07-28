"""Coordinator for the v2 common-cohort router feature-selection pipeline."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/replan_router_feature_selection_v2.json"
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
            f"Hash mismatch for {path}\nexpected={expected}\nactual={actual}"
        )


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2 or payload.get("status") != "active":
        raise ValueError("Expected the active schema-v2 feature-selection contract")
    return payload


def immutable_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite incompatible file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def feature_configs(config: dict[str, Any]) -> list[str]:
    return [str(value) for value in config["offline"]["feature_configs"]]


def output_root(config: dict[str, Any]) -> Path:
    return project_path(config["artifacts"]["output_root"])


def validate(config: dict[str, Any]) -> dict[str, Any]:
    offline = config["offline"]
    online = config["online"]
    artifacts = config["artifacts"]
    checks: dict[str, bool] = {}

    source_dataset_path = project_path(artifacts["source_dataset_manifest"])
    require_hash(
        source_dataset_path, artifacts["source_dataset_manifest_sha256"]
    )
    source_dataset = json.loads(source_dataset_path.read_text())
    checks["dataset_fingerprint"] = (
        source_dataset["dataset"]["dataset_fingerprint"]
        == offline["dataset_fingerprint"]
    )
    checks["dataset_sample_count"] = (
        source_dataset["dataset"]["sample_count"] == offline["sample_count"]
    )
    checks["dataset_scene_count"] = (
        source_dataset["dataset"]["scene_count"] == offline["scene_count"]
    )
    checks["dataset_transition_counts"] = (
        source_dataset["dataset"]["transition_counts"]
        == offline["transition_counts"]
    )

    require_hash(
        project_path(artifacts["base_feature_archive"]),
        artifacts["base_feature_archive_sha256"],
    )
    checks["base_feature_archive_hash"] = True
    require_hash(
        project_path(artifacts["base_feature_manifest"]),
        artifacts["base_feature_manifest_sha256"],
    )
    checks["base_feature_manifest_hash"] = True
    oof_manifest_path = project_path(artifacts["prepared_oof_manifest"])
    require_hash(oof_manifest_path, artifacts["prepared_oof_manifest_sha256"])
    oof_manifest = json.loads(oof_manifest_path.read_text())
    checks["oof_fold_identity"] = (
        oof_manifest["fold_identity_sha256"]
        == offline["oof_protocol"]["fold_identity_sha256"]
    )
    checks["oof_test_covers_348"] = (
        sum(
            fold["sample_counts"]["test"]
            for fold in oof_manifest["fold_protocol"]["fold_reports"]
        )
        == offline["sample_count"]
    )

    cohort = online["cohort"]
    for name in ("seed_manifest", "resolved_manifest", "audit_manifest"):
        require_hash(
            project_path(cohort[name]),
            cohort[f"{name}_sha256"],
        )
        checks[f"{name}_hash"] = True
    seed_manifest = json.loads(project_path(cohort["seed_manifest"]).read_text())
    audit = json.loads(project_path(cohort["audit_manifest"]).read_text())
    checks["cohort_identity"] = (
        seed_manifest["cohort_identity_sha256"]
        == cohort["cohort_identity_sha256"]
    )
    checks["cohort_scene_count"] = (
        len(seed_manifest["scene_seeds"]) == cohort["scene_count"]
    )
    checks["cohort_failure_count"] = (
        seed_manifest["control_outcome_strata"].count("control_failure")
        == cohort["confirmed_control_failure_count"]
    )
    checks["cohort_success_count"] = (
        seed_manifest["control_outcome_strata"].count("control_success")
        == cohort["confirmed_control_success_count"]
    )
    checks["cohort_offline_unseen"] = bool(
        audit["selected_disjoint_from_offline_scenes"]
    )
    for index, source in enumerate(cohort["control_sources"]):
        require_hash(project_path(source["summary"]), source["summary_sha256"])
        require_hash(
            project_path(source["resolved_manifest"]),
            source["resolved_manifest_sha256"],
        )
        checks[f"control_source_{index}_hashes"] = True

    batch_features = [
        name
        for batch in online["batches"]
        for name in batch["features"]
    ]
    checks["each_feature_online_once"] = (
        sorted(batch_features) == sorted(feature_configs(config))
        and len(batch_features) == len(set(batch_features))
    )

    offline_root = output_root(config) / "offline"
    completed = []
    partial = []
    for name in feature_configs(config):
        evaluation_path = offline_root / "evaluations" / f"{name}.json"
        checkpoint_path = offline_root / "checkpoints" / f"{name}.pt"
        if evaluation_path.exists() and checkpoint_path.exists():
            evaluation = json.loads(evaluation_path.read_text())
            checks[f"{name}_oof_n"] = (
                evaluation["metrics"]["oof"]["sample_count"]
                == offline["sample_count"]
            )
            checks[f"{name}_dataset"] = (
                evaluation["dataset"]["dataset_fingerprint"]
                == offline["dataset_fingerprint"]
            )
            checks[f"{name}_folds"] = (
                evaluation["protocol"]["fold_identity_sha256"]
                == offline["oof_protocol"]["fold_identity_sha256"]
            )
            checks[f"{name}_full_train_n"] = (
                evaluation["protocol"]["final_training_sample_count"]
                == offline["sample_count"]
            )
            checks[f"{name}_full_train_transitions"] = (
                evaluation["protocol"]["final_training_transition_counts"]
                == offline["transition_counts"]
            )
            completed.append(name)
        elif evaluation_path.exists() or checkpoint_path.exists():
            partial.append(name)
    if partial:
        raise RuntimeError(f"Partial offline feature artifacts: {partial}")
    if completed and len(completed) != len(feature_configs(config)):
        missing = sorted(set(feature_configs(config)) - set(completed))
        raise RuntimeError(
            "Some feature configs are trained while others are missing: "
            f"{missing}"
        )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Feature-selection contract failed: {failed}")
    return {
        "pipeline_id": config["pipeline_id"],
        "status": "valid",
        "checks_passed": len(checks),
        "offline": {
            "samples": offline["sample_count"],
            "scenes": offline["scene_count"],
            "transition_counts": offline["transition_counts"],
            "oof_folds": offline["oof_protocol"]["folds"],
            "trained_features": completed,
        },
        "online": {
            "scenes": cohort["scene_count"],
            "control_failures": cohort["confirmed_control_failure_count"],
            "control_successes": cohort["confirmed_control_success_count"],
            "cohort_identity_sha256": cohort["cohort_identity_sha256"],
            "features": batch_features,
        },
    }


def common_offline_command(config: dict[str, Any]) -> list[str]:
    offline = config["offline"]
    training = offline["training"]
    artifacts = config["artifacts"]
    command = [
        str(OPENPI_PYTHON),
        str(
            PROJECT_ROOT
            / "robotwin/script/train_replan_router_feature_selection.py"
        ),
        "--task",
        config["task"]["name"],
        "--base-feature-archive",
        str(project_path(artifacts["base_feature_archive"])),
        "--base-feature-manifest",
        str(project_path(artifacts["base_feature_manifest"])),
        "--output-dir",
        str(output_root(config) / "offline"),
        "--folds",
        str(offline["oof_protocol"]["folds"]),
        "--split-seed",
        str(offline["oof_protocol"]["split_seed"]),
        "--train-seed",
        str(training["train_seed"]),
        "--horizon",
        str(config["task"]["horizon"]),
        "--action-dim",
        str(config["task"]["action_dim"]),
        "--lambda",
        str(training["lambda"]),
        "--temporal-k",
        str(training["temporal_k"]),
        "--eef-waypoints",
        str(training["eef_waypoints"]),
        "--gripper-indices",
        training["gripper_indices"],
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
    for root in offline["data_roots"]:
        command.extend(["--data-root", str(project_path(root))])
    return command


def cohort_command(config: dict[str, Any]) -> list[str]:
    cohort = config["online"]["cohort"]
    artifacts = config["artifacts"]
    command = [
        str(ROBOTWIN_PYTHON),
        str(PROJECT_ROOT / "robotwin/script/build_balanced_control_cohort.py"),
        "--output-dir",
        str(project_path(cohort["seed_manifest"]).parent),
        "--offline-dataset-manifest",
        str(project_path(artifacts["source_dataset_manifest"])),
        "--failure-count",
        str(cohort["confirmed_control_failure_count"]),
        "--success-count",
        str(cohort["confirmed_control_success_count"]),
    ]
    for source in cohort["control_sources"]:
        command.extend(
            [
                "--source",
                str(project_path(source["summary"])),
                str(project_path(source["resolved_manifest"])),
            ]
        )
    return command


def batch_config(config: dict[str, Any], batch_name: str) -> dict[str, Any]:
    for batch in config["online"]["batches"]:
        if batch["name"] == batch_name:
            return batch
    raise KeyError(batch_name)


def online_command(
    config: dict[str, Any],
    batch_name: str,
    *,
    prepare_only: bool,
) -> list[str]:
    batch = batch_config(config, batch_name)
    online = config["online"]
    cohort = online["cohort"]
    protocol = online["protocol"]
    root = output_root(config)
    command = [
        str(ROBOTWIN_PYTHON),
        str(
            PROJECT_ROOT
            / "robotwin/script/run_pi05_online_router_feature_candidates_eval.py"
        ),
        "--output-dir",
        str(root / "online" / batch_name),
        "--scene-manifest",
        str(project_path(cohort["seed_manifest"])),
        "--resolved-scene-manifest",
        str(project_path(cohort["resolved_manifest"])),
        "--scene-count",
        str(cohort["scene_count"]),
        "--task-name",
        config["task"]["name"],
        "--task-config",
        config["task"]["config"],
        "--horizon",
        str(config["task"]["horizon"]),
        "--r0",
        str(config["task"]["natural_replan_interval"]),
        "--query-interval",
        str(protocol["query_interval"]),
        "--query-max-action",
        str(protocol["query_max_action"]),
        "--router-max-replans",
        str(protocol["max_replans"]),
        "--router-lambda",
        str(protocol["lambda"]),
        "--gpus",
        protocol["gpus"],
        "--port-base",
        str(protocol["port_base"]),
        "--seed",
        str(protocol["policy_seed"]),
        "--feature-archive",
        str(project_path(config["artifacts"]["base_feature_archive"])),
        "--feature-manifest",
        str(project_path(config["artifacts"]["base_feature_manifest"])),
    ]
    if "reuse_control_from" in batch:
        command.extend(
            [
                "--control-source-plan",
                str(
                    root
                    / "online"
                    / batch["reuse_control_from"]
                    / "experiment_plan.json"
                ),
            ]
        )
    for name in batch["features"]:
        command.extend(
            [
                "--candidate",
                f"{name}={root / 'offline/checkpoints' / f'{name}.pt'}",
            ]
        )
    if prepare_only:
        command.append("--prepare-only")
    return command


def selected_batches(config: dict[str, Any], requested: str) -> list[str]:
    names = [batch["name"] for batch in config["online"]["batches"]]
    if requested == "all":
        return names
    if requested not in names:
        raise ValueError(f"Unknown batch {requested}; expected one of {names}")
    return [requested]


def run_commands(commands: list[list[str]], execute: bool) -> None:
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def result_payload(config: dict[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    offline_rows = []
    offline_complete = True
    for name in feature_configs(config):
        path = root / "offline/evaluations" / f"{name}.json"
        if not path.exists():
            offline_complete = False
            continue
        payload = json.loads(path.read_text())
        metrics = payload["metrics"]["oof"]
        offline_rows.append(
            {
                "feature": name,
                "modules": payload["feature_modules"],
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

    online_by_feature = {}
    online_complete = True
    for batch in config["online"]["batches"]:
        path = (
            root
            / "online"
            / batch["name"]
            / "online_feature_candidates_summary.json"
        )
        if not path.exists():
            online_complete = False
            continue
        summary = json.loads(path.read_text())
        for name in batch["features"]:
            candidate = summary["candidate_summaries"][name]
            failure = candidate["by_control_stratum"]["control_failure"]
            success = candidate["by_control_stratum"]["control_success"]
            valid_pairs = candidate["valid_pairs"]
            expected_pairs = config["online"]["cohort"]["scene_count"]
            if (
                valid_pairs != expected_pairs
                or failure["valid_pairs"]
                != config["online"]["cohort"][
                    "confirmed_control_failure_count"
                ]
                or success["valid_pairs"]
                != config["online"]["cohort"][
                    "confirmed_control_success_count"
                ]
            ):
                raise RuntimeError(f"Incomplete balanced online result for {name}")
            control_matches = (
                failure["control_outcome_matches_selection"]
                + success["control_outcome_matches_selection"]
            )
            online_by_feature[name] = {
                "feature": name,
                "valid_pairs": valid_pairs,
                "control_outcome_match_rate": control_matches / valid_pairs,
                "control_failure_router_success_rate": failure[
                    "router_success_rate"
                ],
                "control_failure_rescues": failure["rescues"],
                "control_success_router_success_rate": success[
                    "router_success_rate"
                ],
                "control_success_harms": success["harms"],
                "overall_router_success_rate": (
                    candidate["router_successes"] / valid_pairs
                ),
                "trigger_rate": candidate["triggered_pairs"] / valid_pairs,
            }
    if online_complete and set(online_by_feature) != set(feature_configs(config)):
        raise RuntimeError("Online results do not cover every feature exactly once")
    return {
        "pipeline_id": config["pipeline_id"],
        "offline_status": "complete" if offline_complete else "pending",
        "online_status": "complete" if online_complete else "pending",
        "result_1_offline_oof": offline_rows,
        "result_2_online_balanced40": [
            online_by_feature[name]
            for name in feature_configs(config)
            if name in online_by_feature
        ],
    }


def result_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['pipeline_id']}",
        "",
        "## Result 1: scene-grouped OOF",
        "",
        "|Feature|Params|N|AUC|F1|Strong Recall|False Trigger|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["result_1_offline_oof"]:
        lines.append(
            f"|{row['feature']}|{row['parameters']}|{row['sample_count']}|"
            f"{row['roc_auc']:.4f}|{row['f1']:.4f}|"
            f"{row['strong_replan_recall']:.4f}|"
            f"{row['false_trigger_rate']:.4f}|"
        )
    lines.extend(
        [
            "",
            "## Result 2: balanced online-40",
            "",
            "|Feature|Valid|Control Match|Fail→Success|Rescues|Success Retained|Harms|Trigger|",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["result_2_online_balanced40"]:
        lines.append(
            f"|{row['feature']}|{row['valid_pairs']}|"
            f"{row['control_outcome_match_rate']:.3f}|"
            f"{row['control_failure_router_success_rate']:.3f}|"
            f"{row['control_failure_rescues']}|"
            f"{row['control_success_router_success_rate']:.3f}|"
            f"{row['control_success_harms']}|{row['trigger_rate']:.3f}|"
        )
    lines.extend(
        [
            "",
            f"Offline status: {payload['offline_status']}.",
            f"Online status: {payload['online_status']}.",
        ]
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
            "offline-prepare",
            "offline-train",
            "cohort",
            "online-prepare",
            "online-run",
            "summarize",
        ),
    )
    parser.add_argument(
        "--batch",
        default="all",
        help="For online stages: all, batch_0, or batch_1.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run or write; without this flag, print commands/results only.",
    )
    args = parser.parse_args()
    config = load_config(args.config.resolve())

    if args.stage == "validate":
        print(json.dumps(validate(config), indent=2))
        return
    if args.stage == "offline-prepare":
        run_commands([[*common_offline_command(config), "--prepare-only"]], args.execute)
        return
    if args.stage == "offline-train":
        run_commands([common_offline_command(config)], args.execute)
        return
    if args.stage == "cohort":
        run_commands([cohort_command(config)], args.execute)
        return
    if args.stage in {"online-prepare", "online-run"}:
        batches = selected_batches(config, args.batch)
        commands = [
            online_command(
                config,
                batch,
                prepare_only=args.stage == "online-prepare",
            )
            for batch in batches
        ]
        run_commands(commands, args.execute)
        return

    payload = result_payload(config)
    markdown = result_markdown(payload)
    print(markdown, end="")
    if args.execute:
        if (
            payload["offline_status"] != "complete"
            or payload["online_status"] != "complete"
        ):
            raise RuntimeError("Refusing to freeze an incomplete final summary")
        root = output_root(config)
        immutable_json(root / "result_1_result_2_summary.json", payload)
        markdown_path = root / "result_1_result_2_summary.md"
        if markdown_path.exists() and markdown_path.read_text() != markdown:
            raise FileExistsError(f"Incompatible summary: {markdown_path}")
        if not markdown_path.exists():
            markdown_path.write_text(markdown)


if __name__ == "__main__":
    main()
