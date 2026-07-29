"""Freeze a train-unseen online cohort balanced by confirmed control outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite incompatible file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def control_pairs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if "pairs" in summary:
        return summary["pairs"]
    candidates = summary.get("candidate_summaries", {})
    if not candidates:
        raise ValueError("Unsupported online summary schema")
    first_name = next(iter(candidates))
    return candidates[first_name]["pairs"]


def offline_scene_seeds(dataset_manifest: dict[str, Any]) -> set[int]:
    return {
        int(scene_seed)
        for split in dataset_manifest["split"]["splits"].values()
        for scene_seed in split["scene_seeds"]
    }


def source_records(
    summary_path: Path, resolved_path: Path
) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    resolved = json.loads(resolved_path.read_text())
    prompts = {
        int(scene_seed): str(prompt)
        for scene_seed, prompt in zip(
            resolved["scene_seeds"],
            resolved["episode_instructions"],
            strict=True,
        )
    }
    records = []
    for pair in control_pairs(summary):
        if pair.get("status") != "complete" or not pair.get("valid_pair"):
            continue
        checks = pair.get("checks", {})
        if not checks or not all(bool(value) for value in checks.values()):
            continue
        scene_seed = int(pair["scene_seed"])
        if scene_seed not in prompts:
            raise ValueError(f"Missing prompt for confirmed scene {scene_seed}")
        control_case_dir = Path(pair["control_case_dir"])
        metrics_path = control_case_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        records.append(
            {
                "scene_seed": scene_seed,
                "prompt": prompts[scene_seed],
                "control_success": bool(pair["control_success"]),
                "control_case_dir": str(control_case_dir.resolve()),
                "control_metrics": str(metrics_path.resolve()),
                "control_metrics_sha256": sha256_file(metrics_path),
                "source_summary": str(summary_path.resolve()),
                "source_summary_sha256": sha256_file(summary_path),
                "source_resolved_manifest": str(resolved_path.resolve()),
                "source_resolved_manifest_sha256": sha256_file(resolved_path),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline-dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("ONLINE_SUMMARY", "RESOLVED_MANIFEST"),
        required=True,
        help="Repeat for each confirmed-control pool.",
    )
    parser.add_argument("--failure-count", type=int, default=20)
    parser.add_argument("--success-count", type=int, default=20)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        parser.error("--output-dir must be absolute")

    dataset_path = args.offline_dataset_manifest.resolve()
    dataset = json.loads(dataset_path.read_text())
    train_scenes = offline_scene_seeds(dataset)
    records_by_seed: dict[int, dict[str, Any]] = {}
    for summary_raw, resolved_raw in args.source:
        for record in source_records(
            Path(summary_raw).resolve(), Path(resolved_raw).resolve()
        ):
            scene_seed = record["scene_seed"]
            if scene_seed in records_by_seed:
                previous = records_by_seed[scene_seed]
                if previous["control_success"] != record["control_success"]:
                    raise ValueError(
                        f"Conflicting control outcome for scene {scene_seed}"
                    )
                continue
            records_by_seed[scene_seed] = record

    leaked = sorted(set(records_by_seed) & train_scenes)
    if leaked:
        raise ValueError(f"Confirmed-control pool overlaps offline scenes: {leaked}")
    failures = sorted(
        (
            record
            for record in records_by_seed.values()
            if not record["control_success"]
        ),
        key=lambda record: record["scene_seed"],
    )
    successes = sorted(
        (
            record
            for record in records_by_seed.values()
            if record["control_success"]
        ),
        key=lambda record: record["scene_seed"],
    )
    if len(failures) < args.failure_count or len(successes) < args.success_count:
        raise ValueError(
            "Insufficient confirmed-control scenes: "
            f"failure={len(failures)}, success={len(successes)}"
        )
    selected = [
        *failures[: args.failure_count],
        *successes[: args.success_count],
    ]
    seeds = [record["scene_seed"] for record in selected]
    prompts = [record["prompt"] for record in selected]
    strata = [
        "control_failure" if not record["control_success"] else "control_success"
        for record in selected
    ]
    cohort_identity = hashlib.sha256(
        json.dumps(
            [
                {
                    "scene_seed": record["scene_seed"],
                    "prompt": record["prompt"],
                    "control_success": record["control_success"],
                    "control_metrics_sha256": record["control_metrics_sha256"],
                }
                for record in selected
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    output_dir = args.output_dir.resolve()
    seed_path = output_dir / "balanced_seed_manifest.json"
    resolved_path = output_dir / "balanced_resolved_episode_manifest.json"
    audit_path = output_dir / "balanced_control_cohort_audit.json"
    write_immutable(
        seed_path,
        {
            "task_name": dataset["dataset"]["task"],
            "task_config": "demo_clean",
            "purpose": (
                "Feature-selection online cohort: 20 confirmed control failures "
                "plus 20 confirmed control successes, all offline-train unseen"
            ),
            "selection_rule": (
                "Sort confirmed valid control scenes by seed independently "
                "within outcome stratum and take the requested prefix"
            ),
            "scene_seeds": seeds,
            "control_outcome_strata": strata,
            "cohort_identity_sha256": cohort_identity,
        },
    )
    write_immutable(
        resolved_path,
        {
            "task_name": dataset["dataset"]["task"],
            "task_config": "demo_clean",
            "source_seed_manifest": str(seed_path),
            "scene_seeds": seeds,
            "episode_instructions": prompts,
            "control_outcome_strata": strata,
            "cohort_identity_sha256": cohort_identity,
        },
    )
    write_immutable(
        audit_path,
        {
            "task_name": dataset["dataset"]["task"],
            "offline_dataset_manifest": str(dataset_path),
            "offline_dataset_manifest_sha256": sha256_file(dataset_path),
            "offline_scene_count": len(train_scenes),
            "confirmed_pool_scene_count": len(records_by_seed),
            "confirmed_pool_failure_count": len(failures),
            "confirmed_pool_success_count": len(successes),
            "selected_failure_count": args.failure_count,
            "selected_success_count": args.success_count,
            "selected_disjoint_from_offline_scenes": True,
            "selection_rule_uses_router_results": False,
            "cohort_identity_sha256": cohort_identity,
            "records": selected,
            "seed_manifest": str(seed_path),
            "seed_manifest_sha256": sha256_file(seed_path),
            "resolved_manifest": str(resolved_path),
            "resolved_manifest_sha256": sha256_file(resolved_path),
        },
    )
    print(seed_path)


if __name__ == "__main__":
    main()
