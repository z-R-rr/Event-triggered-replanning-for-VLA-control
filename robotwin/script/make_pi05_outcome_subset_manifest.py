"""Build an immutable paired-scene subset from per-scene pi0.5 traces."""

import argparse
import json
from pathlib import Path


def write_new(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite incompatible output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--required-failure-r", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-resolved-manifest", type=Path)
    parser.add_argument("--output-resolved-manifest", type=Path)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text())
    source_seeds = [int(seed) for seed in source["scene_seeds"]]
    outcomes: dict[int, dict[int, bool]] = {}
    horizons: set[int] = set()
    for path in sorted(args.trace_root.glob("H*_r*/scene_*.json")):
        config = path.parent.name
        horizon_text, r_text = config.split("_r", maxsplit=1)
        horizon = int(horizon_text.removeprefix("H"))
        execute_steps = int(r_text)
        trace = json.loads(path.read_text())
        seed = int(trace["scene_seed"])
        success = bool(trace["episode_metrics"]["episode_success"])
        previous = outcomes.setdefault(seed, {}).setdefault(execute_steps, success)
        if previous != success:
            raise ValueError(f"Conflicting outcome for seed={seed}, r={execute_steps}")
        horizons.add(horizon)

    selected = []
    for scene_index, seed in enumerate(source_seeds, start=1):
        by_r = outcomes.get(seed, {})
        if by_r.get(args.required_failure_r) is not False:
            continue
        successful_rs = sorted(r for r, success in by_r.items() if success)
        if successful_rs:
            selected.append(
                {
                    "scene_index_in_source_manifest": scene_index,
                    "scene_seed": seed,
                    "required_failure_r": args.required_failure_r,
                    "successful_rs": successful_rs,
                    "failed_rs": sorted(r for r, success in by_r.items() if not success),
                }
            )

    payload = {
        "task_name": source["task_name"],
        "task_config": source["task_config"],
        "base_seed": int(source["base_seed"]),
        "purpose": (
            f"Paired scenes where r={args.required_failure_r} failed and at least one "
            "other evaluated r succeeded."
        ),
        "source_manifest": str(args.source_manifest),
        "source_trace_root": str(args.trace_root),
        "action_horizons": sorted(horizons),
        "required_failure_r": args.required_failure_r,
        "evaluated_rs": sorted({r for by_r in outcomes.values() for r in by_r}),
        "scene_seeds": [row["scene_seed"] for row in selected],
        "scene_selection": selected,
    }
    write_new(args.output, payload)

    if bool(args.source_resolved_manifest) != bool(args.output_resolved_manifest):
        raise ValueError("Provide both resolved-manifest arguments or neither.")
    if args.source_resolved_manifest:
        resolved = json.loads(args.source_resolved_manifest.read_text())
        prompt_by_seed = dict(
            zip(
                [int(seed) for seed in resolved["scene_seeds"]],
                resolved["episode_instructions"],
                strict=True,
            )
        )
        missing = [row["scene_seed"] for row in selected if row["scene_seed"] not in prompt_by_seed]
        if missing:
            raise ValueError(f"Resolved prompt missing for selected seeds: {missing}")
        write_new(
            args.output_resolved_manifest,
            {
                "task_name": source["task_name"],
                "task_config": source["task_config"],
                "source_seed_manifest": str(args.output),
                "source_resolved_manifest": str(args.source_resolved_manifest),
                "scene_seeds": [row["scene_seed"] for row in selected],
                "episode_instructions": [prompt_by_seed[row["scene_seed"]] for row in selected],
            },
        )

    print(f"Selected {len(selected)} scenes")
    print(args.output)
    if args.output_resolved_manifest:
        print(args.output_resolved_manifest)


if __name__ == "__main__":
    main()
