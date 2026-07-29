"""Union concrete replan nodes across immutable selection rounds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, action="append", required=True,
        help="Node manifest to merge. Repeat for every selection round.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nodes_by_seed: dict[int, set[int]] = defaultdict(set)
    r0_by_seed: dict[int, set[int]] = defaultdict(set)
    successful_rs_by_seed: dict[int, set[int]] = defaultdict(set)
    provenance: dict[int, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_scene_counts = {}
    source_node_counts = {}
    for path in args.input:
        payload = json.loads(path.read_text())
        source_scene_counts[str(path)] = len(payload["scenes"])
        source_node_counts[str(path)] = sum(
            len(scene["replan_nodes"]) for scene in payload["scenes"]
        )
        for scene in payload["scenes"]:
            seed = int(scene["scene_seed"])
            r0_by_seed[seed].add(int(scene["r0"]))
            successful_rs_by_seed[seed].update(
                int(r) for r in scene.get("successful_rs", [])
            )
            for node in scene["replan_nodes"]:
                node = int(node)
                nodes_by_seed[seed].add(node)
                provenance[seed][node].append(str(path))

    if not nodes_by_seed:
        raise ValueError("No scenes found in input manifests")
    inconsistent = {
        seed: sorted(values)
        for seed, values in r0_by_seed.items()
        if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"Inconsistent r0 values: {inconsistent}")

    scenes = []
    for seed in sorted(nodes_by_seed):
        nodes = sorted(nodes_by_seed[seed])
        scenes.append(
            {
                "scene_seed": seed,
                "r0": next(iter(r0_by_seed[seed])),
                "successful_rs": sorted(successful_rs_by_seed[seed]),
                "replan_nodes": nodes,
                "node_count": len(nodes),
                "node_provenance": [
                    {
                        "replan_node": node,
                        "source_manifests": provenance[seed][node],
                    }
                    for node in nodes
                ],
            }
        )

    output = {
        "purpose": "Union of concrete single-replan nodes across selection rounds.",
        "source_manifests": [str(path) for path in args.input],
        "source_scene_counts": source_scene_counts,
        "source_node_counts": source_node_counts,
        "scene_count": len(scenes),
        "total_nodes": sum(scene["node_count"] for scene in scenes),
        "deduplicated_nodes": (
            sum(source_node_counts.values())
            - sum(scene["node_count"] for scene in scenes)
        ),
        "scenes": scenes,
    }
    encoded = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists():
        if args.output.read_text() != encoded:
            raise FileExistsError(
                f"Refusing to overwrite incompatible output: {args.output}"
            )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(
        f"Merged {len(args.input)} manifests: {len(scenes)} scenes, "
        f"{output['total_nodes']} unique nodes"
    )
    print(args.output)


if __name__ == "__main__":
    main()
