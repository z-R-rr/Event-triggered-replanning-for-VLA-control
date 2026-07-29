"""Freeze the four-shard `pick_dual_bottles` coarse-grid inputs.

This is deliberately a manifest-only step: it neither starts a policy server
nor runs a RoboTwin episode.  It makes the scene-to-worker allocation explicit
before any r25 control or forced rollout is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COARSE_NODES = (10, 15, 20, 30, 35, 40, 45, 55, 60, 65, 70)
GROUPS = (
    {"group": 0, "server_gpu": 0, "client_gpu": 1, "port": 8600},
    {"group": 1, "server_gpu": 0, "client_gpu": 1, "port": 8601},
    {"group": 2, "server_gpu": 2, "client_gpu": 3, "port": 8610},
    {"group": 3, "server_gpu": 2, "client_gpu": 3, "port": 8611},
)


def write_immutable(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite incompatible file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--resolved-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seeds = json.loads(args.seed_manifest.read_text())
    resolved = json.loads(args.resolved_manifest.read_text())
    scene_seeds = [int(value) for value in seeds["scene_seeds"]]
    if (
        seeds.get("task_name") != "pick_dual_bottles"
        or seeds.get("task_config") != "demo_clean"
        or resolved.get("task_name") != "pick_dual_bottles"
        or resolved.get("task_config") != "demo_clean"
        or [int(value) for value in resolved["scene_seeds"]] != scene_seeds
        or len(resolved["episode_instructions"]) != len(scene_seeds)
    ):
        raise ValueError("Incompatible pick_dual_bottles scene/prompt manifests")
    if len(scene_seeds) != 100 or len(set(scene_seeds)) != 100:
        raise ValueError("The frozen coarse grid requires exactly 100 unique scenes")

    prompt_by_seed = dict(
        zip(scene_seeds, resolved["episode_instructions"], strict=True)
    )
    topology_groups = []
    for spec in GROUPS:
        index = int(spec["group"])
        shard_seeds = scene_seeds[index * 25 : (index + 1) * 25]
        shard_dir = args.output_dir / "counterfactual_train" / "shards" / f"group_{index:02d}"
        shard_seed_path = shard_dir / "seed_manifest.json"
        shard_resolved_path = shard_dir / "resolved_episode_manifest.json"
        shard_nodes_path = shard_dir / "coarse_nodes.json"
        write_immutable(
            shard_seed_path,
            {
                "task_name": "pick_dual_bottles",
                "task_config": "demo_clean",
                "base_seed": int(seeds["base_seed"]),
                "purpose": "Frozen 25-scene shard for r25 shared control and coarse paired grid",
                "source_seed_manifest": str(args.seed_manifest.resolve()),
                "scene_seeds": shard_seeds,
            },
        )
        write_immutable(
            shard_resolved_path,
            {
                "task_name": "pick_dual_bottles",
                "task_config": "demo_clean",
                "source_seed_manifest": str(shard_seed_path.resolve()),
                "scene_seeds": shard_seeds,
                "episode_instructions": [prompt_by_seed[seed] for seed in shard_seeds],
            },
        )
        write_immutable(
            shard_nodes_path,
            {
                "task_name": "pick_dual_bottles",
                "task_config": "demo_clean",
                "r0": 25,
                "selection": "pre-registered task-agnostic coarse grid",
                "scenes": [
                    {"scene_seed": seed, "replan_nodes": list(COARSE_NODES)}
                    for seed in shard_seeds
                ],
            },
        )
        topology_groups.append(
            {
                **spec,
                "scene_count": len(shard_seeds),
                "control_episodes": len(shard_seeds),
                "forced_episodes": len(shard_seeds) * len(COARSE_NODES),
                "seed_manifest": str(shard_seed_path.resolve()),
                "seed_manifest_sha256": sha256_file(shard_seed_path),
                "resolved_manifest": str(shard_resolved_path.resolve()),
                "resolved_manifest_sha256": sha256_file(shard_resolved_path),
                "nodes": str(shard_nodes_path.resolve()),
                "nodes_sha256": sha256_file(shard_nodes_path),
            }
        )

    write_immutable(
        args.output_dir / "counterfactual_train" / "four_group_topology.json",
        {
            "task_name": "pick_dual_bottles",
            "task_config": "demo_clean",
            "horizon": 50,
            "r0": 25,
            "absolute_r0_cadence": True,
            "coarse_nodes": list(COARSE_NODES),
            "source_seed_manifest": str(args.seed_manifest.resolve()),
            "source_seed_manifest_sha256": sha256_file(args.seed_manifest),
            "source_resolved_manifest": str(args.resolved_manifest.resolve()),
            "source_resolved_manifest_sha256": sha256_file(args.resolved_manifest),
            "groups": topology_groups,
            "invariant": (
                "Each scene's one shared control and all eleven forced nodes remain "
                "within exactly one group shard."
            ),
        },
    )
    write_immutable(
        args.output_dir / "discovery" / "checker_static_audit.json",
        {
            "task_name": "pick_dual_bottles",
            "checker_source": "/home/ubuntu/Workspace/RoboTwin/envs/pick_dual_bottles.py",
            "bottle_1_target_xy": [-0.06, -0.105],
            "bottle_2_target_xy": [0.06, -0.105],
            "xy_epsilon_strict": 0.1,
            "required_functional_point_z_strictly_greater_than": 0.89,
            "both_bottles_required": True,
            "gripper_open_required": False,
            "audit_note": (
                "The checker is conjunction over both bottle target xy and z. "
                "Online checker-successes still require post-run semantic visual review."
            ),
        },
    )
    print(args.output_dir / "counterfactual_train" / "four_group_topology.json")


if __name__ == "__main__":
    main()
