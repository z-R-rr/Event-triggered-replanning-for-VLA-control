"""RoboTwin client policy for an openpi WebSocket inference server.

This module intentionally has no openpi model/training imports. It runs in the
RoboTwin environment alongside SAPIEN and only needs the lightweight
``openpi-client`` package.
"""

import time
import math
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi_client import websocket_client_policy


def _as_bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
        raise ValueError(f"Unsupported boolean value: {value!r}")
    return bool(value)


class RemotePiPolicy:
    def __init__(
        self,
        host: str,
        port: int,
        pi0_step: int,
        intervention: str = "none",
        pause_steps: int = 0,
        force_replan_before_actions=None,
        dynamic_r: bool = False,
        dynamic_r_candidates=None,
        dynamic_r_calibration: str | None = None,
        dynamic_r_threshold: float = 0.75,
        absolute_r0_cadence: bool = False,
        router_enabled: bool = False,
        router_nodes_manifest: str | None = None,
        router_lambda: float = 0.05,
        router_max_replans: int = 1,
        router_min_replan_interval: int = 0,
    ):
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.pi0_step = pi0_step
        self.instruction = None
        self.inference_calls = 0
        self.first_inference_start = None
        self.action_traces = []
        self.trace_enabled = False
        self.intervention = intervention
        self.pause_steps = pause_steps
        self.previous_chunk = None
        self.previous_chunk_cursor = 0
        self.chunk_traces = []
        self.episode_seed = None
        self.episode_id = None
        self.csl_enabled = False
        self.csl_probe_interval = 10
        self.csl_compare_horizon = 8
        self.csl_probe_counter = 0
        self.observation_record_dir = None
        self.observation_record_actions = set()
        # Explicit action-index intervention for causal timing diagnostics.
        # It is never enabled in ordinary evaluations.
        self.force_replan_before_actions = {int(value) for value in (force_replan_before_actions or [])}
        self._forced_replan_before_actions_seen = set()
        # Default False preserves the validated behavior in which a forced
        # replan starts a fresh full-r chunk. True inserts one forced replan
        # while keeping later natural boundaries anchored at r0, 2*r0, ...
        self.absolute_r0_cadence = _as_bool(absolute_r0_cadence)
        self.dynamic_r = bool(dynamic_r)
        self.dynamic_r_candidates = sorted({int(value) for value in (dynamic_r_candidates or [pi0_step])})
        if self.pi0_step not in self.dynamic_r_candidates:
            self.dynamic_r_candidates.append(self.pi0_step)
            self.dynamic_r_candidates.sort()
        self.dynamic_r_threshold = float(dynamic_r_threshold)
        self.dynamic_r_calibration = None
        if dynamic_r_calibration:
            self.dynamic_r_calibration = json.loads(open(dynamic_r_calibration).read())
        if self.dynamic_r and self.dynamic_r_calibration is None:
            raise ValueError("pi05_dynamic_r requires pi05_dynamic_r_calibration")
        self.router_enabled = _as_bool(router_enabled)
        self.router_lambda = float(router_lambda)
        self.router_max_replans = int(router_max_replans)
        self.router_min_replan_interval = int(router_min_replan_interval)
        if not 0.0 < self.router_lambda < 1.0:
            raise ValueError("pi05_router_lambda must be strictly between 0 and 1")
        if self.router_max_replans < 1:
            raise ValueError("pi05_router_max_replans must be positive")
        if self.router_min_replan_interval < 0:
            raise ValueError(
                "pi05_router_min_replan_interval must be non-negative"
            )
        self.router_nodes_by_seed = {}
        if router_nodes_manifest:
            manifest = json.loads(Path(router_nodes_manifest).read_text())
            self.router_nodes_by_seed = {
                int(row["scene_seed"]): {
                    int(node) for node in row["candidate_nodes"]
                }
                for row in manifest["scenes"]
            }
        if self.router_enabled and not self.router_nodes_by_seed:
            raise ValueError(
                "pi05_router_enabled requires pi05_router_nodes_manifest"
            )
        self.router_candidate_nodes = set()
        self.router_replans = 0
        self.router_last_replan_action = None
        self.router_queries = []

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def _request_payload(self, observation: dict) -> dict:
        if self.instruction is None:
            raise RuntimeError("Language instruction has not been set.")
        images = observation["observation"]
        state = observation["joint_action"]["vector"]
        return {
            "state": np.asarray(state, dtype=np.float32),
            "images": {
                "cam_high": np.ascontiguousarray(np.transpose(images["head_camera"]["rgb"], (2, 0, 1))),
                "cam_left_wrist": np.ascontiguousarray(np.transpose(images["left_camera"]["rgb"], (2, 0, 1))),
                "cam_right_wrist": np.ascontiguousarray(np.transpose(images["right_camera"]["rgb"], (2, 0, 1))),
            },
            "prompt": self.instruction,
        }

    def _request_actions(self, observation: dict, *, shadow_probe_id: int | None = None) -> np.ndarray:
        request = self._request_payload(observation)
        if self.episode_seed is not None:
            request["episode_seed"] = self.episode_seed
        if self.episode_id is not None:
            request["episode_id"] = self.episode_id
        if shadow_probe_id is not None:
            request["shadow_probe_id"] = int(shadow_probe_id)
        return self.client.infer(request)["actions"]

    def score_router(
        self,
        observation: dict,
        old_action_chunk: np.ndarray,
        old_chunk_cursor: int,
        completed_actions: int,
    ) -> dict:
        """Query the frozen router without sampling actions or advancing VLA RNG."""
        request = self._request_payload(observation)
        request.update(
            {
                "router_query": True,
                "old_action_chunk": np.asarray(
                    old_action_chunk, dtype=np.float32
                ),
                "old_chunk_cursor": int(old_chunk_cursor),
                "completed_actions": int(completed_actions),
                "router_lambda": self.router_lambda,
                "natural_replan_interval": int(self.initial_pi0_step),
                "last_replan_action": self.router_last_replan_action,
            }
        )
        if self.episode_seed is not None:
            request["episode_seed"] = self.episode_seed
        if self.episode_id is not None:
            request["episode_id"] = self.episode_id
        result = self.client.infer(request)["router"]
        if int(result["completed_actions"]) != int(completed_actions):
            raise RuntimeError("Router response action clock mismatch")
        return result

    def infer(self, observation: dict) -> np.ndarray:
        if self.first_inference_start is None:
            self.first_inference_start = time.perf_counter()
        self.inference_calls += 1
        return self._request_actions(observation)

    def shadow_infer(self, observation: dict) -> np.ndarray:
        """Counterfactual VLA chunk; does not alter the executed RNG stream."""
        self.csl_probe_counter += 1
        return self._request_actions(observation, shadow_probe_id=self.csl_probe_counter)

    def reset(self) -> None:
        self.instruction = None
        self.inference_calls = 0
        self.first_inference_start = None
        self.action_traces = []
        self.previous_chunk = None
        self.previous_chunk_cursor = 0
        self.chunk_traces = []
        self.episode_seed = None
        self.episode_id = None
        self.csl_probe_counter = 0
        self._forced_replan_before_actions_seen = set()
        self.router_candidate_nodes = set()
        self.router_replans = 0
        self.router_last_replan_action = None
        self.router_queries = []
        # pi0_step intentionally resets to the configured initial candidate;
        # it may have changed during the preceding episode.
        self.pi0_step = self.initial_pi0_step

    def set_trace_enabled(self, enabled: bool) -> None:
        """Enable compact per-action execution traces for diagnostic replays."""
        self.trace_enabled = enabled

    def set_episode_seed(self, seed: int) -> None:
        self.episode_seed = int(seed)
        self.router_candidate_nodes = set(
            self.router_nodes_by_seed.get(self.episode_seed, set())
        )
        if self.router_enabled and not self.router_candidate_nodes:
            raise ValueError(
                f"No router candidate nodes for scene seed {self.episode_seed}"
            )

    def set_episode_id(self, episode_id: int) -> None:
        """Set an occurrence token; it resets server call indexing, not RNG entropy."""
        self.episode_id = int(episode_id)

    def configure_csl(self, enabled: bool, probe_interval: int, compare_horizon: int) -> None:
        self.csl_enabled = bool(enabled)
        self.csl_probe_interval = int(probe_interval)
        self.csl_compare_horizon = int(compare_horizon)

    def set_observation_record_dir(self, directory: str | None) -> None:
        """Save raw policy inputs at primary inference boundaries when enabled."""
        self.observation_record_dir = Path(directory) if directory else None

    def set_observation_record_actions(self, actions) -> None:
        self.observation_record_actions = {int(action) for action in (actions or [])}


def get_model(usr_args):
    model = RemotePiPolicy(
        host=usr_args.get("server_host", "127.0.0.1"),
        port=int(usr_args.get("server_port", 8000)),
        pi0_step=int(usr_args["pi0_step"]),
        intervention=usr_args.get("pi05_intervention", "none"),
        pause_steps=int(usr_args.get("pi05_pause_steps", 0)),
        force_replan_before_actions=usr_args.get("pi05_force_replan_before_actions", []),
        dynamic_r=usr_args.get("pi05_dynamic_r", False),
        dynamic_r_candidates=usr_args.get("pi05_dynamic_r_candidates", []),
        dynamic_r_calibration=usr_args.get("pi05_dynamic_r_calibration"),
        dynamic_r_threshold=float(usr_args.get("pi05_dynamic_r_threshold", 0.75)),
        absolute_r0_cadence=usr_args.get("pi05_absolute_r0_cadence", False),
        router_enabled=usr_args.get("pi05_router_enabled", False),
        router_nodes_manifest=usr_args.get("pi05_router_nodes_manifest"),
        router_lambda=float(usr_args.get("pi05_router_lambda", 0.05)),
        router_max_replans=int(usr_args.get("pi05_router_max_replans", 1)),
        router_min_replan_interval=int(
            usr_args.get("pi05_router_min_replan_interval", 0)
        ),
    )
    model.initial_pi0_step = model.pi0_step
    return model


def _phase(task_env):
    return task_env.get_policy_phase() if hasattr(task_env, "get_policy_phase") else "unknown"


def _execution_limit(model, completed_actions: int) -> int:
    """Actions to execute before the next natural absolute-r0 boundary."""
    if not model.absolute_r0_cadence:
        return int(model.pi0_step)
    if model.dynamic_r:
        raise ValueError("absolute r0 cadence is incompatible with dynamic r")
    cadence_r = int(model.initial_pi0_step)
    offset = int(completed_actions) % cadence_r
    return cadence_r if offset == 0 else cadence_r - offset


def _observation_fingerprint(observation):
    """Compact, non-reversible trace identity for repeatability checks."""
    digest = hashlib.sha256()
    state = np.ascontiguousarray(np.asarray(observation["joint_action"]["vector"], dtype=np.float32))
    digest.update(state.tobytes())
    for camera in ("head_camera", "left_camera", "right_camera"):
        image = np.ascontiguousarray(observation["observation"][camera]["rgb"])
        digest.update(image.tobytes())
    return digest.hexdigest()


def _record_primary_observation(model, observation, inference_call: int):
    """Persist the exact VLA input without putting image bytes in JSON traces."""
    directory = getattr(model, "observation_record_dir", None)
    if directory is None:
        return None
    episode_id = int(getattr(model, "episode_id", 0) or 0)
    path = Path(directory) / f"episode_{episode_id:03d}_call_{inference_call:03d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    images = observation["observation"]
    np.savez_compressed(
        path,
        state=np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        head_camera_rgb=np.asarray(images["head_camera"]["rgb"], dtype=np.uint8),
        left_camera_rgb=np.asarray(images["left_camera"]["rgb"], dtype=np.uint8),
        right_camera_rgb=np.asarray(images["right_camera"]["rgb"], dtype=np.uint8),
    )
    return str(path)


def _record_action_observation(model, observation, global_action: int):
    directory = getattr(model, "observation_record_dir", None)
    if directory is None:
        return None
    episode_id = int(getattr(model, "episode_id", 0) or 0)
    path = Path(directory) / f"episode_{episode_id:03d}_action_{global_action:03d}_after.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    images = observation["observation"]
    np.savez_compressed(
        path,
        state=np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        head_camera_rgb=np.asarray(images["head_camera"]["rgb"], dtype=np.uint8),
        left_camera_rgb=np.asarray(images["left_camera"]["rgb"], dtype=np.uint8),
        right_camera_rgb=np.asarray(images["right_camera"]["rgb"], dtype=np.uint8),
    )
    return str(path)


def _gripper_contact_summary(task_env):
    """Compact, task-agnostic contact telemetry for replay/eval diagnostics."""
    gripper_links = set(getattr(getattr(task_env, "robot", None), "gripper_name", []))
    pairs = []
    for contact in task_env.scene.get_contacts():
        name0 = contact.bodies[0].entity.name
        name1 = contact.bodies[1].entity.name
        if name0 not in gripper_links and name1 not in gripper_links:
            continue
        impulse = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
        pairs.append({
            "gripper_link": name0 if name0 in gripper_links else name1,
            "other_actor": name1 if name0 in gripper_links else name0,
            "points": len(contact.points),
            "impulse_l2_sum": impulse,
        })
    return {
        "pair_count": len(pairs),
        "point_count": sum(pair["points"] for pair in pairs),
        "impulse_l2_sum": sum(pair["impulse_l2_sum"] for pair in pairs),
        "pairs": pairs,
    }


def _chunk_consistency(old_chunk, old_cursor, new_chunk):
    """Joint/gripper agreement of old unexecuted tail versus a new replan.

    Actions are qpos targets, so Cartesian fields are populated by the evaluator
    when its optional FK probe is available.  This routine is deliberately
    state-free and therefore cannot leak oracle phase to the policy.
    """
    if old_chunk is None or old_cursor >= len(old_chunk):
        return None
    overlap = min(len(old_chunk) - old_cursor, len(new_chunk))
    old = np.asarray(old_chunk[old_cursor:old_cursor + overlap])
    new = np.asarray(new_chunk[:overlap])
    # Aloha action layout: left arm, left gripper, right arm, right gripper.
    left_dim = (old.shape[1] - 2) // 2
    right_start = left_dim + 1
    joint = np.concatenate([old[:, :left_dim] - new[:, :left_dim], old[:, right_start:-1] - new[:, right_start:-1]], axis=1)
    old_gripper = np.stack([old[:, left_dim], old[:, -1]], axis=1)
    new_gripper = np.stack([new[:, left_dim], new[:, -1]], axis=1)
    return {
        "overlap_actions": int(overlap),
        "joint_target_l2_mean": float(np.linalg.norm(joint, axis=1).mean()),
        "joint_target_l2_max": float(np.linalg.norm(joint, axis=1).max()),
        "gripper_sign_agreement": float(np.mean((old_gripper >= 0.5) == (new_gripper >= 0.5))),
        "gripper_abs_delta_mean": float(np.abs(old_gripper - new_gripper).mean()),
    }


def _csl_probe(old_chunk, cursor, fresh_chunk, compare_horizon, execution, source):
    """One label-free estimate of whether an old chunk is still surviving.

    The score compares the old unexecuted tail to a counterfactual VLA chunk
    at the *current* observation.  A TOPP rejection makes the chunk invalid
    independently of action agreement, because its arm target was not applied.
    """
    consistency = _chunk_consistency(old_chunk, cursor, fresh_chunk)
    if consistency is None:
        return None
    overlap = min(int(consistency["overlap_actions"]), int(compare_horizon))
    old = np.asarray(old_chunk[cursor:cursor + overlap])
    fresh = np.asarray(fresh_chunk[:overlap])
    left_dim = (old.shape[1] - 2) // 2
    right_start = left_dim + 1
    joint = np.concatenate([old[:, :left_dim] - fresh[:, :left_dim], old[:, right_start:-1] - fresh[:, right_start:-1]], axis=1)
    gripper_disagreement = np.mean(
        (old[:, [left_dim, -1]] >= 0.5) != (fresh[:, [left_dim, -1]] >= 0.5)
    )
    topp_failed = not execution.get("topp_left_success", True) or not execution.get("topp_right_success", True)
    return {
        "source": source,
        "probe_offset": int(cursor),
        "compare_horizon": int(overlap),
        "joint_tail_l2": float(np.linalg.norm(joint, axis=1).mean()),
        "gripper_disagreement": float(gripper_disagreement),
        "controller_rejected": bool(topp_failed),
    }


def _quat_angle_degrees(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cosine = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def _ee_poses(task_env):
    return {
        arm: np.asarray(task_env.get_arm_pose(arm), dtype=np.float64)
        for arm in ("left", "right")
    }


def _pose_delta(before, after):
    translations = []
    rotations = []
    for arm in ("left", "right"):
        translations.append(float(np.linalg.norm(after[arm][:3] - before[arm][:3])))
        rotations.append(_quat_angle_degrees(before[arm][3:], after[arm][3:]))
    return {
        "translation_delta_m_mean": float(np.mean(translations)),
        "translation_delta_m_max": float(np.max(translations)),
        "rotation_delta_deg_mean": float(np.mean(rotations)),
        "rotation_delta_deg_max": float(np.max(rotations)),
    }


def _calibrated_z(model, field, value):
    stats = model.dynamic_r_calibration["features"][field]
    return float(np.clip(
        (float(value) - float(stats["median"])) / max(float(stats["iqr"]), float(stats["min_scale"])),
        -float(stats["clip"]), float(stats["clip"]),
    ))


def _dynamic_progress_features(model):
    """Features of the preceding executed window, with no future state."""
    if not model.chunk_traces:
        return None
    chunk = model.chunk_traces[-1]
    call = int(chunk["inference_call"])
    rows = [row for row in model.action_traces if int(row["inference_call"]) == call and row.get("effective_action_executed", 1)]
    targets = []
    chunk_actions = chunk.get("chunk_actions", [])
    for row in rows:
        index = int(row["chunk_action_index"])
        if index < len(chunk_actions):
            action = np.asarray(chunk_actions[index], dtype=np.float64)
            if action.size >= 14:
                targets.append(action[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]])
    efficiency = persistence = reversal_rate = 0.0
    if len(targets) >= 2:
        target_array = np.stack(targets)
        deltas = np.diff(target_array, axis=0)
        path_length = float(np.linalg.norm(deltas, axis=1).sum())
        net = float(np.linalg.norm(target_array[-1] - target_array[0]))
        efficiency = net / path_length if path_length > 0 else 0.0
        max_displacement = float(np.linalg.norm(target_array - target_array[0], axis=1).max())
        persistence = net / max_displacement if max_displacement > 0 else 0.0
        if len(deltas) >= 2:
            valid = (np.linalg.norm(deltas[:-1], axis=1) > 0) & (np.linalg.norm(deltas[1:], axis=1) > 0)
            if np.any(valid):
                reversal_rate = float(np.mean(np.sum(deltas[:-1] * deltas[1:], axis=1)[valid] < 0))
    stall = []
    for row in rows:
        cartesian = row.get("cartesian_delta") or {}
        if "translation_delta_m_mean" in cartesian and "rotation_delta_deg_mean" in cartesian:
            stall.append(
                float(cartesian["translation_delta_m_mean"]) < 0.001
                and float(cartesian["rotation_delta_deg_mean"]) < 1.0
            )
    return {
        "target_net_efficiency": efficiency,
        "target_persistence": persistence,
        "direction_reversal_rate": reversal_rate,
        "stall_rate": float(np.mean(stall)) if stall else 0.0,
    }


def _dynamic_r_decision(model):
    """Select execution length after the new chunk exposes current tail correction."""
    features = _dynamic_progress_features(model)
    if features is None:
        return {"enabled": True, "reason": "initial_window", "selected_r": int(model.pi0_step)}
    probes = [
        probe for probe in model.chunk_traces[-1].get("csl_probes", [])
        if not probe.get("controller_rejected", False)
    ]
    if not probes:
        return {"enabled": True, "reason": "no_valid_tail_probe", "selected_r": int(model.pi0_step)}
    probe = max(probes, key=lambda item: int(item["probe_offset"]))
    tail_rate = float(probe["joint_tail_l2"]) / max(int(probe["probe_offset"]), 1)
    z = {field: _calibrated_z(model, field, value) for field, value in features.items()}
    progress = z["target_net_efficiency"] + z["target_persistence"] - z["direction_reversal_rate"] - z["stall_rate"]
    z_tail = _calibrated_z(model, "tail_correction_rate", tail_rate)
    # Average the four progress components so it shares the z-score scale of
    # tail correction. High feedback need means poor progress and/or a large
    # observation-conditioned plan correction, so execute fewer actions.
    feedback_need = z_tail - progress / 4.0
    old_r = int(model.pi0_step)
    index = model.dynamic_r_candidates.index(old_r)
    if feedback_need > model.dynamic_r_threshold and index > 0:
        index = 0
        reason = "decrease_r_to_minimum"
    elif feedback_need < -model.dynamic_r_threshold and index + 1 < len(model.dynamic_r_candidates):
        index += 1
        reason = "increase_r"
    else:
        reason = "hold_r"
    model.pi0_step = int(model.dynamic_r_candidates[index])
    return {
        "enabled": True, "reason": reason, "previous_r": old_r, "selected_r": int(model.pi0_step),
        "tail_correction_rate": tail_rate, "z_tail_correction_rate": z_tail,
        **features,
        **{f"z_{key}": value for key, value in z.items()},
        "progress": progress, "feedback_need": feedback_need,
        "threshold": model.dynamic_r_threshold,
    }


def eval(task_env, model, observation):
    if model.instruction is None:
        model.set_language(task_env.get_instruction())
    phase_before_infer = _phase(task_env)
    observation_fingerprint = _observation_fingerprint(observation) if getattr(model, "trace_enabled", False) else None
    actions = np.asarray(model.infer(observation))
    observation_record_path = _record_primary_observation(model, observation, model.inference_calls)
    consistency = _chunk_consistency(model.previous_chunk, model.previous_chunk_cursor, actions)
    # A real replan boundary is also an informative CSL probe.  In particular
    # r<probe_interval has no in-chunk shadow point at all; comparing the old
    # tail to the next *primary* prediction gives it an observation without an
    # additional inference request or any perturbation of the RNG stream.
    # At interval-aligned boundaries the existing shadow probe already covers
    # the same offset, so avoid double-counting it.
    if (
        model.csl_enabled
        and model.trace_enabled
        and model.chunk_traces
        and model.previous_chunk is not None
        and model.previous_chunk_cursor > 0
        and model.previous_chunk_cursor % model.csl_probe_interval != 0
    ):
        boundary_probe = _csl_probe(
            model.previous_chunk,
            model.previous_chunk_cursor,
            actions,
            model.csl_compare_horizon,
            getattr(task_env, "last_take_action_trace", {}),
            source="primary_replan_boundary",
        )
        if boundary_probe is not None:
            model.chunk_traces[-1]["csl_probes"].append(boundary_probe)
    dynamic_r_decision = _dynamic_r_decision(model) if model.dynamic_r else None
    execution_limit = _execution_limit(model, task_env.take_action_cnt)
    if getattr(model, "trace_enabled", False):
        model.chunk_traces.append({
            "inference_call": model.inference_calls,
            "phase": phase_before_infer,
            "observation_fingerprint": observation_fingerprint,
            "observation_record_path": observation_record_path,
            "gripper_contact_at_inference": _gripper_contact_summary(task_env),
            "prompt_fingerprint": hashlib.sha256(model.instruction.encode("utf-8")).hexdigest(),
            "intervention": model.intervention,
            "previous_chunk_cursor": int(model.previous_chunk_cursor),
            "consistency": consistency,
            "csl_probes": [],
            "router_queries": [],
            # Filled after the first action: measured Cartesian continuity at
            # the old/new chunk boundary. It is safe because it reads only
            # current end-effector poses, unlike speculative FK mutation.
            "boundary_cartesian_delta": None,
            "chunk_actions": actions.tolist(),
            "dynamic_r_decision": dynamic_r_decision,
            "executed_r": execution_limit,
            "absolute_r0_cadence": bool(model.absolute_r0_cadence),
            "absolute_r0_next_boundary": (
                int(task_env.take_action_cnt) + execution_limit
                if model.absolute_r0_cadence
                else None
            ),
        })
    model.previous_chunk = actions.copy()
    model.previous_chunk_cursor = 0
    for action_index, action in enumerate(actions[:execution_limit]):
        next_global_action = int(task_env.take_action_cnt) + 1
        if (
            next_global_action in model.force_replan_before_actions
            and next_global_action not in model._forced_replan_before_actions_seen
        ):
            # Cut the current open-loop chunk immediately before the requested
            # action. The evaluator's next outer iteration obtains a fresh
            # current observation and issues the replacement primary chunk.
            model._forced_replan_before_actions_seen.add(next_global_action)
            if getattr(model, "trace_enabled", False):
                model.chunk_traces[-1].setdefault("forced_replan_before_actions", []).append(next_global_action)
            return
        phase_before_action = _phase(task_env)
        ee_before = _ee_poses(task_env) if getattr(model, "trace_enabled", False) else None
        before_actions = getattr(task_env, "effective_policy_actions", 0)
        before_scene_steps = getattr(task_env, "policy_scene_steps", 0)
        task_env.take_action(action)
        recorded_action_observation = None
        if next_global_action in getattr(model, "observation_record_actions", set()):
            recorded_action_observation = _record_action_observation(
                model, task_env.get_obs(), next_global_action
            )
        ee_after = _ee_poses(task_env) if getattr(model, "trace_enabled", False) else None
        model.previous_chunk_cursor += 1
        phase_after_action = _phase(task_env)
        phase_changed = phase_before_action != phase_after_action
        first_place = phase_before_action == "outbound_to_B" and phase_after_action == "return_to_A"
        execution = getattr(task_env, "last_take_action_trace", {})
        # Shadow probes are counterfactual only: they read the post-action
        # observation and use an isolated server RNG stream. They never alter
        # the chunk that is being executed.
        if (
            model.csl_enabled
            and model.previous_chunk_cursor % model.csl_probe_interval == 0
            and model.previous_chunk_cursor < len(actions)
            and not task_env.eval_success
            and task_env.take_action_cnt < task_env.step_lim
        ):
            fresh_actions = np.asarray(model.shadow_infer(task_env.get_obs()))
            probe = _csl_probe(
                actions,
                model.previous_chunk_cursor,
                fresh_actions,
                model.csl_compare_horizon,
                execution,
                source="shadow_in_chunk",
            )
            if probe is not None:
                model.chunk_traces[-1]["csl_probes"].append(probe)
        router_decision = None
        completed_actions = int(task_env.take_action_cnt)
        if (
            model.router_enabled
            and model.router_replans < model.router_max_replans
            and (
                model.router_last_replan_action is None
                or completed_actions - model.router_last_replan_action
                >= model.router_min_replan_interval
            )
            and completed_actions in model.router_candidate_nodes
            and not task_env.eval_success
            and completed_actions < task_env.step_lim
        ):
            router_observation = task_env.get_obs()
            router_decision = model.score_router(
                router_observation,
                actions,
                model.previous_chunk_cursor,
                completed_actions,
            )
            router_decision = {
                **router_decision,
                "observation_fingerprint": _observation_fingerprint(
                    router_observation
                ),
                "trigger_before_one_based_action": completed_actions + 1,
                "replans_before_query": model.router_replans,
                "last_replan_action": model.router_last_replan_action,
                "min_replan_interval": model.router_min_replan_interval,
            }
            model.router_queries.append(router_decision)
            if getattr(model, "trace_enabled", False):
                model.chunk_traces[-1]["router_queries"].append(
                    router_decision
                )
        if getattr(model, "trace_enabled", False):
            model.action_traces.append({
                "inference_call": model.inference_calls,
                "chunk_action_index": action_index,
                "action_l2": float(np.linalg.norm(action)),
                "action_abs_max": float(np.max(np.abs(action))),
                "cartesian_delta": _pose_delta(ee_before, ee_after),
                "effective_action_executed": int(getattr(task_env, "effective_policy_actions", 0) > before_actions),
                "scene_steps_delta": int(getattr(task_env, "policy_scene_steps", 0) - before_scene_steps),
                "phase_before": phase_before_action,
                "phase_after": phase_after_action,
                "phase_changed": phase_changed,
                "gripper_contact_after_action": _gripper_contact_summary(task_env),
                "observation_record_path_after_action": recorded_action_observation,
                "router_query": router_decision,
                **execution,
            })
            if action_index == 0:
                model.chunk_traces[-1]["boundary_cartesian_delta"] = _pose_delta(ee_before, ee_after)
        should_replan = (
            (model.intervention == "clear_after_first_place" and first_place)
            or (model.intervention == "pause_then_clear_after_first_place" and first_place)
            or (model.intervention == "oracle_replan_all_phase_changes" and phase_changed)
        )
        # Diagnostic control-loop probe: a TOPP failure means the robot arm
        # did not follow this qpos target (only the fallback gripper loop ran).
        # Continuing to consume the rest of that open-loop chunk is therefore
        # a stale-action commitment.  This uses only the controller result,
        # not task phase or H/r-specific logic.
        topp_failed = not execution.get("topp_left_success", True) or not execution.get("topp_right_success", True)
        if model.intervention == "replan_on_topp_failure" and topp_failed:
            should_replan = True
        router_triggered = bool(
            router_decision is not None and router_decision["trigger"]
        )
        if router_triggered:
            should_replan = True
            model.router_replans += 1
            model.router_last_replan_action = completed_actions
        if should_replan:
            pause_steps = model.pause_steps if model.intervention == "pause_then_clear_after_first_place" else 0
            if pause_steps:
                task_env.diagnostic_policy_pause(pause_steps)
            if getattr(model, "trace_enabled", False):
                model.action_traces[-1]["forced_replan"] = True
                model.action_traces[-1]["pause_scene_steps"] = pause_steps
                if model.intervention == "replan_on_topp_failure" and topp_failed:
                    model.action_traces[-1]["forced_replan_reason"] = "topp_failure"
                if router_triggered:
                    model.action_traces[-1]["forced_replan_reason"] = "router"
                    model.action_traces[-1][
                        "router_trigger_before_one_based_action"
                    ] = completed_actions + 1
                    model.chunk_traces[-1].setdefault(
                        "router_trigger_before_actions", []
                    ).append(completed_actions + 1)
            return
        if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
            break


def reset_model(model):
    model.reset()
