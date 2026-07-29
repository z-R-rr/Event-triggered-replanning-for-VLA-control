"""Serve frozen pi0.5 actions plus an offline-trained online router query.

Normal requests are delegated unchanged to the deterministic episode-seeded
pi0.5 policy. Requests carrying ``router_query=True`` extract frozen features
and score the router without sampling actions or advancing policy RNG state.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import tyro

from openpi.models import model as model_types
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


WORKSPACE = Path("/home/ubuntu/Workspace")
PROJECT_ROOT = WORKSPACE / "Event-triggered-replanning-for-VLA-control"
ROUTER_SCRIPT_DIR = PROJECT_ROOT / "robotwin/script"
if str(ROUTER_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_SCRIPT_DIR))

from train_replan_router import (  # noqa: E402
    AlohaOfflineForwardKinematics,
    FeatureAblationRouter,
    OutcomeRouter,
    build_eef_action_descriptor,
)

from serve_robotwin_policy import EpisodeSeededPolicy  # noqa: E402


@dataclasses.dataclass
class Args:
    config: str
    checkpoint_dir: str
    router_checkpoint: str
    action_horizon: int = 50
    port: int = 8000
    inference_seed: int = 0
    deterministic_torch: bool = False
    router_extraction_batch_size: int = 4


def _tree_to_torch(tree, device: torch.device, batch_size: int):
    if isinstance(tree, dict):
        return {
            key: _tree_to_torch(value, device, batch_size)
            for key, value in tree.items()
        }
    array = np.asarray(tree)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(np.repeat(array[None, ...], batch_size, axis=0)).to(device)


def _copy_tree_structure(tree):
    """Copy nested containers while sharing immutable/array leaves.

    OpenPI input transforms may mutate nested dictionaries in place (notably
    converting Aloha images from CHW to HWC). Router queries run multiple
    frozen feature forwards from the same request, so each forward needs an
    independent container tree just like ``Policy.infer`` creates.
    """
    if isinstance(tree, dict):
        return {key: _copy_tree_structure(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_copy_tree_structure(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_copy_tree_structure(value) for value in tree)
    return tree


class RouterScoredPolicy:
    """Dispatch normal action requests and side-effect-free router queries."""

    def __init__(
        self,
        policy,
        *,
        inference_seed: int,
        router_checkpoint: Path,
        device: torch.device,
        extraction_batch_size: int,
    ):
        self._policy = policy
        self._action_policy = EpisodeSeededPolicy(policy, inference_seed)
        self._model = policy._model  # noqa: SLF001
        self._device = device
        self._extraction_batch_size = int(extraction_batch_size)
        if self._extraction_batch_size < 1:
            raise ValueError("Router extraction batch size must be positive")

        payload = torch.load(router_checkpoint, map_location="cpu", weights_only=False)
        architecture = payload["architecture"]
        self._architecture_class = str(architecture["class"])
        self._lambda = float(payload["lambda"])
        self._horizon = int(architecture["horizon"])
        self._action_dim = int(architecture["action_dim"])
        self._feature_config = None
        self._eef_forward_kinematics = None
        self._eef_waypoints = None
        if self._architecture_class == "OutcomeRouter":
            self._feature_type = str(payload["feature_type"])
            self._router = OutcomeRouter(
                visual_dim=int(architecture["visual_dim"]),
                horizon=self._horizon,
                action_dim=self._action_dim,
                router_input=str(architecture["router_input"]),
                hidden_dim=int(architecture["hidden_dim"]),
                action_hidden_dim=int(architecture["action_hidden_dim"]),
                action_embedding_dim=int(architecture["action_embedding_dim"]),
                dropout=float(architecture["dropout"]),
            ).to(device)
            normalization = payload["normalization"]
            self._visual_mean = normalization["visual_mean"].to(device)
            self._visual_std = normalization["visual_std"].to(device)
            self._action_mean = normalization["action_mean"].to(device)
            self._action_std = normalization["action_std"].to(device)
            self._router_input = str(architecture["router_input"])
            self._component_normalization = None
        elif self._architecture_class == "FeatureAblationRouter":
            self._feature_config = str(payload["feature_config"])
            feature_manifest_path = Path(payload["feature_manifest"])
            feature_manifest = json.loads(feature_manifest_path.read_text())
            self._feature_type = str(feature_manifest["source_feature_type"])
            self._router = FeatureAblationRouter(
                {
                    str(name): int(dimension)
                    for name, dimension in architecture["direct_feature_dims"].items()
                },
                action_mode=str(architecture["action_mode"]),
                horizon=self._horizon,
                action_dim=self._action_dim,
                action_feature_dim=architecture.get("action_feature_dim"),
                hidden_dim=int(architecture["hidden_dim"]),
                action_hidden_dim=int(architecture["action_hidden_dim"]),
                action_embedding_dim=int(architecture["action_embedding_dim"]),
                dropout=float(architecture["dropout"]),
            ).to(device)
            self._component_normalization = {
                str(name): {
                    str(statistic): value.to(device)
                    for statistic, value in statistics.items()
                }
                for name, statistics in payload["normalization"].items()
            }
            self._router_input = self._feature_config
            if architecture["action_mode"] == "eef_trajectory":
                definition = feature_manifest["definitions"]["eef_trajectory"]
                fk_definition = definition["forward_kinematics"]
                self._eef_forward_kinematics = AlohaOfflineForwardKinematics(
                    Path(fk_definition["urdf"])
                )
                self._eef_waypoints = int(definition["waypoints"]["count"])
        else:
            raise ValueError(
                f"Unsupported router architecture: {self._architecture_class}"
            )
        self._router.load_state_dict(payload["model_state_dict"], strict=True)
        self._router.eval()
        for parameter in self._router.parameters():
            parameter.requires_grad_(False)

        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            raise AssertionError("pi0.5 was not fully frozen")
        logging.info(
            "Loaded frozen router %s: feature=%s input=%s lambda=%.4f params=%d",
            router_checkpoint,
            self._feature_type,
            self._router_input,
            self._lambda,
            int(architecture["trainable_parameters"]),
        )

    @property
    def metadata(self):
        return {
            **self._action_policy.metadata,
            "router_feature_type": self._feature_type,
            "router_input": self._router_input,
            "router_feature_config": self._feature_config,
            "router_lambda": self._lambda,
            "router_extraction_batch_size": self._extraction_batch_size,
        }

    def _frozen_visual_feature(self, request: dict) -> torch.Tensor:
        transformed = self._policy._input_transform(  # noqa: SLF001
            _copy_tree_structure(request)
        )
        # Offline features were extracted in batches of four.  Keeping the
        # same leading dimension also keeps CUDA kernel selection/numerics
        # aligned; duplicate rows are independent and only row zero is used.
        inputs = _tree_to_torch(transformed, self._device, self._extraction_batch_size)
        observation = model_types.Observation.from_dict(inputs)
        images, image_masks, language_tokens, language_masks, _ = (
            self._model._preprocess_observation(  # noqa: SLF001
                observation, train=False
            )
        )

        if self._feature_type == "vision_encoder":
            camera_features = []
            for image, image_mask in zip(images, image_masks, strict=True):
                tokens = self._model.paligemma_with_expert.embed_image(image)
                pooled = tokens.mean(dim=1)
                pooled = pooled * image_mask[:, None].to(pooled.dtype)
                camera_features.append(pooled)
            return torch.cat(camera_features, dim=-1).float()

        if self._feature_type == "vlm_hidden":
            prefix_embeddings, prefix_pad_masks, prefix_attention_masks = (
                self._model.embed_prefix(
                    images, image_masks, language_tokens, language_masks
                )
            )
            if (
                self._model.paligemma_with_expert.paligemma.language_model.layers[
                    0
                ].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embeddings = prefix_embeddings.to(torch.bfloat16)
            attention_2d = make_att_2d_masks(prefix_pad_masks, prefix_attention_masks)
            attention_4d = self._model._prepare_attention_masks_4d(  # noqa: SLF001
                attention_2d
            )
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            (
                (
                    prefix_hidden,
                    _,
                ),
                _,
            ) = self._model.paligemma_with_expert.forward(
                attention_mask=attention_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embeddings, None],
                use_cache=False,
            )
            language_length = int(language_masks.shape[1])
            language_hidden = prefix_hidden[:, -language_length:, :]
            language_mask = language_masks[:, :, None].to(language_hidden.dtype)
            return (
                (language_hidden * language_mask).sum(dim=1)
                / language_mask.sum(dim=1).clamp_min(1.0)
            ).float()

        raise ValueError(f"Unsupported router feature type: {self._feature_type}")

    def _frozen_action_hidden_tail(
        self,
        request: dict,
        old_action_chunk: np.ndarray,
        old_chunk_cursor: int,
    ) -> torch.Tensor:
        """Extract deterministic tau=0 action-expert hidden state without sampling."""
        feature_request = _copy_tree_structure(request)
        feature_request["actions"] = np.asarray(
            old_action_chunk, dtype=np.float32
        )
        transformed = self._policy._input_transform(  # noqa: SLF001
            feature_request
        )
        inputs = _tree_to_torch(
            transformed, self._device, self._extraction_batch_size
        )
        observation = model_types.Observation.from_dict(inputs)
        images, image_masks, language_tokens, language_masks, state = (
            self._model._preprocess_observation(  # noqa: SLF001
                observation, train=False
            )
        )
        normalized_actions = inputs["actions"].to(
            dtype=self._model.action_in_proj.weight.dtype
        )
        expected_shape = (
            self._extraction_batch_size,
            self._model.config.action_horizon,
            self._model.config.action_dim,
        )
        if tuple(normalized_actions.shape) != expected_shape:
            raise ValueError(
                "Transformed old chunk does not match pi0.5 action space: "
                f"{tuple(normalized_actions.shape)} != {expected_shape}"
            )
        prefix, prefix_pad, prefix_attention = self._model.embed_prefix(
            images, image_masks, language_tokens, language_masks
        )
        flow_time = torch.zeros(
            self._extraction_batch_size,
            dtype=torch.float32,
            device=self._device,
        )
        suffix, suffix_pad, suffix_attention, adarms_cond = (
            self._model.embed_suffix(state, normalized_actions, flow_time)
        )
        if (
            self._model.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix = prefix.to(torch.bfloat16)
            suffix = suffix.to(torch.bfloat16)
        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        attention_masks = torch.cat(
            [prefix_attention, suffix_attention], dim=1
        )
        attention_4d = self._model._prepare_attention_masks_4d(  # noqa: SLF001
            make_att_2d_masks(pad_masks, attention_masks)
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (hidden_outputs, _), _ = (
            self._model.paligemma_with_expert.forward(
                attention_mask=attention_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        )
        action_hidden = hidden_outputs[
            :, -self._model.config.action_horizon :
        ].float()
        return action_hidden[:, old_chunk_cursor:].mean(dim=1)

    @torch.inference_mode()
    def _router_query(self, observation: dict) -> dict:
        request = dict(observation)
        request.pop("router_query", None)
        old_action_chunk = np.asarray(request.pop("old_action_chunk"), dtype=np.float32)
        old_chunk_cursor = int(request.pop("old_chunk_cursor"))
        completed_actions = int(request.pop("completed_actions"))
        natural_replan_interval = int(request.pop("natural_replan_interval", 25))
        last_replan_action = request.pop("last_replan_action", None)
        requested_lambda = float(request.pop("router_lambda", self._lambda))
        episode_seed = request.pop("episode_seed", None)
        episode_id = request.pop("episode_id", None)
        current_state = np.asarray(request["state"], dtype=np.float32)

        if old_action_chunk.shape != (self._horizon, self._action_dim):
            raise ValueError(
                f"Expected old chunk {(self._horizon, self._action_dim)}, "
                f"got {old_action_chunk.shape}"
            )
        if not 0 < old_chunk_cursor < self._horizon:
            raise ValueError(f"Invalid old chunk cursor: {old_chunk_cursor}")
        if current_state.shape != (self._action_dim,):
            raise ValueError(
                f"Expected current state {(self._action_dim,)}, "
                f"got {current_state.shape}"
            )
        if natural_replan_interval < 1:
            raise ValueError("natural_replan_interval must be positive")

        visual = self._frozen_visual_feature(request)
        if self._architecture_class == "OutcomeRouter":
            visual = (visual - self._visual_mean) / self._visual_std
            action = torch.from_numpy(
                np.repeat(
                    old_action_chunk[None, ...],
                    self._extraction_batch_size,
                    axis=0,
                )
            ).to(self._device)
            action = (action - self._action_mean[None, None, :]) / self._action_std[
                None, None, :
            ]
            logits = self._router(visual, action)
        else:
            raw_features = {"vision": visual}
            component_names = set(self._component_normalization)
            if "chunk_state" in component_names:
                offset = completed_actions % natural_replan_interval
                distance_to_next_r0 = (
                    natural_replan_interval
                    if offset == 0
                    else natural_replan_interval - offset
                )
                previous_router_clock = (
                    int(last_replan_action) if last_replan_action is not None else 0
                )
                chunk_state = np.asarray(
                    [
                        old_chunk_cursor / self._horizon,
                        (self._horizon - old_chunk_cursor) / self._horizon,
                        old_chunk_cursor,
                        distance_to_next_r0,
                        completed_actions - previous_router_clock,
                    ],
                    dtype=np.float32,
                )
                raw_features["chunk_state"] = torch.from_numpy(
                    np.repeat(
                        chunk_state[None, :],
                        self._extraction_batch_size,
                        axis=0,
                    )
                ).to(self._device)
            if "eef_trajectory" in component_names:
                descriptor, integration_error = build_eef_action_descriptor(
                    old_action_chunk[old_chunk_cursor:],
                    current_state,
                    forward_kinematics=self._eef_forward_kinematics,
                    waypoint_count=self._eef_waypoints,
                )
                if integration_error > 1e-6:
                    raise AssertionError(
                        f"Online EEF integration error: {integration_error}"
                    )
                raw_features["eef_trajectory"] = torch.from_numpy(
                    np.repeat(
                        descriptor[None, :],
                        self._extraction_batch_size,
                        axis=0,
                    )
                ).to(self._device)
            if "action_expert_hidden_tail" in component_names:
                raw_features["action_expert_hidden_tail"] = (
                    self._frozen_action_hidden_tail(
                        request,
                        old_action_chunk,
                        old_chunk_cursor,
                    )
                )
            normalized_features = {}
            for name, value in raw_features.items():
                normalization = self._component_normalization[name]
                normalized_features[name] = (
                    value - normalization["mean"]
                ) / normalization["std"]
            logits = self._router(normalized_features)

        probabilities = torch.sigmoid(logits)[0]
        p_keep = float(probabilities[0].cpu())
        p_replan = float(probabilities[1].cpu())
        advantage = p_replan - p_keep
        return {
            "router": {
                "p_keep": p_keep,
                "p_replan": p_replan,
                "advantage": advantage,
                "lambda": requested_lambda,
                "trigger": bool(advantage > requested_lambda),
                "feature_type": self._feature_type,
                "router_input": self._router_input,
                "feature_config": self._feature_config,
                "old_chunk_cursor": old_chunk_cursor,
                "completed_actions": completed_actions,
                "episode_seed": episode_seed,
                "episode_id": episode_id,
            }
        }

    def infer(self, observation: dict):
        if bool(observation.get("router_query", False)):
            return self._router_query(observation)
        return self._action_policy.infer(observation)


def main(args: Args) -> None:
    if args.deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logging.info("Enabled deterministic PyTorch inference kernels")

    config = training_config.get_config(args.config)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, action_horizon=args.action_horizon),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        pytorch_device=str(device),
    )
    served_policy = RouterScoredPolicy(
        policy,
        inference_seed=args.inference_seed,
        router_checkpoint=Path(args.router_checkpoint),
        device=device,
        extraction_batch_size=args.router_extraction_batch_size,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=served_policy,
        host="0.0.0.0",
        port=args.port,
        metadata=served_policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
