"""Serve a RoboTwin fine-tuned checkpoint with an explicit action horizon H.

Run this in the openpi environment. RoboTwin connects through the lightweight
``openpi-client`` WebSocket package, so no SAPIEN/Open3D dependency is needed
on this server.
"""

import dataclasses
import logging
import socket

import jax
import torch
import tyro

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    config: str
    checkpoint_dir: str
    action_horizon: int = 50
    port: int = 8000
    inference_seed: int = 0
    deterministic_torch: bool = False


class EpisodeSeededPolicy:
    """Reset diffusion sampling per RoboTwin scene for paired evaluation."""

    def __init__(self, policy, inference_seed: int):
        self._policy = policy
        self._base_key = jax.random.key(inference_seed)
        self._inference_seed = int(inference_seed)
        self._episode_seed = None
        self._episode_id = None
        self._primary_inference_index = 0

    @property
    def _is_pytorch(self) -> bool:
        return bool(getattr(self._policy, "_is_pytorch_model", False))

    def _torch_seed(self, episode_seed: int, stream: int, index: int) -> int:
        """Stable seed in torch's accepted signed-64-bit range.

        ``stream=0`` is the executed policy sequence and ``stream=1`` is the
        counterfactual shadow sequence.  Reseeding every inference prevents
        episode order, prior episodes, and shadow calls from affecting PyTorch
        diffusion noise.
        """
        value = (
            self._inference_seed * 1_000_003
            + int(episode_seed) * 9_176
            + int(stream) * 1_000_000_007
            + int(index)
        )
        return value % (2**63 - 1)

    @staticmethod
    def _seed_torch(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _save_torch_rng() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        return torch.random.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    @staticmethod
    def _restore_torch_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
        cpu_state, cuda_states = state
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, observation):
        observation = dict(observation)
        episode_seed = observation.pop("episode_seed", None)
        # ``episode_id`` identifies an evaluation occurrence, while
        # ``episode_seed`` remains the only episode-specific entropy source.
        # This distinction matters when deliberately replaying the *same*
        # scene seed multiple times in one server process.
        episode_id = observation.pop("episode_id", None)
        shadow_probe_id = observation.pop("shadow_probe_id", None)
        # A shadow probe must never advance the sampling RNG used by the
        # executed policy stream.  It receives a deterministic, separately
        # folded key and restores the normal stream afterward.
        if shadow_probe_id is not None:
            if episode_seed is None:
                raise ValueError("shadow_probe_id requires episode_seed")
            if self._is_pytorch:
                saved_rng = self._save_torch_rng()
                self._seed_torch(self._torch_seed(int(episode_seed), stream=1, index=int(shadow_probe_id)))
                try:
                    return self._policy.infer(observation)
                finally:
                    self._restore_torch_rng(saved_rng)
            saved_rng = self._policy._rng
            probe_key = jax.random.fold_in(self._base_key, int(episode_seed))
            self._policy._rng = jax.random.fold_in(probe_key, int(shadow_probe_id))
            try:
                return self._policy.infer(observation)
            finally:
                self._policy._rng = saved_rng
        if episode_seed is not None:
            episode_seed = int(episode_seed)
            is_new_episode = (
                episode_seed != self._episode_seed
                or (episode_id is not None and episode_id != self._episode_id)
            )
            if is_new_episode:
                if not self._is_pytorch:
                    self._policy._rng = jax.random.fold_in(self._base_key, episode_seed)
                self._episode_seed = episode_seed
                self._episode_id = episode_id
                self._primary_inference_index = 0
                logging.info("Reset policy RNG for episode_seed=%d episode_id=%r", episode_seed, episode_id)
            if self._is_pytorch:
                self._seed_torch(self._torch_seed(episode_seed, stream=0, index=self._primary_inference_index))
                self._primary_inference_index += 1
        return self._policy.infer(observation)


def main(args: Args) -> None:
    if args.deterministic_torch:
        # This is deliberately opt-in: some CUDA kernels have no deterministic
        # implementation and should fail loudly in a paired diagnostic rather
        # than silently produce different action chunks on different runs.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logging.info("Enabled deterministic PyTorch inference kernels")
    config = training_config.get_config(args.config)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, action_horizon=args.action_horizon))
    policy = EpisodeSeededPolicy(
        policy_config.create_trained_policy(config, args.checkpoint_dir),
        args.inference_seed,
    )
    logging.info("Serving %s with predicted action horizon H=%d on port %d", args.checkpoint_dir, args.action_horizon, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
