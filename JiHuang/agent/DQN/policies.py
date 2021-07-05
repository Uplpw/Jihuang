from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation


class QNetwork(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor: nn.Module,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super(QNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        action_dim = self.action_space.n  # number of actions
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.q_net(self.extract_features(obs))

    def _predict(self, observation: th.Tensor, action_mask=None, deterministic: bool = True,
                 device: Union[th.device, str] = "auto") -> th.Tensor:
        if action_mask is None:
            q_values = self.forward(observation)
            action = q_values.argmax(dim=1).reshape(-1)
        else:
            action_mask_cpu = th.BoolTensor(th.tensor(action_mask).bool()).to(device)
            q_values = self.forward(observation)
            q_values_copy = q_values.clone().detach()
            mask_q_values = q_values_copy.masked_fill(action_mask_cpu, value=th.tensor(-10e7).to(device))
            action = mask_q_values.argmax(dim=1).reshape(-1)

        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DQNPolicy(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(DQNPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [64, 64]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, action_mask=None, deterministic: bool = True,
                device: Union[th.device, str] = "auto") -> th.Tensor:
        return self._predict(obs, action_mask=action_mask, deterministic=deterministic, device=device)

    def _predict(self, obs: th.Tensor, action_mask=None, deterministic: bool = True,
                 device: Union[th.device, str] = "auto") -> th.Tensor:
        return self.q_net._predict(obs, action_mask=action_mask, deterministic=deterministic, device=device)

    def predict(
            self,
            observation: np.ndarray,
            action_mask=None,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
            device: Union[th.device, str] = "auto"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        if isinstance(observation, dict):
            observation = ObsDictWrapper.convert_dict(observation)
        else:
            observation = np.array(observation)

        # Handle the different cases for images
        # as PyTorch use channel first format
        observation = maybe_transpose(observation, self.observation_space)

        vectorized_env = is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = th.as_tensor(observation).to(self.device)
        with th.no_grad():
            actions = self._predict(observation, action_mask=action_mask, deterministic=deterministic, device=device)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data


MlpPolicy = DQNPolicy


class CnnPolicy(DQNPolicy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
