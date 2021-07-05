# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
import collections
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
import gym
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.preprocessing import get_action_dim, maybe_transpose, preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, \
    create_mlp
from stable_baselines3.common.utils import get_device, is_vectorized_observation
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper


class MyActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                sde_net_arch=default_none_kwargs["sde_net_arch"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        assert isinstance(self.action_dist,
                          StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, action_mask=None, deterministic: bool = False,
                device: Union[th.device, str] = "auto") -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask, latent_sde=latent_sde, device=device)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, action_mask=None,
                                     latent_sde: Optional[th.Tensor] = None,
                                     device: Union[th.device, str] = "auto") -> Distribution:
        mean_actions = self.action_net(latent_pi)

        if action_mask is None:
            if isinstance(self.action_dist, CategoricalDistribution):
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            else:
                raise ValueError("Invalid action distribution")
        else:
            mean_actions_cpu = th.tensor(mean_actions.clone().detach()).to(device)
            action_mask_cpu = th.BoolTensor(th.tensor(action_mask).bool()).to(device)
            mean_actions_cpu = mean_actions_cpu.masked_fill(action_mask_cpu, value=th.tensor(-10e7).to(device))
            if isinstance(self.action_dist, CategoricalDistribution):
                return self.action_dist.proba_distribution(action_logits=mean_actions_cpu)
            else:
                raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, action_mask=None, deterministic: bool = False,
                 device: Union[th.device, str] = "auto") -> th.Tensor:
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask, latent_sde, device=device)
        return distribution.get_actions(deterministic=deterministic)

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
            actions = self._predict(observation, action_mask, deterministic=deterministic, device=device)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, action_mask,
                         device: Union[th.device, str] = "auto") -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask, latent_sde, device=device)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


MlpPolicy = MyActorCriticPolicy

register_policy("MlpPolicy1", MyActorCriticPolicy)
