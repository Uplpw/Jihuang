import sys
sys.path.append("..")
import warnings
from collections import deque
import io
import pathlib
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch as th
import gym
from gym import spaces
from torch.nn import functional as F
from .policies import ActorCriticPolicy
from stable_baselines3.common import logger, utils
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, \
    save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    explained_variance,
    get_device,
    get_schedule_fn,
    safe_mean,
    set_random_seed,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from utils.trick import action_mask_from_obs


class PPO:
    def __init__(
            self,
            env: Union[GymEnv, str],
            action_mask: bool = False,
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: Optional[int] = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        supported_action_spaces = (spaces.Box, spaces.Discrete, spaces.MultiDiscrete)

        self.policy_class = ActorCriticPolicy

        self.device = get_device(device)
        if verbose > 0:
            print(f"Using {self.device} device")

        self.env = None
        self.action_mask = action_mask
        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self.num_timesteps = 0
        self._total_timesteps = 0
        self.eval_env = None
        self.seed = seed
        self.action_noise = None
        self.start_time = None
        self.policy = None
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.lr_schedule = None
        self._last_obs = None
        self._last_dones = None
        self._last_original_obs = None
        self._episode_num = 0
        self._current_progress_remaining = 1
        self.ep_info_buffer = None
        self.ep_success_buffer = None
        self._n_updates = 0

        if env is not None:
            if isinstance(env, str):
                if create_eval_env:
                    if verbose >= 1:
                        print(f"Creating environment from the given name '{env}'")
                    self.eval_env = gym.make(env)

            if isinstance(env, str):
                """If env is a string, make the environment; otherwise, do nothing."""
                if verbose >= 1:
                    print(f"Creating environment from the given name '{env}'")
                env = gym.make(env)
            env = self._wrap_env(env, self.verbose)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            if supported_action_spaces is not None:
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        if not isinstance(env, VecEnv):
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])

        if (
                is_image_space(env.observation_space)
                and not is_vecenv_wrapped(env, VecTransposeImage)
                and not is_image_space_channels_first(env.observation_space)
        ):
            if verbose >= 1:
                print("Wrapping the env in a VecTransposeImage.")
            env = VecTransposeImage(env)

        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            env = ObsDictWrapper(env)

        return env

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=False,
            **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _get_eval_env(self, eval_env: Optional[GymEnv]) -> Optional[GymEnv]:
        if eval_env is None:
            eval_env = self.eval_env

        if eval_env is not None:
            eval_env = self._wrap_env(eval_env, self.verbose)
            assert eval_env.num_envs == 1
        return eval_env

    def _setup_lr_schedule(self) -> None:
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:

        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def _excluded_save_params(self) -> List[str]:
        return [
            "policy",
            "device",
            "env",
            "eval_env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []

    def _init_callback(
            self,
            callback: MaybeCallback,
            eval_env: Optional[VecEnv] = None,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            log_path: Optional[str] = None,
    ) -> BaseCallback:
        if isinstance(callback, list):
            callback = CallbackList(callback)

        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_path,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
            )
            callback = CallbackList([callback, eval_callback])

        callback.init_callback(self)
        return callback

    def _setup_learn(
            self,
            total_timesteps: int,
            eval_env: Optional[GymEnv],
            callback: MaybeCallback = None,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        self.start_time = time.time()
        if self.ep_info_buffer is None or reset_num_timesteps:
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()

            self._last_dones = np.zeros((self.env.num_envs,), dtype=bool)
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)
        utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def get_env(self) -> Optional[VecEnv]:
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        return self._vec_normalize_env

    def set_env(self, env: GymEnv) -> None:
        env = self._wrap_env(env, self.verbose)
        check_for_correct_spaces(env, self.observation_space, self.action_space)

        self.n_envs = env.num_envs
        self.env = env

    def collect_rollouts(
            self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                if self.action_mask:
                    action_mask = action_mask_from_obs(self._last_obs)
                    actions, values, log_probs = self.policy.forward(obs_tensor, action_mask, device=self.device)
                else:
                    actions, values, log_probs = self.policy.forward(obs_tensor, device=self.device)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # print("clipped_actions:", actions)
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            if self.action_mask:
                action_mask = action_mask_from_obs(self._last_obs)
                _, values, _ = self.policy.forward(obs_tensor, action_mask, device=self.device)
            else:
                _, values, _ = self.policy.forward(obs_tensor, device=self.device)

        rollout_buffer.compute_returns_and_advantage(values, dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.action_mask:
                    action_mask = action_mask_from_obs(rollout_data.observations)
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions,
                                                                             action_mask, device=self.device)
                else:
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions, None,
                                                                             device=self.device)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "PPO":

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            self.train()
        callback.on_training_end()

        return self

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        if self.action_mask:
            action_mask = action_mask_from_obs(observation)
            return self.policy.predict(observation, action_mask, state, mask, deterministic, device=self.device)
        else:
            return self.policy.predict(observation, state, mask, deterministic, device=self.device)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)
        if self.eval_env is not None:
            self.eval_env.seed(seed)

    def set_parameters(
            self,
            load_path_or_dict: Union[str, Dict[str, Dict]],
            exact_match: bool = True,
            device: Union[th.device, str] = "auto",
    ) -> None:
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception:
                raise ValueError(f"Key {name} is an invalid object name.")

            if isinstance(attr, th.optim.Optimizer):
                attr.load_state_dict(params[name])
            else:
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    @classmethod
    def load(
            cls,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> "PPO":
        data, params, pytorch_variables = load_from_zip_file(path, device=device, custom_objects=custom_objects)

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            env = cls._wrap_env(env, data["verbose"])
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        else:
            if "env" in data:
                env = data["env"]

        model = cls(
            env=env,
            device=device,
            _init_setup_model=False,
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        model.set_parameters(params, exact_match=True, device=device)

        if pytorch_variables is not None:
            for name in pytorch_variables:
                recursive_setattr(model, name, pytorch_variables[name])

        return model

    def get_parameters(self) -> Dict[str, Dict]:
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            params[name] = attr.state_dict()
        return params

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:

        data = self.__dict__.copy()
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            var_name = torch_var.split(".")[0]
            exclude.add(var_name)

        for param_name in exclude:
            data.pop(param_name, None)

        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)
