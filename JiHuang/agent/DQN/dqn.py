import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from .policies import DQNPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from utils.trick import action_mask_from_obs


class DQN(OffPolicyAlgorithm):

    def __init__(
            self,
            policy: Union[str, Type[DQNPolicy]],
            env: Union[GymEnv, str],
            action_mask: bool = False,
            learning_rate: Union[float, Schedule] = 1e-4,
            buffer_size: int = 1000000,
            learning_starts: int = 100,
            batch_size: Optional[int] = 32,
            tau: float = 1.0,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 4,
            gradient_steps: int = 1,
            optimize_memory_usage: bool = False,
            target_update_interval: int = 10000,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 10,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        super(DQN, self).__init__(
            policy,
            env,
            DQNPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
        )

        self.action_mask = action_mask
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.exploration_rate = 0.0
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(DQN, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        # self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.exploration_rate = self._get_linear_fn()
        logger.record("rollout/exploration rate", self.exploration_rate)

    def _get_linear_fn(self):
        epsilon = 1. - (self.exploration_initial_eps - self.exploration_final_eps) * self.num_timesteps / 1000000
        return epsilon

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def predict(
            self,
            observation: np.ndarray,
            action_mask=None,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
            device: Union[th.device, str] = "auto"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, action_mask, state, mask, deterministic, device)
        return action, state

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 4,
              eval_env: Optional[GymEnv] = None, eval_freq: int = -1, n_eval_episodes: int = 5,
              tb_log_name: str = "DQN", eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True,
              ) -> OffPolicyAlgorithm:

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

    def _excluded_save_params(self) -> List[str]:
        return super(DQN, self)._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []

    # 新增，替代父类中的collect_rollouts
    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, train_freq: TrainFreq, replay_buffer: ReplayBuffer,
                         action_noise: Optional[ActionNoise] = None, learning_starts: int = 0,
                         log_interval: Optional[int] = None, device: Union[th.device, str] = "auto"
                         ) -> RolloutReturn:

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)

    def _sample_action(self, learning_starts: int, action_noise: Optional[ActionNoise] = None,
                       ) -> Tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            unscaled_action = np.array([self.action_space.sample()])
        else:
            if self.action_mask:
                action_mask = action_mask_from_obs(self._last_obs)
            else:
                action_mask = None
            unscaled_action, _ = self.predict(self._last_obs, action_mask, deterministic=False, device=self.device)

        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
