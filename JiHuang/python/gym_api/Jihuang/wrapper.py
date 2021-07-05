# from core import JihuangSimple
from gym import Wrapper, ObservationWrapper, RewardWrapper, ActionWrapper
from Jihuang.utils.reward import reward_func
from Jihuang.utils.log import log_func
from Jihuang.utils.trick import select_target_func, _obs_scale


class ScaledObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return _obs_scale(observation)


class Reward(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.init_path, self.init_list = log_func("init")

    def reset(self, **kwargs):
        self.init_path, self.init_list = log_func("init")
        return self.env.reset(**kwargs)

    def step(self, action: list):
        observation, _, done, info = self.env.step(action)
        reward, result_type = self.my_reward(action, self.env._prev_obs, observation)
        update = (action[0], result_type, reward)
        log_func("update", log_list=self.init_list, log_update=update)

        if done:
            log_func("show", log_list=self.init_list)

        return observation, reward, done, info

    def reward(self, reward):
        raise NotImplementedError

    def my_reward(self, action: list, preb_obs, obs):
        reward, result_type = reward_func(action[0], preb_obs, obs)

        return reward, result_type


class SelectTarget(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(self.my_action(self.env.obs, action))

    def action(self, action):
        raise NotImplementedError

    def my_action(self, observation, action):
        target_x, target_y = select_target_func(observation, action)
        return [action, target_x, target_y]

    def reverse_action(self, action):
        raise NotImplementedError


# class Logger(Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.init_path, self.init_list = log_func("init")
#
#     def reset(self, **kwargs):
#         self.init_path, self.init_list = log_func("init")
#         return self.env.reset(**kwargs)
#
#     def step(self, action):
#         observation, reward, done, info = self.env.step(action)
#
#         update = (action, result_type, reward)
#         log_func("update", log_list=self.init_list, log_update=update)
#         if done:
#             log_func("show", log_list=self.init_list)
#         return self.env.step(self.action(action))
#
#     def action(self, action):
#         raise NotImplementedError
#
#     def my_action(self, observation, action):
#         return select_target_func(action)
#
#     def reverse_action(self, action):
#         raise NotImplementedError
#
#
# class AllWrapper(Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         env = ScaledObservation(env)
#         env = Reward(env)
#         env = SelectTarget(env)
#         self.env = env
#
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)
