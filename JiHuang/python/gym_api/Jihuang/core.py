import sys

sys.path.append("..")
import gym
import random
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import Jihuang._jihuang as game
from Jihuang.display_utils import MapDisplayer

from utils.log import log_func


class JihuangSimple(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 env_param="python/gym_api/Jihuang/config/example_simple.prototxt",
                 env_config="python/gym_api/Jihuang/config/config_simple.prototxt",
                 log_dir="logs", log_name="py_jihuang", log_level=0, seed=0, displayer=False):
        self.env = game.Env(env_param, env_config, log_dir, log_name, log_level, seed)
        self.step_count = 0
        if displayer:
            self.displayer = MapDisplayer()
        # self.obs_dim = 7
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-20, high=200000, shape=(77,))

        # initialize the observation and action_count
        self.reward_range = (-float('inf'), float('inf'))
        self._prev_obs = np.zeros((4219,))

        self.obs = None

        # log
        # self.init_path, self.init_list = log_func("init")

    def step(self, action: list):
        if action[0] <= 0:  # idle, attack, collect
            self.action = [[0., float(0), float(0), float(0)]]

        elif action[0] <= 5:  # pickup, consume, equip
            self.action = [[0., float(action[0]), float(action[1]), float(action[2])]]

        elif action[0] == 6:  # move
            self.action = [[0., 8.0, float(action[1]), float(action[2])]]

        if self.obs is not None:
            self._prev_obs = self.obs.copy()
        self.env.step(self.action)
        self.obs = self.env.get_agent_observe()[0]
        observation = self.obs
        self.step_count += 1

        # update = (action[0], result_type, reward)
        # log_func("update", log_list=self.init_list, log_update=update)

        # Done when the agent dies.
        if self.obs[1] <= 2.0 or self.obs[2] <= 2.0 or self.obs[3] <= 2.0:
            done = True
        else:
            done = False

        info = {}

        return observation, 0., done, info

    def reset(self):
        self.env.reset()
        self.obs = None
        self._prev_obs = np.zeros((4219,))
        self.step_count = 0
        self.obs = self.env.get_agent_observe()[0]
        observation = self.obs
        return observation

    def render(self, mode='human'):
        if self.displayer is not None:
            self.displayer.display_map(self.env, self.step_count)
        else:
            print(self.obs)

    def close(self):
        pass


class JihuangSimpleTarget(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 env_param="python/gym_api/Jihuang/config/example_simple.prototxt",
                 env_config="python/gym_api/Jihuang/config/config_simple.prototxt",
                 log_dir="logs", log_name="py_jihuang", log_level=0, seed=0, displayer=False, select_target_func=None):
        self.env = game.Env(env_param, env_config, log_dir, log_name, log_level, seed)
        self.step_count = 0
        if displayer:
            self.displayer = MapDisplayer()

        self.obs_dim = 4219
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-20, high=200000, shape=(4219,))

        # initialize the observation and action_count
        self._prev_obs = np.zeros((4219,))
        self.obs = None

        self.select_target_func = select_target_func
        if self.select_target_func is None:
            print("please select a target_func")
            raise NotImplementedError

    def step(self, action: int):
        if action <= 2:
            target_x, target_y = self.select_target_func(self.obs, action)
            self.action = [[0., float(action), float(target_x), float(target_y)]]

        elif action <= 5:
            target_x, target_y = self.select_target_func(self.obs, action)
            self.action = [[0., float(action), float(target_x), 1.]]

        elif action == 6:
            target_x, target_y = self.select_target_func(self.obs, action)
            self.action = [[0., 8.0, float(target_x), float(target_y)]]

        if self.obs is not None:
            self._prev_obs = self.obs.copy()
        self.env.step(self.action)
        self.obs = self.env.get_agent_observe()[0]
        observation = self.obs.copy()
        self.step_count += 1

        # Done when the agent dies.
        if self.obs[1] <= 2.0 or self.obs[2] <= 2.0 or self.obs[3] <= 2.0:
            done = True
        else:
            done = False
        info = {}

        return observation, 0, done, info

    def reset(self):
        self.env.reset()
        self.obs = None
        self._prev_obs = np.zeros((4219,))
        self.step_count = 0
        self.obs = self.env.get_agent_observe()[0]
        observation = self.obs.copy()
        return observation

    def render(self, mode='human'):
        if self.displayer is not None:
            self.displayer.display_map(self.env, self.step_count)
        else:
            print(self.obs)

    def close(self):
        pass


class JihuangOriginal(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 env_param="python/gym_api/Jihuang/config/example_simple.prototxt",
                 env_config="python/gym_api/Jihuang/config/config_simple.prototxt",
                 log_dir="logs", log_name="py_jihuang", log_level=0, seed=0, displayer=False):
        self.env = game.Env(env_param, env_config, log_dir, log_name, log_level, seed)
        self.step_count = 0
        if displayer:
            self.displayer = MapDisplayer()

        self.obs_dim = 4219
        self.action_space = spaces.MultiDiscrete([7, 40, 40])
        self.observation_space = spaces.Box(low=-20, high=200000, shape=(4219,))
        self._prev_obs = np.zeros((4219,))
        self.obs = None

    def step(self, action: list):
        if action[0] <= 5:
            self.action = [[0., float(action[0]), float(action[1]), float(action[2])]]
        elif action[0] == 6:
            self.action = [[0., 8.0, float(action[1]), float(action[2])]]

        if self.obs is not None:
            self._prev_obs = self.obs.copy()
        self.env.step(self.action)
        self.obs = self.env.get_agent_observe()[0]
        observation = self.obs
        self.step_count += 1

        # Done when the agent dies.
        if self.obs[1] <= 2.0 or self.obs[2] <= 2.0 or self.obs[3] <= 2.0:
            done = True
        else:
            done = False
        info = {}

        return observation, 0, done, info

    def reset(self):
        self.env.reset()
        self.obs = None
        self._prev_obs = np.zeros((4219,))
        self.step_count = 0
        self.obs = self.env.get_agent_observe()[0]
        observation = self.obs.copy()
        return observation

    def render(self, mode='human'):
        if self.displayer is not None:
            self.displayer.display_map(self.env, self.step_count)
        else:
            print(self.obs)

    def close(self):
        pass
