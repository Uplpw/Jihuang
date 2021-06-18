import gym
import random
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

import Jihuang._jihuang as game
from numpy.core.numeric import Infinity
from Jihuang.display_utils import MapDisplayer
from Jihuang.reward import JihuangReward
from Jihuang.obs_utils import JihuangObsProcess
from Jihuang.utils import fprint, fprintNoHp, getLogPath, writeEpisodeLog, updateEpisodeLog, showEpisodeLog, \
    initEpisodeLog, list_dict_contain
from Jihuang.config import water_meat_count, torch_count
from Jihuang.find_pig_algo import random_find_pig, random_constraint_find_pig


class JihuangSimple(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 env_param="python/gym_api/Jihuang/config/example_simple.prototxt",
                 env_config="python/gym_api/Jihuang/config/config_simple.prototxt",
                 log_dir="logs", log_name="py_jihuang", log_level=0,
                 displayer=False):
        self.env = game.Env(env_param, env_config, log_dir, log_name, log_level)
        self.step_count = 0
        if displayer:
            self.displayer = MapDisplayer()

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-20, high=200000, shape=(38,))

        self.obs_process = JihuangObsProcess()
        self.reward = JihuangReward()

        # initialize the observation and action_count
        self._prev_obs = np.zeros((4219,))
        self.obs = None

        # save animal or plants or material list
        self._prev_pigs = []
        self._prev_rivers = []
        self._prev_material = []

        # log
        self.log_txt = 'log/' + getLogPath()
        self.episode_log_list = initEpisodeLog()

    def step(self, action: int):
        print("action:", action, type(action))
        # translate into C++ Jihuang actions
        if action == 0:  # idle
            self.action = [[0., 0., 0., 0.]]

        elif action <= 2:  # attack, collect
            target_x, target_y = self.select_target(action)
            self.action = [[0., float(action), float(target_x), float(target_y)]]

        elif action <= 5:  # pickup, consume, equip
            target_type, _ = self.select_target(action)
            self.action = [[0., float(action), float(target_type), 1.]]

        elif action == 6:  # move
            offset_x, offset_y = self.select_target(action)
            self.action = [[0., 8.0, float(offset_x), float(offset_y)]]

        # run action and obtain new obs
        print("asda:", self.action)
        if self.obs is not None:
            self._prev_obs = self.obs.copy()
        print("999999")
        self.env.step(self.action)
        print("step")
        self.obs = self.env.get_agent_observe()[0]
        obs_scale = self.obs_process._obs_scale(self.obs)
        reward, result_type = self._calc_reward(action)
        self.step_count += 1
        print(self.step_count)

        # update episode log
        updateEpisodeLog(self.episode_log_list, action, result_type, reward)

        # Done when the agent dies.
        if self.obs[1] <= 2.0 or self.obs[2] <= 2.0 or self.obs[3] <= 2.0:
            showEpisodeLog(self.episode_log_list)
            # writeEpisodeLog(self.log_txt, self.episode_log_list)
            done = True
        else:
            done = False
        info = {}

        return obs_scale, reward, done, info

    def select_target(self, action: int):
        if action <= 2:
            if action == 1:
                pigs = self.obs_process._find_pigs(self.obs)
                if len(pigs) <= 0:
                    return random.randint(0, 40), random.randint(0, 40)
                return pigs[0][0], pigs[0][1]
            if action == 2:
                rivers = self.obs_process._find_rivers(self.obs)
                if len(rivers) <= 0:
                    return random.randint(0, 40), random.randint(0, 40)
                return rivers[0][0], rivers[0][1]
        elif action <= 5:
            if action == 3:  # pickup
                backpack = self.obs_process._analyse_backpack(self.obs)
                material = self.obs_process._material_process_pickup(self.obs,
                                                                     self.obs_process._find_material(self.obs))
                if backpack[30001] <= 0 and backpack[30002] > 0:
                    if list_dict_contain(material, 30001):
                        return 30001, 0
                    else:
                        return 30002, 0
                elif backpack[30002] <= 0 or backpack[30002] < backpack[30001]:
                    if list_dict_contain(material, 30002):
                        return 30002, 0
                    else:
                        return 30001, 0
                else:
                    return 30001, 0

            if action == 4:  # consume
                satiety = self.obs[2]
                thirsty = self.obs[3]
                if satiety <= thirsty:
                    return 30002, 0
                else:
                    return 30001, 0
            if action == 5:
                return 70009, 0

        elif action == 6:  # move, pigs(*)/material
            x_init, y_init = self.obs[5], self.obs[6]
            pigs = self.obs_process._find_pigs(self.obs)
            # materials = self.obs_process._find_material(self.obs)  # 向材料移动
            if len(pigs) > 0:
                pig_x, pig_y = pigs[0][0], pigs[0][1]
                agent_pig_distance = np.sqrt((pig_x - x_init) ** 2 + (pig_y - y_init) ** 2)
                offset_x, offset_y = 4.1 * (pig_x - x_init) / agent_pig_distance, 4.1 * (
                        pig_y - y_init) / agent_pig_distance
                print("select_target1", int(offset_x), int(offset_y))
                return int(offset_x), int(offset_y)
            else:
                offset_x, offset_y = random_find_pig()
                print("select_target2", offset_x, offset_y)
                return offset_x, offset_y

    # -------------------- calc_reward -------------------- #
    def _calc_reward(self, action_id):
        reward = 0.
        result_type = ""

        # calc idle reward
        if action_id == 0:
            pass

        # calc attack reward
        if action_id == 1:
            reward, result_type = self.reward._attack_pig_reward(self._prev_obs, self.obs, self._prev_pigs)

        # calc move reward
        if action_id == 2:
            pigs = self.obs_process._find_pigs(self.obs)
            reward, result_type = self.reward._move_reward(self._prev_obs, self.obs, self._prev_pigs, pigs)

        # calc consume reward
        if action_id == 3:
            reward, result_type = self.reward._consume_reward(self._prev_obs, self.obs)

        # calc collect reward
        if action_id == 4:
            reward, result_type = self.reward._collect_water_reward(self.obs)

        # calc pickup reward
        if action_id == 5:
            reward, result_type = self.reward._pickup_material_reward(self._prev_obs, self.obs)

        # calc equip reward
        if action_id == 6:
            reward, result_type = self.reward._equip_torch_reward(self._prev_obs, self.obs)

        return reward, result_type

    # -------------------- env reset -------------------- #
    def reset(self):
        print("------------------------------- env start -------------------------------")
        # Variable initialization
        self.env.reset()
        self._prev_pigs = []
        self._prev_rivers = []
        self._prev_material = []
        self.obs = None
        self._prev_obs = np.zeros((4219,))
        self.step_count = 0

        self.episode_log_list = initEpisodeLog()

        self.obs = self.env.get_agent_observe()[0]
        obs_scale = self.obs_process._obs_scale(self.obs)
        return obs_scale

    def render(self, mode='human'):
        if self.displayer is not None:
            self.displayer.display_map(self.env, self.step_count)
        else:
            print(self.obs)

    def close(self):
        pass
