import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

import Jihuang._jihuang as game
from numpy.core.numeric import Infinity
from Jihuang.display_utils import MapDisplayer
from Jihuang.reward import JihuangReward
from Jihuang.obs_utils import JihuangObsProcess
from Jihuang.utils import fprint, fprintNoHp, getLogPath, writeEpisodeLog, updateEpisodeLog, showEpisodeLog, \
    initEpisodeLog


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

        self.action_space = spaces.MultiDiscrete([7, 40, 40])
        self.observation_space = spaces.Box(low=-20, high=200000, shape=(30,))

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

    def step(self, action):
        # translate into C++ Jihuang actions
        if 0 == action[0]:  # Idle
            self.action = [[0., 0., 0., 0.]]
        elif 1 == action[0]:  # Auto attack
            self._prev_pigs = self.obs_process._find_pigs(self.obs).copy()
            if len(self._prev_pigs) == 0:
                self.action = [[0., 1., float(action[1]), float(action[2])]]
            else:
                # print("----------------------find pig----------------------", len(self._prev_pigs))
                self.action = [[0., 1., float(self._prev_pigs[0][0]), float(self._prev_pigs[0][1])]]
        elif 2 == action[0]:  # Move, x and y
            self.action = [[0., 8., float(action[1]), float(action[2])]]

        elif 3 == action[0]:  # Consume, type and num
            type_of_consume = 30001. if action[1] >= 20 else 30002.  # Only meat or water
            # print("consume before:", self.reward._analyse_backpack(self.obs), type_of_consume)
            self.action = [[0., 4., type_of_consume, 1.]]

        elif 4 == action[0]:  # Collect, type and num
            self.rivers = self.obs_process._find_rivers(self.obs).copy()
            self.action = [[0., 2., float(self.rivers[0][0]), float(self.rivers[0][1])]] if len(self.rivers) > 0 else [
                [0., 2., float(action[1]), float(action[2])]]

        elif 5 == action[0]:  # Pickup automatically
            material = self.obs_process._find_material(self.obs)
            backpack = self.obs_process._analyse_backpack(self.obs)
            if len(material) <= 0:
                self.action = [[0., 3., float(action[1]), float(action[2])]]
            else:
                if int(material[0][2]) == 70009:
                    if backpack[int(material[0][2])] >= 1:
                        self.action = [[0., 3., float(action[1]), float(action[2])]]
                    else:
                        self.action = [[0., 3., float(material[0][2]), 1.]]
                else:
                    if backpack[int(material[0][2])] >= 3:
                        self.action = [[0., 3., float(action[1]), float(action[2])]]
                    else:
                        self.action = [[0., 3., float(material[0][2]), 1.]]

        elif 6 == action[0]:  # Equip automatically
            backpack = self.obs_process._analyse_backpack(self.obs)
            if backpack[70009] <= 0:
                self.action = [[0., 5., float(action[1]), float(action[2])]]
            else:
                self.action = [[0., 5., float(70009), 0.]]

        # run action and obtain new obs
        if self.obs is not None:
            self._prev_obs = self.obs.copy()
        self.env.step(self.action)
        self.obs = self.env.get_agent_observe()[0]
        obs_scale = self.obs_process._obs_scale(self.obs)
        reward, result_type = self._calc_reward(action[0])
        self.step_count += 1

        # step log
        # if self.step_count % 100 <= 70:  # day
        #     fprintNoHp((0, self.step_count, self.obs, action[0], reward, result))
        # else:  # night
        #     fprintNoHp((1, self.step_count, self.obs, action[0], reward, result))

        # update episode log
        updateEpisodeLog(self.episode_log_list, action[0], result_type, reward)

        # Done when the agent dies.
        if self.obs[1] <= 2.0 or self.obs[2] <= 2.0 or self.obs[3] <= 2.0:

            # show episode log
            showEpisodeLog(self.episode_log_list)

            # write episode log
            writeEpisodeLog(self.log_txt, self.episode_log_list)

            print("done")

            done = True
        else:
            done = False
        info = {}

        return obs_scale, reward, done, info

    # -------------------- calc_reward -------------------- #
    def _calc_reward(self, action_id):
        reward = 0.
        result = ""
        result_type = ""

        # calc idle reward
        if action_id == 0:
            pass

        # calc attack reward
        if action_id == 1:
            reward, result_type = self.reward._attack_pig_reward(self._prev_obs, self.obs, self._prev_pigs)
            # result = "| attack: {} | kill: {} |".format(attack, killed)

        # calc move reward
        if action_id == 2:
            pigs = self.obs_process._find_pigs(self.obs)
            reward, result_type = self.reward._move_reward(self._prev_obs, self.obs, self._prev_pigs, pigs)
            # result = "| move: {} |".format(move)

        # calc consume reward
        if action_id == 3:
            reward, result_type = self.reward._consume_reward(self._prev_obs, self.obs)
            # result = "| consume_type: {} |".format(consume_type)

        # calc collect reward
        if action_id == 4:
            reward, result_type = self.reward._collect_water_reward(self.obs)
            # result = "| collect_type: {} |".format(collect_type)

        # calc pickup reward
        if action_id == 5:
            reward, result_type = self.reward._pickup_material_reward(self._prev_obs, self.obs)
            # result = "| pick_type: {} |".format(pick_type)

        # calc equip reward
        # 待修改
        if action_id == 6:
            reward, result_type = self.reward._equip_torch_reward(self._prev_obs, self.obs)
            # result = "| equip_type: {} |".format(pick_type)

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
