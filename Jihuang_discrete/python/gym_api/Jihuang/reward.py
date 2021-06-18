import numpy as np
from Jihuang.utils import dictCompare
from Jihuang.obs_utils import JihuangObsProcess
from Jihuang.config import water_meat_count, torch_count


class JihuangReward:
    def __init__(self):
        self.obs_utils = JihuangObsProcess()

    def _move_reward(self, prev_obs, obs, prev_pigs, pigs):
        result_type = "fail"

        move_reward_pig = 0
        prev_pigs = prev_pigs.copy()
        prev_x_init = prev_obs[5]
        prev_y_init = prev_obs[6]
        x_init = obs[5]
        y_init = obs[6]

        if len(prev_pigs) == 0 and len(pigs) > 0:
            move_reward_pig = 3.
            result_type = "move"
        if len(prev_pigs) > 0 and len(pigs) > 0:
            prev_pig_distance = np.sqrt(
                (prev_pigs[0][0] - prev_x_init) ** 2 + (prev_pigs[0][1] - prev_y_init) ** 2)  # 上个状态与猪的距离
            pig_distance = np.sqrt((pigs[0][0] - x_init) ** 2 + (pigs[0][1] - y_init) ** 2)   # 当前状态与猪的距离

            if prev_pig_distance - pig_distance > 0:
                move_reward_pig = 2.
                result_type = "move"

        return move_reward_pig, result_type

    def _attack_pig_reward(self, prev_obs, obs, prev_pigs):
        prev_pigs = prev_pigs.copy()
        result_type = "fail"

        if len(prev_pigs) == 0:
            attack_reward = 0
        else:
            x_init = prev_obs[5]
            y_init = prev_obs[6]
            attack_pig_distance = np.sqrt(
                (x_init - prev_pigs[0][0]) ** 2 + ((y_init - prev_pigs[0][1])) ** 2)  # 攻击之前的距离
            if attack_pig_distance <= 4.1:
                if prev_pigs[0][2] > 60:  # 一次攻击 60 hp
                    attack_reward = 10
                    result_type = "attack"
                else:
                    attack_reward = 1000
                    result_type = "kill"
            else:
                attack_reward = 1.0 / abs(attack_pig_distance)
        return attack_reward, result_type

    # ---------- consume reward design ---------- #
    def _consume_reward(self, prev_obs, obs):
        result_type = "fail"
        if obs[7] == 0.0:
            consume_reward = 0
        else:
            consume_reward = 0
            consume_reward += max(obs[2] - prev_obs[2] + 2, 0)
            if consume_reward > 0:
                if consume_reward < 30:
                    consume_reward /= 2.
                result_type = "meat"
            else:
                consume_reward += max(obs[3] - prev_obs[3] + 2, 0)
                if consume_reward > 0:
                    if consume_reward < 30:
                        consume_reward /= 2.
                    result_type = "water"

        return consume_reward, result_type

    # ---------- collect water reward design ---------- #
    def _collect_water_reward(self, obs):
        result_type = "fail"
        if obs[7] == 0.0:
            collect_reward = 0
        else:
            # print("obs[7]", obs[7])
            collect_reward = 2.
            result_type = "water"
        return collect_reward, result_type

    # ---------- pickup material reward design ---------- #
    def _pickup_material_reward(self, prev_obs, obs):
        result_type = "fail"
        if obs[7] == 0.0:
            pickup_reward = 0.
        else:
            pickup_reward = 0.
            prev_pack = self.obs_utils._analyse_backpack(prev_obs)
            pack = self.obs_utils._analyse_backpack(obs)
            list_key = dictCompare(prev_pack, pack)
            for key in list_key:
                if key == 30001:
                    if pack[30001] < water_meat_count:
                        pickup_reward = 10.
                    result_type = "water"
                if key == 30002:
                    if pack[30002] < water_meat_count:
                        pickup_reward = 10.
                    result_type = "meat"
                if key == 70009:
                    if pack[70009] < torch_count:
                        pickup_reward = 5.
                    result_type = "torch"

        return pickup_reward, result_type

    # ---------- equip torch reward design ---------- #
    def _equip_torch_reward(self, prev_obs, obs):
        result_type = "false"
        if obs[7] == 0.0:
            equip_reward = 0
        else:
            prev_equipments = self.obs_utils._find_equipment(prev_obs)
            equipments = self.obs_utils._find_equipment(obs)
            list_key_equipments = dictCompare(prev_equipments, equipments)

            prev_buffs = self.obs_utils._find_buff(prev_obs)
            buffs = self.obs_utils._find_buff(obs)
            list_key_buffs = dictCompare(prev_buffs, buffs, key=1001)

            equip_reward = 0
            equip_type = ""

            prev_pack = self.obs_utils._analyse_backpack(prev_obs)
            pack = self.obs_utils._analyse_backpack(obs)
            list_key = dictCompare(prev_pack, pack)

            # print("length:", len(list_key_equipments), len(list_key_buffs))
            for key in list_key:
                if key == 70009 and len(list_key_equipments) > 0:
                    equip_reward = 10.
                    result_type = "torch"

        return equip_reward, result_type
