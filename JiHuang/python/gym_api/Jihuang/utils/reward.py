import numpy as np
from .utils import dictCompare
from .obs_utils import _find_pigs, _find_rivers, _analyse_backpack, _material_process_pickup, _find_material, \
    _analyse_materials
from .args import water_meat_count, torch_count


# 根据前后状态变化设计 reward
def reward_func(action, prev_obs, obs):
    reward = 0.
    result_type = ""

    # calc idle reward
    if action == 0:
        pass

    # calc attack reward
    if action == 1:
        reward, result_type = _attack_pig_reward(prev_obs, obs)

    # calc collect reward
    if action == 2:
        reward, result_type = _collect_water_reward(obs)

    # calc pickup reward
    if action == 3:
        reward, result_type = _pickup_material_reward(prev_obs, obs)

    # calc consume reward
    if action == 4:
        reward, result_type = _consume_reward(prev_obs, obs)

    # calc equip reward
    if action == 5:
        reward, result_type = _equip_torch_reward(prev_obs, obs)

    # calc move reward
    if action == 6:
        reward, result_type = _move_reward(prev_obs, obs)

    return reward, result_type


def _move_reward(prev_obs, obs):
    result_type = "fail"
    move_reward_pig = 0

    prev_obs_copy = prev_obs.copy()
    obs_copy = obs.copy()
    prev_pigs = _find_pigs(prev_obs_copy)
    pigs = _find_pigs(obs_copy)

    prev_x_init = prev_obs[5]
    prev_y_init = prev_obs[6]
    x_init = obs[5]
    y_init = obs[6]

    if len(prev_pigs) == 0 and len(pigs) > 0:
        move_reward_pig = 1.
        result_type = "move"
    if len(prev_pigs) > 0 and len(pigs) > 0:
        prev_pig_distance = np.sqrt(
            (prev_pigs[0][0] - prev_x_init) ** 2 + (prev_pigs[0][1] - prev_y_init) ** 2)  # 上个状态与猪的距离
        pig_distance = np.sqrt((pigs[0][0] - x_init) ** 2 + (pigs[0][1] - y_init) ** 2)  # 当前状态与猪的距离

        if prev_pig_distance - pig_distance > 0:
            move_reward_pig = 1.
            result_type = "move"

    return move_reward_pig, result_type


def _attack_pig_reward(prev_obs, obs):
    attack_reward = 0
    result_type = "fail"

    prev_obs_copy = prev_obs.copy()
    prev_pigs = _find_pigs(prev_obs_copy)
    if len(prev_pigs) > 0:
        x_init = prev_obs[5]
        y_init = prev_obs[6]
        attack_pig_distance = np.sqrt(
            (x_init - prev_pigs[0][0]) ** 2 + ((y_init - prev_pigs[0][1])) ** 2)  # 攻击之前的距离
        if attack_pig_distance <= 4.1:
            if prev_pigs[0][2] > 60:  # 一次攻击 60 hp
                attack_reward = 5
                result_type = "attack"
            else:
                attack_reward = 50
                result_type = "kill"

    return attack_reward, result_type


# ---------- consume reward design ---------- #
def _consume_reward(prev_obs, obs):
    result_type = "fail"
    if obs[7] == 0.0:
        consume_reward = 0
    else:
        consume_reward = 0
        consume_reward += max(obs[2] - prev_obs[2] + 2, 0)
        if consume_reward > 0:
            if consume_reward < 40:
                consume_reward /= 2.
            result_type = "meat"
        else:
            consume_reward += max(obs[3] - prev_obs[3] + 2, 0)
            if consume_reward > 0:
                if consume_reward < 40:
                    consume_reward /= 2.
                result_type = "water"

    return consume_reward, result_type


# ---------- collect water reward design ---------- #
def _collect_water_reward(obs):
    result_type = "fail"
    if obs[7] == 0.0:
        collect_reward = 0
    else:
        collect_reward = 2.
        result_type = "water"
    return collect_reward, result_type


# ---------- pickup material reward design ---------- #
def _pickup_material_reward(prev_obs, obs):
    result_type = "fail"
    if obs[7] == 0.0:
        pickup_reward = 0.
    else:
        pickup_reward = 0.
        prev_pack = _analyse_backpack(prev_obs)
        pack = _analyse_backpack(obs)
        list_key = dictCompare(prev_pack, pack)
        for key in list_key:
            if key == 30001:
                if pack[30001] <= water_meat_count:
                    pickup_reward = 10.
                result_type = "water"
            if key == 30002:
                if pack[30002] <= water_meat_count:
                    pickup_reward = 10.
                result_type = "meat"
            if key == 70009:
                if pack[70009] <= torch_count:
                    pickup_reward = 10.
                result_type = "torch"

    return pickup_reward, result_type


# ---------- equip torch reward design ---------- #
def _equip_torch_reward(prev_obs, obs):
    result_type = "false"
    if obs[7] == 0.0:
        equip_reward = 0
    else:
        equip_reward = 10.
        result_type = "torch"

    return equip_reward, result_type
