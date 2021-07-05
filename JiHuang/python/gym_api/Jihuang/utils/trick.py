import random
import numpy as np
from .utils import list_dict_contain, random_find_pig
from .obs_utils import _find_pigs, _find_rivers, _analyse_backpack, _material_process_pickup, _find_material, \
    _analyse_materials
from .args import water_meat_count, torch_count, equip_count


def select_target_func(obs, action: int):
    """
    Choose different action parameters according to the action type
    :param obs: Current observation
    :param action: Current action
    :return: action parameters
    """
    if action <= 2:
        if action == 0:
            return 0, 0
        if action == 1:
            pigs = _find_pigs(obs)
            if len(pigs) <= 0:
                return random.randint(0, 40), random.randint(0, 40)
            return pigs[0][0], pigs[0][1]
        if action == 2:
            rivers = _find_rivers(obs)
            if len(rivers) <= 0:
                return random.randint(0, 40), random.randint(0, 40)
            return rivers[0][0], rivers[0][1]
    elif action <= 5:
        if action == 3:  # pickup
            backpack = _analyse_backpack(obs)
            material = _material_process_pickup(obs, _find_material(obs))
            if backpack[70009] <= 0:
                if list_dict_contain(material, 70009):
                    return 70009, 1
                elif list_dict_contain(material, 30001) and list_dict_contain(material, 30002):
                    if backpack[30001] < backpack[30002]:
                        return 30001, 1
                    else:
                        return 30002, 1
                elif list_dict_contain(material, 30001) == True and list_dict_contain(material, 30002) == False:
                    return 30001, 1
                elif list_dict_contain(material, 30001) == False and list_dict_contain(material, 30002) == True:
                    return 30002, 1
                else:
                    return 0, 0
            else:
                if list_dict_contain(material, 30001) and list_dict_contain(material, 30002):
                    if backpack[30001] < backpack[30002]:
                        return 30001, 1
                    else:
                        return 30002, 1
                elif list_dict_contain(material, 30001) == True and list_dict_contain(material, 30002) == False:
                    return 30001, 1
                elif list_dict_contain(material, 30001) == False and list_dict_contain(material, 30002) == True:
                    return 30002, 1
                else:
                    return 0, 0

        if action == 4:  # consume
            satiety = obs[2]
            thirsty = obs[3]
            if satiety <= thirsty:
                return 30002, 1
            else:
                return 30001, 1

        if action == 5:
            return 70009, 1

    elif action == 6:  # move, pigs(*)/material
        x_init, y_init = obs[5], obs[6]
        pigs = _find_pigs(obs)
        # materials = _find_material(obs)
        if len(pigs) > 0:
            pig_x, pig_y = pigs[0][0], pigs[0][1]
            agent_pig_distance = np.sqrt((pig_x - x_init) ** 2 + (pig_y - y_init) ** 2)
            offset_x, offset_y = 4.1 * (pig_x - x_init) / agent_pig_distance, 4.1 * (
                    pig_y - y_init) / agent_pig_distance
            return int(offset_x), int(offset_y)
        else:
            offset_x, offset_y = random_find_pig()
            return offset_x, offset_y


def action_mask_from_obs(observations):
    """
    Mask some unreasonable or doomed actions based on current observations
    :param observations: Current observations
    :return: A list of all actions, if value of the mask action is 1, otherwise 0
    """
    if observations is None:
        return []
    if type(observations) == list:
        obs = np.array(observations)
    elif type(observations) != np.ndarray:
        obs = np.array(observations.cpu()).copy()
    else:
        obs = observations.copy()
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    action_mask = np.zeros((obs.shape[0], 7))
    for i in range(obs.shape[0]):
        # 7 action
        # 0 idle
        if obs[i][9] == 0.0:  # day
            action_mask[i][0] = 1  # idle
        else:
            action_mask[i][0] = 0  # idle

        # 1 attack
        # note: x, y of pig
        if obs[i][63] == 0. and obs[i][64] == 0. and obs[i][65] == 0.:
            action_mask[i][1] = 1
        else:
            x_init = obs[i][5]
            y_init = obs[i][6]
            attack_pig_distance = np.sqrt((x_init - obs[i][63]) ** 2 + (y_init - obs[i][64]) ** 2)
            if attack_pig_distance > 4.1:
                action_mask[i][1] = 1

        # 2 Collect
        if obs[i][66] == 0. and obs[i][67] == 0.:
            action_mask[i][2] = 1

        # 3 pickup
        flag = True
        for k in range(9):
            if obs[i][68 + k] != 0.:
                flag = False
                break
        if flag:
            action_mask[i][3] = 1

        # 4 Consume
        backpack = _analyse_backpack(obs[i])
        if backpack[30001] <= 0 and backpack[30002] <= 0:
            action_mask[i][4] = 1
        if obs[i][2] >= 50 and obs[i][3] >= 50:
            action_mask[i][4] = 1

        # 5 equip

        if obs[i][9] == 0.0 or backpack[70009] <= 0 or obs[i][61] != 0:
            action_mask[i][5] = 1

        # 2 move
        # action_mask[6] = 1

    return action_mask


def _obs_scale(obs):
    """
    Delete some attributes of observations as needed to achieve scaling of observations
    :param obs: Current observation
    :return: new scaling of observations
    note: size of observations is changed
    """

    obs_ = obs.copy()
    obs_scale = []

    # block 1：基本数据（编号、hp、饱食度、饥渴度、体温、x、y、动作结果、时间、昼夜、季节、格子地貌、格子天气）= 13
    # index: 0-12
    # (饱食度、饥渴度、x、y、动作结果、昼夜) 0-6

    obs_block_1 = obs_[:13].copy()  # sum = 6
    # (0, 13)
    for i in range(len(obs_block_1)):
        obs_scale.append(obs_block_1[i])

    # block 2：背包（每个格子包含 item id 与 耐久度）* 24 = 48
    # (13 - 61)
    obs_block_2 = obs_[13:61].copy()
    for i in range(len(obs_block_2)):
        obs_scale.append(obs_block_2[i])

    # block 3：装备（武器、防具、首饰） 2 + 4 *5（tools最大值） = 22
    # index: 61-82
    # (61 - 63)
    obs_block_3 = obs_[61:(61 + 2 * equip_count)].copy()  # torch = 1, sum = 1 * 2 = 2
    for i in range(len(obs_block_3)):
        obs_scale.append(obs_block_3[i])

    # block 4：Buff（buff_id 饱食度 饥渴度 血量 温度 攻击力 防御力 移动速度 视野）*10
    # index: 83-172
    # obs_block_4 = obs_[61:63].copy()  # torch = 1, sum = 1*2=2
    # for i in range(83, 173):
    #     obs_scale.append(obs_block_4[i])

    # block 5：视野第一分类（agent animal plant resource）7 * 17 * 17 = 2023
    # (63 - 68)
    pigs = _find_pigs(obs)
    rivers = _find_rivers(obs)
    if len(pigs) > 0:
        obs_scale.append(pigs[0][0])
        obs_scale.append(pigs[0][1])
        obs_scale.append(pigs[0][2])
    else:
        obs_scale.append(0.)
        obs_scale.append(0.)
        obs_scale.append(0.)

    if len(rivers) > 0:
        obs_scale.append(rivers[0][0])
        obs_scale.append(rivers[0][1])
    else:
        obs_scale.append(0.)
        obs_scale.append(0.)

    # block 6：material（类型, x坐标, y坐标） 3 * 17 * 17 = 867
    # index: 2196-3062
    # water meat and torch = 1  sum = 3 * 3 = 9
    # (68 - 77)
    material = _find_material(obs)
    water, meat, torch = _analyse_materials(material)
    if len(water) > 0:  # water
        obs_scale.append(water[0][0])
        obs_scale.append(water[0][1])
        obs_scale.append(water[0][2])
    else:
        obs_scale.append(0.)
        obs_scale.append(0.)
        obs_scale.append(0.)

    if len(meat) > 0:  # meat
        obs_scale.append(meat[0][0])
        obs_scale.append(meat[0][1])
        obs_scale.append(meat[0][2])
    else:
        obs_scale.append(0.)
        obs_scale.append(0.)
        obs_scale.append(0.)

    if len(torch) > 0:  # torch
        obs_scale.append(torch[0][0])
        obs_scale.append(torch[0][1])
        obs_scale.append(torch[0][2])
    else:
        obs_scale.append(0.)
        obs_scale.append(0.)
        obs_scale.append(0.)

    # block 7：地貌 [x坐标, y坐标, 天气，地貌]  4 * 17 * 17 = 1156
    # index: 3063-4218
    # obs_block_7 = obs_[].copy()
    # for i in range(len(obs_block_7)):
    #     pass

    # sum = 13 + 48 + 2 + 0 + 5 + 9 + 0 = 77
    return obs_scale


def _obs_mask(obs):
    """
    Mask some attributes of the observations as needed
    :param obs: Current observation
    :return: new observation
    """
    obs_mask = obs.copy()
    # block 1：基本数据（编号、hp、饱食度、饥渴度、体温、x、y、动作结果、时间、昼夜、季节、格子地貌、格子天气）
    # for i in range(0, 13):

    # block 2：背包（每个格子包含 item id 与 耐久度）*24
    # for i in range(13, 61):
    #     obs_mask[i] = 0.

    # block 3：装备（武器、防具、首饰）
    # for i in range(61, 83):
    #     obs_mask[i] = 0.

    # block 4：Buff（buff_id 饱食度 饥渴度 血量 温度 攻击力 防御力 移动速度 视野）*10
    # for i in range(83, 173):
    #     obs_mask[i] = 0.

    # # block 5：视野第一分类（agent animal plant resource）7*17*17
    # for i in range(173, 2196):

    # block 6：material（类型, x坐标, y坐标） 3 * 17 * 17
    # for i in range(2196, 3063):
    #     obs_mask[i] = 0.

    # block 7：地貌 [x坐标, y坐标, 天气，地貌]  4 * 17 * 17
    for i in range(3063, 4219):
        obs_mask[i] = 0.

    return obs_mask
