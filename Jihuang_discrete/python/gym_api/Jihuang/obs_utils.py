import numpy as np
from torch import Tensor
from Jihuang.config import water_meat_count, torch_count


class JihuangObsProcess:
    def __init__(self):
        pass

    def _obs_mask(self, obs):
        obs_mask = obs.copy()
        # block 1：基本数据（编号、hp、饱食度、饥渴度、体温、x、y、动作结果、时间、昼夜、季节、格子地貌、格子天气）
        # for i in range(0, 13):

        # block 2：背包（每个格子包含 item id 与 耐久度）*24
        # for i in range(13, 61):
        #     obs_mask[i] = 0.

        # block 3：装备（武器、防具、首饰）
        for i in range(61, 83):
            obs_mask[i] = 0.

        # block 4：Buff（buff_id 饱食度 饥渴度 血量 温度 攻击力 防御力 移动速度 视野）*10
        for i in range(83, 173):
            obs_mask[i] = 0.

        # # block 5：视野第一分类（agent animal plant resource）7*17*17
        # for i in range(173, 2196):

        # block 6：material（类型, x坐标, y坐标） 3 * 17 * 17
        # for i in range(2196, 3063):
        #     obs_mask[i] = 0.

        # block 7：地貌 [x坐标, y坐标, 天气，地貌]  4 * 17 * 17
        for i in range(3063, 4219):
            obs_mask[i] = 0.

        return obs_mask

    def _find_pigs(self, obs):
        # get agent-animal-plant-resources observation
        if obs is None:
            return []
        aapr_obs = obs[173:2196].copy()
        pigs = []
        x_init = obs[5]
        y_init = obs[6]
        for idx in range(int(len(aapr_obs) / 7)):
            if int(aapr_obs[idx * 7]) == 2:  # pig
                pigs.append((aapr_obs[idx * 7 + 1], aapr_obs[idx * 7 + 2], aapr_obs[idx * 7 + 3]))
        pigs.sort(key=lambda x: (x[0] - x_init) ** 2 + (x[1] - y_init) ** 2)
        return pigs

    # ---------- find river from env ---------- #
    def _find_rivers(self, obs):
        if obs is None:
            return []
        aapr_obs = obs[173:2196].copy()
        rivers = []
        x_init = obs[5]
        y_init = obs[6]
        for idx in range(int(len(aapr_obs) / 7)):
            if int(aapr_obs[idx * 7]) == 10005:  # River
                rivers.append((aapr_obs[idx * 7 + 1], aapr_obs[idx * 7 + 2]))
        rivers.sort(key=lambda x: (x[0] - x_init) ** 2 + (x[1] - y_init) ** 2)
        return rivers

    # ---------- find material from env ---------- #
    def _find_material(self, obs):
        if obs is None:
            return []
        material_obs = obs[2196:3063].copy()
        materials = []
        x_init = obs[5]
        y_init = obs[6]
        for idx in range(int(len(material_obs) / 3)):
            if int(material_obs[idx * 3]) == 30002:  # meat
                materials.append((material_obs[idx * 3 + 1], material_obs[idx * 3 + 2], material_obs[idx * 3]))
            if int(material_obs[idx * 3]) == 30001:  # water
                materials.append((material_obs[idx * 3 + 1], material_obs[idx * 3 + 2], material_obs[idx * 3]))
            if int(material_obs[idx * 3]) == 70009:  # torch
                materials.append((material_obs[idx * 3 + 1], material_obs[idx * 3 + 2], material_obs[idx * 3]))
        materials.sort(key=lambda x: (x[0] - x_init) ** 2 + (x[1] - y_init) ** 2)
        return materials


    def _material_process_pickup(self, obs, materials):
        x_init = obs[5]
        y_init = obs[6]
        materials_copy = materials.copy()
        new_materials = []
        for i in materials_copy:
            distance = np.sqrt((i[0] - x_init) ** 2 + (i[1] - y_init) ** 2)
            if distance <= 10.1:
                new_materials.append(i)
        return new_materials

    # ---------- find equipment from env ---------- #
    def _find_equipment(self, obs):
        if obs is None:
            return []
        equipment_obs = obs[61:83].copy()
        equipments = {70009: 0}
        for idx in range(int(len(equipment_obs) / 2)):
            if int(equipment_obs[idx * 2]) == 70009:  # torch
                # print("------------------------------find equipment: Torch----------------------------")
                equipments[70009] += 1

        return equipments

    # ---------- find buff from env ---------- #
    def _find_buff(self, obs):
        if obs is None:
            return []
        buff_obs = obs[83:173].copy()
        buffs = {1001: 0, 3001: 0}
        for idx in range(int(len(buff_obs) / 9)):
            if int(buff_obs[idx * 9]) == 1001:  # night vision
                buffs[1001] += 1
            if int(buff_obs[idx * 9]) == 3001:  # night vision
                buffs[3001] += 1
        return buffs

    # ---------- analyse items from backpack ---------- #
    def _analyse_backpack(self, obs):
        if obs is None:
            return []
        backpack_info = obs[13:61].copy()
        backpack = {30001: 0, 30002: 0, 70009: 0}
        for idx in range(int(len(backpack_info) / 2)):
            if int(backpack_info[idx * 2]) in backpack.keys():
                backpack[int(backpack_info[idx * 2])] += 1
        return backpack

    def _analyse_backpack_1(self, obs):
        if obs is None:
            return []
        backpack_info = obs[6:28].copy()
        backpack = {30001: 0, 30002: 0, 70009: 0}
        for idx in range(int(len(backpack_info) / 2)):
            if int(backpack_info[idx * 2]) in backpack.keys():
                backpack[int(backpack_info[idx * 2])] += 1
        return backpack

    def action_mask_from_obs(self, obs_tensor):
        if obs_tensor is None:
            return []
        if type(obs_tensor) != np.ndarray:
            obs = np.array(obs_tensor.cpu()).copy()
        else:
            obs = obs_tensor.copy()
        action_mask = np.zeros((obs_tensor.shape[0], 87))
        for i in range(obs_tensor.shape[0]):
            # 7 action

            # 0 idle
            action_mask[i][0] = 1  # idle

            # 1 attack
            if obs[i][30] == 0. and obs[i][31] == 0. or obs[i][32] == 0.:
                action_mask[i][1] = 1
            else:
                x_init = obs[i][2]
                y_init = obs[i][3]
                attack_pig_distance = np.sqrt((x_init - obs[i][30]) ** 2 + (y_init - obs[i][31]) ** 2)
                if attack_pig_distance > 4.1:
                    action_mask[i][1] = 1

            # 2 move
            # action_mask[2] = 1

            # 3 Consume
            backpack = self._analyse_backpack_1(obs[i])
            if backpack[30001] <= 0 and backpack[30002] <= 0:
                action_mask[i][3] = 1
            if obs[i][0] >= 50 and obs[i][1] >= 50:
                action_mask[i][3] = 1

            # 4 Collect
            if obs[i][33] == 0. and obs[i][34] == 0.:
                action_mask[i][1] = 1

            # 5 pickup
            if obs[i][35] == 0. and obs[i][36] == 0. or obs[i][37] == 0.:
                action_mask[i][5] = 1

            # 6 equip
            if backpack[70009] <= 0:
                action_mask[i][6] = 1

        return action_mask

    #  obs scale
    def _obs_scale(self, obs):
        obs_ = obs.copy()
        obs_scale = []

        # block 1：基本数据（编号、hp、饱食度、饥渴度、体温、x、y、动作结果、时间、昼夜、季节、格子地貌、格子天气）= 13
        # index: 0-12
        # (饱食度、饥渴度、x、y、动作结果、昼夜) 0-6
        obs_block_1 = obs_[:13].copy()  # sum = 6
        obs_scale.append(obs_block_1[2])  #
        obs_scale.append(obs_block_1[3])
        obs_scale.append(obs_block_1[5])
        obs_scale.append(obs_block_1[6])
        obs_scale.append(obs_block_1[7])
        obs_scale.append(obs_block_1[9])

        # block 2：背包（每个格子包含 item id 与 耐久度）* 24 = 48
        # index: 13 - 60
        # meat and water = water_meat_count, torch = torch_count
        # sum = 2 * (2 * water_meat_count + torch_count)
        # 22 (6-28)
        end_index = 13 + 2 * (2 * water_meat_count + torch_count)
        obs_block_2 = obs_[13:end_index].copy()
        for i in range(len(obs_block_2)):
            obs_scale.append(obs_block_2[i])

        # block 3：装备（武器、防具、首饰） 2 + 4 *5（tools最大值） = 22
        # index: 61-82
        # 2 (28 - 30)
        obs_block_3 = obs_[61:63].copy()  # torch = 1, sum = 1 * 2 = 2
        for i in range(len(obs_block_3)):
            obs_scale.append(obs_block_3[i])

        # block 4：Buff（buff_id 饱食度 饥渴度 血量 温度 攻击力 防御力 移动速度 视野）*10
        # index: 83-172
        # obs_block_4 = obs_[61:63].copy()  # torch = 1, sum = 1*2=2
        # for i in range(83, 173):
        #     obs_scale.append(obs_block_4[i])

        # # block 5：视野第一分类（agent animal plant resource）7 * 17 * 17 = 2023
        # index: 173-2195
        # pig and river = 1, sum = 3 + 2 (x, y, hp/0) = 5
        # (30-35)
        pigs = self._find_pigs(obs)
        rivers = self._find_rivers(obs)
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
        # water meat or torch = 1  sum = 1 * 3 = 3
        # (35 - 38)
        material = self._find_material(obs)
        if len(material) > 0:
            obs_scale.append(material[0][0])
            obs_scale.append(material[0][1])
            obs_scale.append(material[0][2])
        else:
            obs_scale.append(0.)
            obs_scale.append(0.)
            obs_scale.append(0.)

        # block 7：地貌 [x坐标, y坐标, 天气，地貌]  4 * 17 * 17 = 1156
        # index: 3063-4218
        # obs_block_7 = obs_[].copy()
        # for i in range(len(obs_block_7)):
        #     pass

        # sum = 6 + 6 + 2 + 0 + 5 + 3 + 0 = 22
        return obs_scale