"""
jihuang obs utils file
"""
import numpy as np


def _find_pigs(obs):
    """
    Get all pig information from current observation
    :param obs: Current observation
    :return: list of all pigs with its x, y, hp
    """
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


def _find_rivers(obs):
    """
    Get all rivers information from current observation
    :param obs: Current observation
    :return: list of all rivers with its x, y
    """
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


def _find_material(obs):
    """
    Get all material information (including water、meat、torch and so on) from current observation
    :param obs: Current observation
    :return: list of all materials with its x, y, id
    """
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


def _material_process_pickup(obs, materials):
    """
    Get All materials within the field of view that are within a small distance
        from the agent and the pick-up distance
    :param obs: Current observation
    :param materials: All materials in the field of view
    :return: list of all materials with small pickup distance, including x, y
    """
    x_init = obs[5]
    y_init = obs[6]
    materials_copy = materials.copy()
    new_materials = []
    for i in materials_copy:
        distance = np.sqrt((i[0] - x_init) ** 2 + (i[1] - y_init) ** 2)
        if distance <= 10.1:
            new_materials.append(i)

    return new_materials


def _find_equipment(obs):
    """
    Get the torch equipment information currently equipped
    :param obs: Current observation
    :return: dict of torch
    """
    if obs is None:
        return []
    equipment_obs = obs[61:83].copy()
    equipments = {70009: 0}
    for idx in range(int(len(equipment_obs) / 2)):
        if int(equipment_obs[idx * 2]) == 70009:  # torch
            equipments[70009] += 1

    return equipments


def _find_buff(obs):
    """
    Get all buffs information
    :param obs: Current observation
    :return: dict of buffs information
    """
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


def _analyse_backpack(obs):
    """
    Get information about backpack items
    :param obs: Current observation
    :return: dict of backpack items information
    """
    if obs is None:
        return []
    backpack_info = obs[13:61].copy()
    backpack = {30001: 0, 30002: 0, 70009: 0}
    for idx in range(int(len(backpack_info) / 2)):
        if int(backpack_info[idx * 2]) in backpack.keys():
            backpack[int(backpack_info[idx * 2])] += 1
    return backpack


def _analyse_materials(materials):
    """
    Get the specific type and count of current field of view material
    :param materials: Information of current field of view material
    :return: List of all material types and count
    """
    meat = []
    water = []
    torch = []
    if len(materials) <= 0:
        return water, meat, torch
    else:
        for i in materials:
            if i[2] == 30001.:
                water.append(i)
            elif i[2] == 30002.:
                meat.append(i)
            elif i[2] == 70009.:
                torch.append(i)
        return water, meat, torch
