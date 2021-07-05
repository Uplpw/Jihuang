"""
jihuang obs utils file
"""


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
