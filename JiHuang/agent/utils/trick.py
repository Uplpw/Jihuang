import numpy as np
from .obs_utils import _analyse_backpack


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
