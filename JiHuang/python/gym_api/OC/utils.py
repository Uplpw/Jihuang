pro_state = [i for i in range(512)]  # 2 ** 9 = 512

index = [8 - i for i in range(9)]


def state2int(state):
    """
    饱食度、饥渴度、昼夜、背包water、背包meat、视野pig、视野river、视野water、视野meat、
    """
    satiety = 0
    thirsty = 0
    day_night = 0
    backpack_water = 0
    backpack_meat = 0
    vision_pig = 0
    vision_river = 0
    vision_water = 0
    vision_meat = 0

    # basic state
    if state[2] <= 50.:
        satiety = 1
    if state[3] <= 50.:
        thirsty = 1
    if state[9] == 1.:
        day_night = 1

    # backpack
    backpack = _analyse_backpack(state)
    if backpack[30001] > 0.:
        backpack_water = 1
    if backpack[30002] > 0.:
        backpack_meat = 1

    # vision
    if state[63] == 0. and state[64] == 0. and state[65] == 0.:
        vision_pig = 0
    else:
        vision_pig = 1
    if state[66] == 0. and state[67] == 0.:
        vision_river = 0
    else:
        vision_river = 1

    if state[68] == 0. and state[69] == 0. and state[70] == 0.:
        vision_water = 0
    else:
        vision_water = 1

    if state[71] == 0. and state[72] == 0. and state[73] == 0.:
        vision_meat = 0
    else:
        vision_meat = 1

    s = 2 ** index[0] * satiety + 2 ** index[1] * thirsty + 2 ** index[2] * day_night + \
        2 ** index[3] * backpack_water + 2 ** index[4] * backpack_meat + 2 ** index[5] * vision_pig + \
        2 ** index[6] * vision_river + 2 ** index[7] * vision_water + 2 ** index[8] * vision_meat

    return s

def _analyse_backpack(obs):
    if obs is None:
        return []
    backpack_info = obs[13:61].copy()
    backpack = {30001: 0, 30002: 0, 70009: 0}
    for idx in range(int(len(backpack_info) / 2)):
        if int(backpack_info[idx * 2]) in backpack.keys():
            backpack[int(backpack_info[idx * 2])] += 1
    return backpack


# print(pro_state)
# print(index)
