# from Jihuang.reward import JihuangReward
from Jihuang.obs_utils import JihuangObsProcess
from datetime import datetime


# int action to string action
def action2string(action):
    action_id = action
    if action_id == 0:
        action_string = "Idle"
        return action_string
    elif action_id == 1:
        action_string = "Attack"
        return action_string
    elif action_id == 2:
        action_string = "Move"
        return action_string
    elif action_id == 3:
        action_string = "Consume"
        return action_string
    elif action_id == 4:
        action_string = "Collect"
        return action_string
    elif action_id == 5:
        action_string = "Pickup"
        return action_string
    elif action_id == 6:
        action_string = "Equip"
        return action_string


# 规范化输出
def fprint(obj):
    day, step_count, obs, action, reward, result = obj
    if day == 0:
        print("|--白天--| Step: {:>2d} | Agent: {:>5f} {:>4f} {:>4f} | Action: {} | Reward: {} | Result: {}"
              .format(step_count, obs[1], obs[2], obs[3], action2string(action), reward, result))
    else:
        print("|--黑夜--| Step: {:>2d} | Agent: {:>5f} {:>4f} {:>4f} | Action: {} | Reward: {} | Result: {}"
              .format(step_count, obs[1], obs[2], obs[3], action2string(action), reward, result))


# 规范化输出
def fprintNoHp(obj):
    obs_utils_func = JihuangObsProcess()
    day, step_count, obs, action, reward, result = obj
    backpack = obs_utils_func._analyse_backpack(obs)
    equipment = obs_utils_func._find_equipment(obs)
    buff = obs_utils_func._find_buff(obs)
    if day == 0:
        print(
            "|白天| Step: {:>2d} | Agent: {:>2d} {:>2d} | Water Meat Torch: {} {} {} | Equip: {} "
            "| night vision: {} {} | Action: {} | Reward: {} | Result: {}"
                .format(step_count, int(obs[2]), int(obs[3]), backpack[30001], backpack[30002], backpack[70009],
                        equipment[70009], buff[3001], buff[1001], action2string(action), reward, result))
    else:
        print(
            "|黑夜| Step: {:>2d} | Agent: {:>2d} {:>2d} | Water Meat Torch: {} {} {} | Equip: {} "
            "| night vision: {} {} | Action: {} | Reward: {} | Result: {}"
                .format(step_count, int(obs[2]), int(obs[3]), backpack[30001], backpack[30002], backpack[70009],
                        equipment[70009], buff[3001], buff[1001], action2string(action), reward, result))


def dictCompare(dict1, dict2, key=None):
    list_key = []
    if key == None:
        for key in dict1.keys():
            if dict1[key] != dict2[key]:
                list_key.append(key)
    else:
        if dict1[key] != dict2[key]:
            list_key.append(key)
    return list_key


def getLogPath():
    current_time = datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
    path = "{}.txt".format(current_time)
    return path

"""
loglist: 

step: count

reward_sum: sum

# each action has its format -> dict

0 -> Idle: sum_count

1 -> Attack: type, sum_count, reward_sum, attack_count, attack_reward, kill_count, kill_reward, fail_count

2 -> Move: type, sum_count, reward_sum, success_count, success_reward, fail_count

3 -> Consume: type, sum_count, reward_sum, meat_s_count, meat_f_count, meat_reward, water_s_count, water_f_count, water_reward

4 -> Collect: type, sum_count, reward_sum, water_s_count

5 -> Pickup: type, sum_count, reward_sum, meat_s_count, meat_f_count, meat_reward, water_s_count, water_f_count, water_reward

6 -> Equip: type, sum_count, reward_sum, torch_s_count, torch_f_count

"""


def initEpisodeLog():
    log_list = {}
    log_list["step"] = 0
    log_list["sum_reward"] = 0.
    for i in range(7):
        log_list[i] = {}
        log_list[i]["count_sum"] = 0
        log_list[i]["action_result_type"] = {}
        for j in range(3):
            log_list[i]["action_result_type"][i] = 0
            log_list[i]["action_result_type"]["reward"] = 0.


def showEpisodeLog(log_list):
    print("|-------------------------------------episode-------------------------------------|")
    print("| step: {} | sum_reward: {} |".format(log_list["step"], log_list["sum_reward"]))

    print("|-------------------------action-------------------------|")
    for i in range(7):
        print("| {} : {} | success: {} | fail: {} | reward_sum: {} |"
              .format(action2string(i), log_list[i]["count_sum"], log_list[i]["success"], log_list[i]["fail"],
                      log_list[i]["reward"]))


def writeEpisodeLog(path, log_list):
    f = open(path, 'a+', encoding='utf-8')
    block1 = "|-------------------------------------episode-------------------------------------|\n"
    block2 = "| step: {} | sum_reward: {} |\n".format(log_list["step"], log_list["sum_reward"])
    block3 = "|-------------------------action-------------------------|"
    block4 = ""
    for i in log_list["action"]:
        block4 = block4 + "| {} : {} | success: {} | fail: {} | reward_sum: {} |\n".format(action2string(i),
                                                                                           log_list[i]["count_sum"],
                                                                                           log_list[i]["success"],
                                                                                           log_list[i]["fail"],
                                                                                           log_list[i]["reward"])
    txt = block1 + block2 + block3 + block4 + "\n\n"
    f.write(txt)
    f.close()


def updateEpisodeLog(log_list, action_type, result_type, reward):
    log_list["step"] += 1
    log_list["sum_reward"] += reward

    log_list[action_type]["count_sum"] += 1
    if result_type:
        log_list[action_type]["success"] += 1
    else:
        log_list[action_type]["fail"] += 1

    log_list[action_type]["reward"] += reward
