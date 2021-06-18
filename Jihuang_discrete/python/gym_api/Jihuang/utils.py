# from Jihuang.reward import JihuangReward
import os
from Jihuang.obs_utils import JihuangObsProcess
from datetime import datetime


def list_dict_contain(list_dict, obj):
    for i in list_dict:
        if i[2] == obj:
            return True
    return False



# int action to string action
def action2string(action):
    action_id = action
    if action_id == 0:
        return "Idle"
    elif action_id == 1:
        return "Attack"
    elif action_id == 2:
        return "Collect"
    elif action_id == 3:
        return "Pickup"
    elif action_id == 4:
        return "Consume"
    elif action_id == 5:
        return "Equip"
    elif action_id == 6:
        return "Move"


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

0 -> Idle: count

1 -> Attack: type, count, reward, attack_count, attack_reward, kill_count, kill_reward, fail_count

2 -> Move: type, count, reward, success_count, success_reward, fail_count

3 -> Consume: type, count, reward, meat_count, meat_reward, water_count, water_reward, fail_count

4 -> Collect: type, count, reward, water_count, water_reward, fail_count

5 -> Pickup: type, count, reward, meat_count, meat_reward, water_count, water_reward, fail_count

6 -> Equip: type, count, reward, torch_count, torch_reward, fail_count

"""


def initEpisodeLog():
    log_list = []
    log_dict_head = {}
    log_dict_head["type"] = "Result"
    log_dict_head["step"] = 0
    log_dict_head["reward"] = 0.
    log_list.append(log_dict_head)

    log_dict_Idle = {}
    log_dict_Idle["type"] = "Idle"
    log_dict_Idle["count"] = 0
    log_list.append(log_dict_Idle)

    log_dict_Attack = {}
    log_dict_Attack["type"] = "Attack"
    log_dict_Attack["count"] = 0
    log_dict_Attack["reward"] = 0.
    log_dict_Attack["attack_count"] = 0
    log_dict_Attack["attack_reward"] = 0.
    log_dict_Attack["kill_count"] = 0
    log_dict_Attack["kill_reward"] = 0.
    log_dict_Attack["fail_count"] = 0
    log_list.append(log_dict_Attack)

    log_dict_Move = {}
    log_dict_Move["type"] = "Move"
    log_dict_Move["count"] = 0
    log_dict_Move["reward"] = 0.
    log_dict_Move["success_count"] = 0
    log_dict_Move["success_reward"] = 0.
    log_dict_Move["fail_count"] = 0
    log_list.append(log_dict_Move)

    log_dict_Consume = {}
    log_dict_Consume["type"] = "Consume"
    log_dict_Consume["count"] = 0
    log_dict_Consume["reward"] = 0.
    log_dict_Consume["meat_count"] = 0
    log_dict_Consume["meat_reward"] = 0.
    log_dict_Consume["water_count"] = 0
    log_dict_Consume["water_reward"] = 0.
    log_dict_Consume["fail_count"] = 0
    log_list.append(log_dict_Consume)

    log_dict_Collect = {}
    log_dict_Collect["type"] = "Collect"
    log_dict_Collect["count"] = 0
    log_dict_Collect["reward"] = 0.
    log_dict_Collect["water_count"] = 0
    log_dict_Collect["water_reward"] = 0.
    log_dict_Collect["fail_count"] = 0
    log_list.append(log_dict_Collect)

    log_dict_Pickup = {}
    log_dict_Pickup["type"] = "Pickup"
    log_dict_Pickup["count"] = 0
    log_dict_Pickup["reward"] = 0.
    log_dict_Pickup["meat_count"] = 0
    log_dict_Pickup["meat_reward"] = 0.
    log_dict_Pickup["water_count"] = 0
    log_dict_Pickup["water_reward"] = 0.
    log_dict_Pickup["fail_count"] = 0
    log_list.append(log_dict_Pickup)

    log_dict_Equip = {}
    log_dict_Equip["type"] = "Equip"
    log_dict_Equip["count"] = 0
    log_dict_Equip["reward"] = 0.
    log_dict_Equip["torch_count"] = 0
    log_dict_Equip["torch_reward"] = 0.
    log_dict_Equip["fail_count"] = 0.
    log_list.append(log_dict_Equip)

    return log_list


def showEpisodeLog(log_list):
    # print("------------------------------------------------------------------")
    for i in log_list:
        print(i)
    print()


def writeEpisodeLog(path, log_list):
    path = os.getcwd() + '/python/gym_api/Jihuang/' + path
    if os.path.exists(path):
        f = open(path, 'a+', encoding='utf-8')
        txt = "------------------------------------------------------------------\n"
        for i in log_list:
            txt = txt + str(i) + "\n"
        txt = txt + "\n\n"
        f.write(txt)
        f.close()
    else:
        f = open(path, 'w', encoding='utf-8')
        txt = "------------------------------------------------------------------\n"
        for i in log_list:
            txt = txt + str(i) + "\n"
        txt = txt + "\n\n"
        f.write(txt)
        f.close()


def updateEpisodeLog(log_list, action_type, result_type, reward):
    action_id = action_type
    log_list[0]["step"] += 1
    log_list[0]["reward"] += reward

    # Idle
    if action_id == 0:
        log_list[1]["count"] += 1

    # Attack
    elif action_id == 1:
        log_list[2]["count"] += 1
        log_list[2]["reward"] += reward
        if result_type == "attack":
            log_list[2]["attack_count"] += 1
            log_list[2]["attack_reward"] += reward
        if result_type == "kill":
            log_list[2]["kill_count"] += 1
            log_list[2]["kill_reward"] += reward
        if result_type == "fail":
            log_list[2]["fail_count"] += 1

    # Move
    elif action_id == 2:
        log_list[3]["count"] += 1
        log_list[3]["reward"] += reward
        if result_type == "move":
            log_list[3]["success_count"] += 1
            log_list[3]["success_reward"] += reward
        if result_type == "fail":
            log_list[3]["fail_count"] += 1

    # Consume
    elif action_id == 3:
        log_list[4]["count"] += 1
        log_list[4]["reward"] += reward
        if result_type == "meat":
            log_list[4]["meat_count"] += 1
            log_list[4]["meat_reward"] += reward

        if result_type == "water":
            log_list[4]["water_count"] += 1
            log_list[4]["water_reward"] += reward

        if result_type == "fail":
            log_list[4]["fail_count"] += 1

    # Collect
    elif action_id == 4:
        log_list[5]["count"] += 1
        log_list[5]["reward"] += reward
        if result_type == "water":
            log_list[5]["water_count"] += 1
            log_list[5]["water_reward"] += reward
        if result_type == "fail":
            log_list[5]["fail_count"] += 1

    # Pickup
    elif action_id == 5:
        log_list[6]["count"] += 1
        log_list[6]["reward"] += reward
        if result_type == "meat":
            log_list[6]["meat_count"] += 1
            log_list[6]["meat_reward"] += reward

        if result_type == "water":
            log_list[6]["water_count"] += 1
            log_list[6]["water_reward"] += reward

        if result_type == "fail":
            log_list[6]["fail_count"] += 1

    # Equip
    elif action_id == 6:
        log_list[7]["count"] += 1
        log_list[7]["reward"] += reward
        if result_type == "torch":
            log_list[7]["torch_count"] += 1
            log_list[7]["torch_reward"] += reward
        if result_type == "fail":
            log_list[7]["fail_count"] += 1


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
