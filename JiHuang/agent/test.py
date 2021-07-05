import numpy as np
from PPO.ppo import PPO
from DQN.dqn import DQN
from A2C.a2c import A2C


def test(env, algo, device):
    print("----------------------------- test start -----------------------------")
    if algo == "ppo":
        model = PPO.load("./agent/model/ppo_jihuang", device=device)
    elif algo == "dqn":
        model = DQN.load("./agent/model/dqn_jihuang", device=device)
    elif algo == "a2c":
        model = A2C.load("./agent/model/a2c_jihuang", device=device)
    else:
        raise NotImplementedError

    step_list = []
    action_seq = []
    for eposide in range(5):
        obs = env.reset()
        dones = False
        step = 0
        episode = []
        while not dones:
            action, _states = model.predict(obs)
            episode.append(int2str(action))
            obs, rewards, dones, info = env.step(action)
            step += 1
        step_list.append(step)
        action_seq.append(episode)

    print("mean:", np.mean(step_list), "max:", max(step_list), "min:", min(step_list))
    print("step:", step_list)
    # f = open('result.txt', 'w', encoding='utf-8')
    # for i in range(len(action_seq)):
    #     for j in range(len(action_seq[i])):
    #         if (j + 1) % 10 == 0:
    #             f.write(action_seq[i][j] + "\n")
    #         else:
    #             f.write(action_seq[i][j] + " ")
    #     f.write("\n\n\n")
    # f.close()


def int2str(action):
    if action == 0:
        return "Idle"
    if action == 1:
        return "Attack"
    if action == 2:
        return "Collect"
    if action == 3:
        return "Pickup"
    if action == 4:
        return "Consume"
    if action == 5:
        return "Equip"
    if action == 6:
        return "Move"
