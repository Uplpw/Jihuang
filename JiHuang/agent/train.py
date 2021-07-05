from PPO.ppo import PPO
from DQN.dqn import DQN
from A2C.a2c import A2C
from datetime import datetime


def train(env, algo, device):
    print("----------------------------- train start -----------------------------")
    train_start = datetime.now()
    if algo == "ppo":
        model = PPO(env, action_mask=True, device=device, verbose=1, tensorboard_log="./ppo_run/")
        model.learn(total_timesteps=int(1000000))
        model.save("ppo_jihuang")
        del model
    elif algo == "dqn":
        model = DQN("MlpPolicy", env, action_mask=True, verbose=1, tensorboard_log='./dqn_run/')
        model.learn(total_timesteps=int(1000000))
        model.save("dqn_jihuang")
        del model
    elif algo == "a2c":
        model = A2C("MlpPolicy1", env, action_mask=True, verbose=1, tensorboard_log='./a2c_run/')
        model.learn(total_timesteps=int(1000000))
        model.save("a2c_jihuang")
        del model
    else:
        raise NotImplementedError
    train_end = datetime.now()
    durn = (train_start - train_end).seconds / 1000
    m, s = divmod(int(durn), 60)
    h, m = divmod(m, 60)
    print("train complete, time cost: %02d:%02d:%02d" % (h, m, s))
