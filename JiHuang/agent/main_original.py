import sys

sys.path.insert(-1, "JiHuang/python")
import gym, Jihuang
import torch
import argparse
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

parser = argparse.ArgumentParser(description='rl algo in jihuang environment')
parser.add_argument('--env', type=str, default='jihuang-original-v0', help='select an algorithm among ppo, dqn, a2c')
parser.add_argument('--algo', type=str, default='ppo', help='select an algorithm among ppo, dqn, a2c')
parser.add_argument('--mode', type=str, default='train', help='choose training or testing')

args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = make_vec_env(args.env, n_envs=4)
    if args.mode == "train":
        if args.algo == "ppo":
            model = PPO("MlpPolicy", env, verbose=1, device=device)
            model.learn(total_timesteps=25000)
            model.save("ppo_jihuang_original")
            del model
        elif args.algo == "a2c":
            model = A2C("MlpPolicy", env, verbose=1, device=device)
            model.learn(total_timesteps=25000)
            model.save("a2c_jihuang_original")
        else:
            print(args.algo, "not implement")
            raise NotImplementedError

    elif args.mode == "test":
        test_start = datetime.now()
        if args.algo == "ppo":
            model = PPO.load("ppo_jihuang_original", device=device)
        elif args.algo == "a2c":
            model = A2C.load("a2c_jihuang_original", device=device)
        else:
            print(args.algo, "not implement")
            raise NotImplementedError

        step_list = []
        for eposide in range(10):
            obs = env.reset()
            dones = False
            step = 0
            while not dones:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                step += 1
            step_list.append(step)
        test_end = datetime.now()
        durn = (test_start - test_end).seconds / 1000
        m, s = divmod(int(durn), 60)
        h, m = divmod(m, 60)
        print("test complete, time cost:%02d:%02d:%02d" % (h, m, s))
        print("mean:", np.mean(step_list), "max:", max(step_list), "min:", min(step_list))
        print("step:", step_list)
    else:
        print(args.mode, "is either train or test")
        raise NotImplementedError
