import sys

# sys.path.append("..")
sys.path.insert(-1, "JiHuang/python")

import gym, Jihuang
import torch
import argparse
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from train import train
from test import test
from Jihuang.wrapper import ScaledObservation, Reward, SelectTarget

parser = argparse.ArgumentParser(description='rl algo in jihuang environment')
parser.add_argument('--env', type=str, default='jihuang-simple-v0', help='select an algorithm among ppo, dqn, a2c')
parser.add_argument('--algo', type=str, default='ppo', help='select an algorithm among ppo, dqn, a2c')
parser.add_argument('--mode', type=str, default='train', help='choose training or testing')


args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        env = gym.make(args.env)
        env = ScaledObservation(SelectTarget(Reward(env)))
        # env = make_vec_env(env_id=env, n_envs=1)
        train(env, args.algo, device)
    elif args.mode == "test":
        env = gym.make(args.env)
        env = ScaledObservation(SelectTarget(Reward(env)))
        test(env, args.algo, device)
    else:
        raise NotImplementedError
