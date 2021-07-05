import sys

sys.path.insert(-1, "JiHuang/python")
import gym, Jihuang
import torch
import argparse
from train import train
from test import test
from Jihuang.wrapper import ScaledObservation, Reward, SelectTarget

parser = argparse.ArgumentParser(description='rl algo in jihuang environment')
parser.add_argument('--env', type=str, default='jihuang-simple-v0', help='select an algorithm among ppo, dqn, a2c')
parser.add_argument('--algo', type=str, default='ppo', help='select an algorithm among ppo, dqn, a2c')
parser.add_argument('--mode', type=str, default='train', help='choose training or testing')
parser.add_argument('--is_mask', type=bool, default=True, help='whether use action mask or not')
parser.add_argument('--timesteps', type=int, default=1000, help='train timesteps')

args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # is_mask = args.is_mask
    # timesteps = args.timesteps
    is_mask = True
    timesteps = int(1000)

    if args.mode == "train":
        env = gym.make(args.env)
        env = ScaledObservation(SelectTarget(Reward(env)))
        train(env, args.algo, timesteps, is_mask, device)
    elif args.mode == "test":
        env = gym.make(args.env)
        env = ScaledObservation(SelectTarget(Reward(env)))
        test(env, args.algo, device)
    else:
        raise NotImplementedError
