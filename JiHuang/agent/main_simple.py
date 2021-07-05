import sys

sys.path.insert(-1, "JiHuang/python")
import gym, Jihuang
import torch
import argparse
from stable_baselines3.common.env_util import make_vec_env
from utils.trick import select_target, action_mask_from_obs, _obs_scale
from utils.reward import reward_func
from train import train
from test import test

parser = argparse.ArgumentParser(description='rl algo in jihuang environment')
parser.add_argument('--env', type=str, default='jihuang-simple-v0', help='select an algorithm among ppo, dqn, a2c')
parser.add_argument('--algo', type=str, default='ppo', help='select an algorithm among ppo, dqn, a2c')
parser.add_argument('--mode', type=str, default='train', help='choose training or testing')

args = parser.parse_args()

if __name__ == "__main__":
    train_kwargs = {'select_target_func': select_target, 'reward_func': reward_func}
    # train_kwargs = {'select_target_func': select_target}
    test_kwargs = {'select_target_func': select_target, 'reward_func': reward_func}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        # env = make_vec_env("jihuang-simple-v0", n_envs=1, env_kwargs=train_kwargs)
        env = make_vec_env(args.env, n_envs=1, env_kwargs=train_kwargs)
        # env = gym.make("jihuang-simple-v0", **kwargs)
        train(env, args.algo, device)
    elif args.mode == "test":
        env = make_vec_env(args.env, n_envs=4, env_kwargs=test_kwargs)
        # env = gym.make("jihuang-simple-v0", **kwargs)
        test(env, args.algo, device)
    else:
        raise NotImplementedError
