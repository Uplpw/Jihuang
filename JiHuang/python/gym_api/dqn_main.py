import sys
import torch

# load jihuang path
sys.path.insert(-1, "JiHuang/python")

import gym, Jihuang
import numpy as np
from DQN.dqn import DQN
from stable_baselines3.common.env_util import make_vec_env

# create a jhuang gym env by jihuang name
env = make_vec_env("jihuang-simple-v0", n_envs=1)

# use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create ppo model
model = DQN("MlpPolicy", env, learning_rate=0.0003, batch_size=64, device=device, verbose=1, tensorboard_log='./run/')

# start train and learn
model.learn(total_timesteps=int(10000000))

# save model
model.save("dqn_jihuang")

del model  # remove to demonstrate saving and loading

model = DQN.load("dqn_jihuang", device=device)

# test
reward_list = []
step_list = []
sum_reward = 0
for eposide in range(50):
    obs = env.reset()
    dones = False
    step = 0
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        sum_reward = sum_reward + rewards
        step += 1
    reward_list.append(sum_reward[0])
    step_list.append(step)
    sum_reward = 0
print("reward_list", reward_list)
print("mean:", np.mean(reward_list), "max:", max(reward_list), "min:", min(reward_list))
print("step_list", step_list)
print("mean:", np.mean(step_list), "max:", max(step_list), "min:", min(step_list))
