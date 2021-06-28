# Quick Start

## 1 Deploy Environment  

The following is based on the Jihuang's gym API

### 1.1 download gym environment 

```shell
git clone -b 4-gym http://62.234.201.16/RZChen/JiHuang.git
```

### 1.2 configure environment  path

Open the ***.bashrc*** file and add the environment variable of the system. The name of the environment variable cannot be modified, and the path can be modified according to personal configuration

```shell
# gcc的路径
export JIHUANG_CC=/root/install/gcc-7.4.0/bin/gcc
# g++的路径
export JIHUANG_CXX=/root/install/gcc-7.4.0/bin/g++

# gflag的文件夹路径
export JIHUANG_INCLUDE_GFLAG=/usr/include/gflags/
# glog的文件夹路径
export JIHUANG_INCLUDE_GLOG=/usr/include/glog/
# protobuf的文件夹路径
export JIHUANG_INCLUDE_PROTOBUF=/share/install/protobuf-3.11.2/include/google/protobuf

# gflag的库文件
export JIHUANG_LD_LIB_GFLAG=/usr/lib/x86_64-linux-gnu/libgflags.so
# glog的库文件
export JIHUANG_LD_LIB_GLOG=/usr/lib/x86_64-linux-gnu/libglog.so
# protobuf的库文件
export JIHUANG_LD_LIB_PROTOBUF=/share/install/protobuf-3.11.2/lib/libprotobuf.so

# python文件
export JIHUANG_PYTHON_INCLUDE_DIRS=/share/miniconda3/envs/jihuang/include/python3.6m
# python的include文件夹
export JIHUANG_BOOST_INCLUDE_DIRS=/share/miniconda3/envs/jihuang/include/

# python库文件
export JIHUANG_PYTHON_LD_LIB_DIRS=/share/miniconda3/envs/jihuang/lib/libpython3.6m.so
# boost的库文件
export JIHUANG_BOOST_LD_LIB_DIRS=/share/miniconda3/envs/jihuang/lib/libboost_python3.so
```

### 1.3 compiling environment

```shell
# enter jihuang dir
cd JiHuang

# Create a build folder
mkdir build && cd build

# start compliling
cmake .. && make -j

# after successful compilation, the executable file will be generated in the current directory: jihuang_main

# test()
./jihuang_main
```



## 2 Run RL Algorithm

After the environment is successfully deployed, an example is given to show how to run reinforcement learning algorithm in our jihuang environment

### 2.1 install package

The following libraries need to be installed to run the ppo example:

- gym
- stable_baselines3



### 2.2 run ppo example

In the current deployed environment, there is an example of PPO algorithm to run

```shell
# enter jihuang dir 
cd JiHuang

# run ppo.py file
python python/gym_api/ppo.py
```



The ppo.py file is as follows:

```python
import sys
import torch

# load jihuang path
sys.path.insert(-1, "JiHuang/python")

import gym, Jihuang
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# create a jhuang gym env by jihuang name
env = make_vec_env("jihuang-simple-v0", n_envs=1)

# use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create ppo model
model = PPO("MlpPolicy", env, learning_rate=0.0003, batch_size=64, device=device, verbose=1)

# start train and learn
model.learn(total_timesteps=int(1000000))

# save model
model.save("ppo_jihuang")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_jihuang", device=device)

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

```

