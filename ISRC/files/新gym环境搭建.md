# gym API新环境创建手册

## 1. gym通用API与简单使用说明

利用gym的api，我们在一个C++的Jihuang基础上可以衍生若干个自定义环境，他们拥有不同的配置文件和reward函数。要创建这样的自定义环境，需要稍微了解gym的api。

### register & gym.make

gym.make(env_name) -> gym.Env是我们实例化一个自定义环境的方式。要允许使用gym.make来实例化自定义的环境，我们需要register这个环境，例如我们定义了一个叫JihuangSimple的自定义环境，需要这样做register的操作：

```python
# package name: Jihuang
import gym
from gym.envs.registration import register

register(
    id='jihuang-simple-v0',
    entry_point='Jihuang:JihuangSimple',  # entry_point: packagename:classname
)
class JihuangSimple(gym.Env):
    def __init__(self, ...):
        ...
    ...
```

在这之后，我们就可以使用

```python
e = gym.make('jihuang-simple-v0')  # use the id to create an environment
```

来创建一个我们自己的Jihuang环境。

下面来看这个JihuangSimple，也就是自定义环境怎么写。

### init

1.  (recommend)在\_\_init\_\_函数中应当定义action_space和observation_space，以保证多数开源代码能够读取这一信息。
2.  (recommend)后续计算reward可能会需要前一次的observation和这一次的observation，可以考虑在init的时候初始化
3.  (necessary)需要连接C++后端的接口。

```python
import Jihuang._jihuang as game  # This is the c++ backend

def __init__(self, ...):
    self.action_space = gym.spaces.xxx
    self.observation_space = gym.spaces.xxx
    
    self._prev_obs = xxx
    self.obs = xxx
    
    self.env = game.Env(env_param, env_config, log_dir, log_name, log_level)
```

### step

def step(action) -> obs, reward, done, info

step函数接受传入的action(格式可以自己定义，但通常情况下，开源算法都会根据action_space给出)，传出的分别是observation reward done info. observation通常需要和observation_space对应，因此我们知ihuang中，我们的c++后端给出了step的接口，但这个接口只接受action信息，不给出返回值，observation由get_agent_observe单独给出，一般需要处理一下再给强化学习端发送。done决定该环境是否已经不能运行，一般我们是判定agent有没有死掉。info可以自行返回一些需要的信息用于debug之类的用途。

```python
def step(self, action):
    self.action = _pre_process(action)
    # C++ take action
    self.env.step(self.action)
    # store the old observation and get new observation
    self._prev_obs = self.obs
    self.obs = self.env.get_agent_observe()
    # usually we do not return the full observation for reinforcement learning
    obs_return = _process_obs(self.obs)
    # calculate rewards
    reward = _calc_reward(self._prev_obs, self.obs, action)
    # done?
    done = _check_done(self.obs)
    # info returns whatever you want
    info = {}
    
    return obs_return, reward, done, info
```

### reset

def reset() ->obs, reward, done, info

reset函数会重置环境，并给出初次的observation等信息。这里我们调用后端的reset接口重置即可。

```python
def reset(self):
    self.env.reset()
    
    self.obs = self.env.get_agent_observe()
    obs_return = _process_obs(self.obs)
    reward = 0
    done = False
    info = {}
    
    return obs_return, reward, done, info
```



## 2. 自定义环境注意事项

-   observation和action都是二维的，有一个维度表示agent，在单agent的条件下这个维度只会是1。
-   C++后端只接受list类型，数据为float的输入。

