## Jihuang

### 1 Trick Improve

#### 1.1 obs

- obs_scale：将4219维空间进行缩放，只保留部分信息。

  - block1：基本数据（编号、hp、饱食度、饥渴度、体温、x、y、动作结果、时间、昼夜、季节、格子地貌、格子天气）= 13

  - block2：背包数据（id、耐久度）（24*2=48）

  - block3：装备数据（id、耐久度）（1*2=2）

  - block4：buff数据（无）

    - 格式：buff_id 饱食度 饥渴度 血量 温度 攻击力 防御力 移动速度 视野

  - block5：视野分类1数据（pig、river）（3+2=5）

    - 内容：agent animal plant resource
    - 格式：[类型, 编号, x坐标, y坐标, hp, 饱食度, 饥渴度] 、[类型, x坐标, y坐标, hp, 0, 0, 0] 、[类型, x坐标, y坐标, hp, 0, 0, 0]、[类型, x坐标, y坐标, 0, 0, 0]

  - block6：视野分类2数据（water、meat、torch）（3*3=9）

    - 格式：[类型, x坐标, y坐标]

  - block7：地貌数据（无）

    - 格式：[x坐标, y坐标, 天气，地貌]

    

  - 总计大小：sum = 13 + 48 + 2 + 0 + 5 + 9 + 0 = 77

  

- obs_mask：将4219维空间的部分信息进行mask，即赋值为0，但是大小仍为4219




- 借鉴open ai five 将对应的值进行embedding*



#### 1.2 action

- base action 类型（9种。已实现7种）

  - Idle：不动，后面两个参数无效*
  - Attack，攻击，后面两个参数是坐标*
  - Collect，收集，后面两个参数是坐标*
  - Pickup，拾取，后面两个参数是类别，数量（默认1）*
  - Consume，消耗，后面两个参数是类别，数量（默认1）*
  - EquipAction，装备，后面两个参数是类别，穿脱(0穿1脱)*
  - Synthesis，合成，后面两个参数是类别，数量（默认1）
  - Discard，丢弃，后面两个参数是类别，数量（默认1）
  - Move，移动，后面两个参数是偏移量*

  

- action space

  - 最初设计形式

    - 多维离散：基础动作以及相应参数都由策略网络给出

    - 形式：[action_id, para1, para2]
    - 不足：本身obs已经不好训练，参数也有网络训练得到是难上加难

  - 现在设计形式

    - 一维离散：基础动作由策略网络给出，动作参数由 $para1, para2=select\_target()$ 函数计算得到
    - 形式：[action_id]
    - 设计原因
      - 一是借鉴相同类型环境针对参数动作空间的处理办法
      - 二是jihuang环境的参数空间跳跃比较大（如：1~90007，中间有很多值没有用到），除非人工设计函数（挺麻烦的）将这些值集中化处理
      - 三是隐藏作用：也降低了action的空间大小
      - 四是未来选项：针对不同的动作，利用相关的obs部分block设计网络来输出对应的参数，向user提供实现的接口（base简单点，难点留给user）

    

- action mask
  - 原因：本身action空间比较大，为了降低一些无效动作，加入先验部分，使得action大部分甚至是全部都是有效/有一定作用
  - 实例
    - 视野内没有猪的时候是一般不会进行attack
    - 没有kill pig前一般不会进行pickup meat
    - 等等



- 混合动作设计*
  - 在baseline任务基本完成后实现，并为user提供实现的接口。
  - 问题
    - 目前jihuang环境无法应用到像ddpg、td3等算法上，因为环境的动作并不是真正意义上的连续，需要一个函数进行转换
  - 概述
    - 实际任务中，混合动作的需求经常出现：如王者荣耀游戏既需要离散动作（选择技能），又需要连续动作（移动角色）
  - 解决办法
    - 强行使用DQN类算法，把连续动作分成多个离散动作
    - [SAC for Discrete Action Space](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1910.07207)，把输出的连续动作当成是离散动作的执行概率
    - [P-DQN](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1810.06394)（Parameterized DQN），把DQN和DDPG合起来
    - [H-PPO](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1903.01344) （Hybrid PPO），同时让策略网络输出混合动作



#### 1.3 奖励设计

- Move Reward
  - 与猪的距离减少、视野内出现pig则加1
- Attack Reward
  - 成功攻击加5
  - kill pig 加50
- Consume Reward
  - 恢复饱食度/饥渴度的值即为Reward（上限50）
  - 当恢复值没有超过40时，其Reward只有恢复值的0.5倍
- Collect Reward
  - 成功收集加2
- Pickup Reward
  - 成功拾取meat、water加10
  - 成功拾取torch加5
- Equip Reward
  - 成功装备加5
- 动作失败均无奖励，即加0



- 计划：添加负奖励来均衡动作的选择以及结果的稳定性



#### 1.4 日志设计

- 为了通过结果优化奖励函数、action mask函数、select_target函数等的设计，实现了结果格式输出

```python
# result of each episode

{'type': 'Result', 'step': 474, 'reward': 4137.0}
{'type': 'Idle', 'count': 24}
{'type': 'Attack', 'count': 72, 'reward': 1935.0, 'attack_count': 37, 'attack_reward': 185.0, 'kill_count': 35, 'kill_reward': 1750.0, 'fail_count': 0}
{'type': 'Collect', 'count': 110, 'reward': 220.0, 'water_count': 110, 'water_reward': 220.0, 'fail_count': 0}
{'type': 'Pickup', 'count': 102, 'reward': 190.0, 'meat_count': 25, 'meat_reward': 100.0, 'water_count': 33, 'water_reward': 80.0, 'torch_count': 1, 'torch_reward': 10.0, 'fail_count': 43}
{'type': 'Consume', 'count': 34, 'reward': 1700.0, 'meat_count': 17, 'meat_reward': 850.0, 'water_count': 17, 'water_reward': 850.0, 'fail_count': 0}
{'type': 'Equip', 'count': 1, 'reward': 10.0, 'torch_count': 1, 'torch_reward': 10.0, 'fail_count': 0.0}
{'type': 'Move', 'count': 131, 'reward': 82.0, 'success_count': 82, 'success_reward': 82.0, 'fail_count': 49}
```



### 2 Summary

#### 2.1 环境配置——简化

- config_simple.prototxt

  - agent
    -  饱食度或饥渴度（初始/上限为100）为0结束一局游戏，每个时间步消耗2
    - 攻击距离4.1、攻击伤害60hp、拾取/收集距离10.1、视野大小17*17（黑夜为0，火把可以恢复视野）
    - 消耗meat恢复饱食度50、消耗water恢复饥渴度50
  - 地图
    - 地图大小：40*40，没有wolf
    - 一天100个时间步、白天黑夜7:3、一月30天、一年12月
  - 其他全是默认值

- example_simple.prototxt

  - agent（1个）、pig（30个，hp=100）、river（10个）、torch（20个）
  - 其他无

  

#### 2.2 算法实现

- PPO

  - 原始
  - trick
- DQN
  - 原始
  - trick

- A2C

  - 原始
  - trick

  

#### 2.3 Result
