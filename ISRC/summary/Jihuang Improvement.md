## Jihuang

### 1 Trick Improve

#### 1.1 obs

- 将4219维空间缩放到极小的空间

- 只提取关键的信息




- 借鉴open ai five 将对应的值进行embedding*



#### 1.2 action

- action mask
- mask一些不合理、不合法以及注定会失败的动作



- 混合动作设计*



#### 1.3 action space

- 一维离散
- 动作参数由函数生成



- 其参数由对应设计的网络来输出，其输入为action涉及的相应的状态块（简单一些便于训练，难点留给他人）*



#### 1.4 奖励设计

- 负奖励来均衡动作的选择以及结果的稳定性



### 2 Summary

#### 2.1 环境配置——简化

- config_simple.prototxt

  - agent： hp不变，通过饱食度（初始为500/100）为0结束一局游戏
  - 地图大小：40*40，没有wolf

- example_simple.prototxt

  - agent, （10, 10）
  - 5 pigs, hp=200/1000, position：（12，10）/（28，30）

  

#### 2.2 obs处理

- 除了前面13项（编号、hp、饱食度、饥渴度、体温、x、y、动作结果、时间、昼夜、季节、格子地貌、格子天气）以及视野环境block（agent animal plant resource），其他全部mask



#### 2.3 action space

- 只有三个动作，其格式为（action，x，y）
- 不动、攻击（自动攻击）、移动



#### 2.4 reward 设计

- move reward （move action）

  - 视野内没有猪为0
  - 视野内有猪：1.0 / pig_distance——pig_distance（与猪的距离）

  

- 攻击 reward （attack action）

  - 视野内没有猪：攻击无效，奖励为0
  - 视野内有猪
    - 与猪的距离在攻击范围内：攻击一次 reward+10，猪死亡 reward+1000（没有装备情况下攻击一次60 hp）
    - 与猪的距离不在攻击范围内：reward + 1.0 / attack_pig_distance——（与猪的距离）

- find pig reward

  - 视野内有猪 + 0.01
  
  
  
- move

  - reward
  - move_flag----------------1

  - result_type: success or fail (待完善)

  attack

  - reward
  - attack and not kill
  - attack and kill

  更新：

  - reward
  - attack_type（attack or kill）----------------2

  - result_type: success or fail (待完善)

   collect

  - reward
  - collect_type----------------1

  - result_type: success or fail

   consume

  - reward
  - consume_type----------------2

  - result_type: success or fail (待完善)

   pickup

  - reward
  - pickup_type----------------3

  - result_type: success or fail (待完善)

  equip

  - reward
  - equip_type----------------1

  - result_type: success or fail (待完善)

#### 2.5 results

reward_list [array([39.40765], dtype=float32), array([94.95648], dtype=float32), array([1184.9833], dtype=float32), array([156.37396], dtype=float32), array([13.478944], dtype=float32), array([105.71006], dtype=float32), array([1162.1512], dtype=float32), array([125.21085], dtype=float32), array([11.241217], dtype=float32), array([1162.939], dtype=float32)]

mean: 405.64526

