## 1 抓猪

### 1.1 环境配置——简化

- config_simple.prototxt

  - agent： hp不变，通过饱食度（初始为500/100）为0结束一局游戏
  - 地图大小：40*40，没有wolf

- example_simple.prototxt

  - agent, （10, 10）
  - 5 pigs, hp=200/1000, position：（12，10）/（28，30）

  

### 1.2 obs处理

- 除了前面13项（编号、hp、饱食度、饥渴度、体温、x、y、动作结果、时间、昼夜、季节、格子地貌、格子天气）以及视野环境block（agent animal plant resource），其他全部mask



### 1.3 action space

- 只有三个动作，其格式为（action，x，y）
- 不动、攻击（自动攻击）、移动



### 1.4 reward 设计

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

### 1.5 results

reward_list [array([39.40765], dtype=float32), array([94.95648], dtype=float32), array([1184.9833], dtype=float32), array([156.37396], dtype=float32), array([13.478944], dtype=float32), array([105.71006], dtype=float32), array([1162.1512], dtype=float32), array([125.21085], dtype=float32), array([11.241217], dtype=float32), array([1162.939], dtype=float32)]

mean: 405.64526