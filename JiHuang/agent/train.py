from PPO.ppo import PPO
from DQN.dqn import DQN
from A2C.a2c import A2C


def train(env, algo, timesteps, is_mask, device):
    print("----------------------------- train start -----------------------------")
    if algo == "ppo":
        model = PPO(env, action_mask=is_mask, device=device, verbose=1, tensorboard_log="./agent/run/ppo/")
        model.learn(total_timesteps=timesteps)
        model.save("./agent/model/ppo_jihuang")
        del model
    elif algo == "dqn":
        model = DQN("MlpPolicy", env, action_mask=is_mask, verbose=1, tensorboard_log='./agent/run/dqn/')
        model.learn(total_timesteps=timesteps)
        model.save("./agent/model/dqn_jihuang")
        del model
    elif algo == "a2c":
        model = A2C("MlpPolicy1", env, action_mask=is_mask, verbose=1, tensorboard_log='./agent/run/a2c/')
        model.learn(total_timesteps=timesteps)
        model.save("./agent/model/a2c_jihuang")
        del model
    else:
        raise NotImplementedError
