import sys

sys.path.insert(-1, "JiHuang/python")
from datetime import datetime
import gym, Jihuang
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
from OC_1.oc_utils import SoftmaxPolicy, SigmoidTermination, EpsGreedyPolicy, Critic
from OC_1.config import discount, lr_term, lr_intra, lr_critic, epsilon, nruns, temperature, nepisodes, nsteps, noptions
from OC_1.utils import state2int
from Jihuang.obs_utils import JihuangObsProcess

# Random number generator for reproducability
rng = np.random.RandomState(1234)
history = np.zeros((nruns, nepisodes, 2))
option_terminations_list = []

current_time = datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
log_path = "./my_oc/oc_{}".format(current_time)
writer = SummaryWriter(log_path)
total_timesteps = 0
obs_utils = JihuangObsProcess()

for run in range(nruns):

    env = gym.make("jihuang-simple-v0")

    # nstates = env.observation_space.shape[0]
    nstates = 512

    nactions = env.action_space.n

    print(nstates, nactions)

    option_policies = [SoftmaxPolicy(rng, lr_intra, nstates, nactions, temperature) for _ in range(noptions)]
    option_terminations = [SigmoidTermination(rng, lr_term, nstates) for _ in range(noptions)]
    policy_over_options = EpsGreedyPolicy(rng, nstates, noptions, epsilon)
    critic = Critic(lr_critic, discount, policy_over_options.Q_Omega_table, nstates, noptions, nactions)

    for episode in range(nepisodes):
        state = env.reset()
        state_copy = state.copy()
        state_copy_tensor = th.tensor(state_copy).reshape(-1, 77)
        state_pro = state2int(state)

        action_mask = obs_utils.action_mask_from_obs(state_copy_tensor)
        action_mask_cpu = th.BoolTensor(th.tensor(action_mask).bool())
        # print("nruns:", run, "episode:", episode, "state:", np.array(state).astype(dtype=int), )
        option = policy_over_options.sample(np.array(state_pro).astype(dtype=int))
        action = option_policies[option].sample(state_pro, action_mask, action_mask_cpu)

        critic.cache(state_pro, option, action)

        duration = 1
        option_switches = 0
        avg_duration = 0.0

        for step in range(nsteps):
            state, reward, done, _ = env.step(action)
            state_copy1 = state.copy()
            state_copy_tensor1 = th.tensor(state_copy1).reshape(-1, 77)
            action_mask1 = obs_utils.action_mask_from_obs(state_copy_tensor1)
            action_mask_cpu1 = th.BoolTensor(th.tensor(action_mask1).bool())
            state_pro = state2int(state)

            if option_terminations[option].sample(state_pro):
                option = policy_over_options.sample(state_pro)
                option_switches += 1
                avg_duration += (1.0 / option_switches) * (duration - avg_duration)
                duration = 1

            action = option_policies[option].sample(state_pro, action_mask1, action_mask_cpu1)

            # Critic update
            critic.update_Qs(state_pro, option, action, reward, done, option_terminations)

            # Intra-option policy update with baseline
            Q_U = critic.Q_U(state_pro, option, action)
            Q_U = Q_U - critic.Q_Omega(state_pro, option)
            option_policies[option].update(state_pro, action, Q_U, action_mask1, action_mask_cpu1)

            # Termination condition update
            option_terminations[option].update(state_pro, critic.A_Omega(state_pro, option))

            if done:
                break
        history[run, episode, 0] = step + 1
        history[run, episode, 1] = avg_duration
        total_timesteps += history[run, episode, 0]
        writer.add_scalar("episode_len", history[run, episode, 0], total_timesteps)

    option_terminations_list.append(option_terminations)
    print("episode:", history[run, :, 0])

# for i in range(nruns):
#     f.write(str(i) + "\n")
#     temp = history[i, :, 0]
#     for j in temp:
#         f.write(str(j) + " ")
#     print("\n")
#
# f.close()
