import sys

sys.path.insert(-1, "JiHuang/python")
from datetime import datetime
import gym, Jihuang
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from OC.oc_utils import SoftmaxPolicy, SigmoidTermination, EpsGreedyPolicy, Critic
from OC.config import discount, lr_term, lr_intra, lr_critic, epsilon, nruns, temperature, nepisodes, nsteps, noptions
from OC.utils import state2int

# Random number generator for reproducability
rng = np.random.RandomState(1234)
history = np.zeros((nruns, nepisodes, 2))
option_terminations_list = []

current_time = datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
log_path = "./oc_{}".format(current_time)
writer = SummaryWriter(log_path)
total_timesteps = 0

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
        state_pro = state2int(state)
        # print("nruns:", run, "episode:", episode, "state:", np.array(state).astype(dtype=int), )
        option = policy_over_options.sample(np.array(state_pro).astype(dtype=int))
        action = option_policies[option].sample(state_pro)

        critic.cache(state_pro, option, action)

        duration = 1
        option_switches = 0
        avg_duration = 0.0

        for step in range(nsteps):
            state, reward, done, _ = env.step(action)
            state_pro = state2int(state)

            if option_terminations[option].sample(state_pro):
                option = policy_over_options.sample(state_pro)
                option_switches += 1
                avg_duration += (1.0 / option_switches) * (duration - avg_duration)
                duration = 1

            action = option_policies[option].sample(state_pro)

            # Critic update
            critic.update_Qs(state_pro, option, action, reward, done, option_terminations)

            # Intra-option policy update with baseline
            Q_U = critic.Q_U(state_pro, option, action)
            Q_U = Q_U - critic.Q_Omega(state_pro, option)
            option_policies[option].update(state_pro, action, Q_U)

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
