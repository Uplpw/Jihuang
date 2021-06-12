import sys

sys.path.insert(-1, "JiHuang/python")

import gym, Jihuang
import numpy as np
from utils import SoftmaxPolicy, SigmoidTermination, EpsGreedyPolicy, Critic
from config import discount, lr_term, lr_intra, lr_critic, epsilon, nruns, temperature, nepisodes, nsteps, noptions

# Random number generator for reproducability
rng = np.random.RandomState(1234)

option_terminations_list = []

for run in range(nruns):

    env = gym.make("jihuang-simple-v0")

    nstates = env.observation_space.shape[0]
    nactions = env.action_space.shape[0]

    option_policies = [SoftmaxPolicy(rng, lr_intra, nstates, nactions, temperature) for _ in range(noptions)]
    option_terminations = [SigmoidTermination(rng, lr_term, nstates) for _ in range(noptions)]
    policy_over_options = EpsGreedyPolicy(rng, nstates, noptions, epsilon)
    critic = Critic(lr_critic, discount, policy_over_options.Q_Omega_table, nstates, noptions, nactions)

    for episode in range(nepisodes):
        state = env.reset()
        option = policy_over_options.sample(state)
        action = option_policies[option].sample(state)

        critic.cache(state, option, action)

        for step in range(nsteps):
            state, reward, done, _ = env.step(action)

            if option_terminations[option].sample(state):
                option = policy_over_options.sample(state)
                # option_switches += 1
                # avg_duration += (1.0 / option_switches) * (duration - avg_duration)
                duration = 1

            action = option_policies[option].sample(state)

            # Critic update
            critic.update_Qs(state, option, action, reward, done, option_terminations)

            # Intra-option policy update with baseline
            Q_U = critic.Q_U(state, option, action)
            Q_U = Q_U - critic.Q_Omega(state, option)
            option_policies[option].update(state, action, Q_U)

            # Termination condition update
            option_terminations[option].update(state, critic.A_Omega(state, option))

            if done:
                break
    option_terminations_list.append(option_terminations)