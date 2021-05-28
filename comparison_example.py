import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment
from TS_Learner import TS_Learner
from Greedy_Learner import Greedy_Learner


if __name__ == '__main__':
    p = np.array([0.4, 0.3, 0.1, 0.35]) #bernauli distribution for the reward function
    n_arms = len(p)
    opt = np.max(p)

    T = 300

    n_experiments = 100
    ts_rewards_per_experiment = []
    ts_rewards_per_arm = []
    gr_rewards_per_experiment = []

    for e in range(0, n_experiments):
        env = Environment(n_arms=n_arms, probabilities=p)
        ts_learner = TS_Learner(n_arms=n_arms)
        gr_learner = Greedy_Learner(n_arms=n_arms)
        for t in range(T): #interaction between the environment and learner
            # Thompson Sampling Learner
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            # Greedy Learner
            pulled_arm = gr_learner.pull_arm()
            reward = env.round(pulled_arm)
            gr_learner.update(pulled_arm, reward)

        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        ts_rewards_per_arm.append(ts_learner.rewards_per_arm)
        gr_rewards_per_experiment.append(gr_learner.collected_rewards)
    

    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
    plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
    plt.legend(["TS", "Greedy"])
    plt.show()
