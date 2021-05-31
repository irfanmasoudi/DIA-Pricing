import numpy as np


class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms #number of arms
        self.t = 0 # current round value 
        self.rewards_per_arm = [[] for i in range(n_arms)] #reward per arm
        self.collected_rewards = np.array([])

    def pull_arm(self):
        pass

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward) #update the list reward per arm and collect the reward 
        self.collected_rewards = np.append(self.collected_rewards, reward)
