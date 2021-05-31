from Learner import Learner
import numpy as np

#if there is better choice then directly put as the choosen one. which is not consider another better possibilitis

class Greedy_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_payoffs = np.zeros(n_arms)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.expected_payoffs[pulled_arm] = (self.expected_payoffs[pulled_arm]*(self.t - 1.0) + reward)/self.t

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        idxs = np.argwhere(self.expected_payoffs == self.expected_payoffs.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm
