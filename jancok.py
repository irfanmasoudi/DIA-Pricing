import numpy as np
from Learner import *
 

 class Greedy_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_payoffs = np.zeros(n_arms)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.expected_payoffs[pulled_arm] = (self.expected_payoffs[pulled_arm]*(self.t - 1.0) + reward)/self.t