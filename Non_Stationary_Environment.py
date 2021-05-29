from Environment import Environment
import numpy as np

#if we have really small change distribution, there is no really different.
#in Non stationary, we have phases and each phase, the parameter of each time changes in time.

class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probabilities, horizon): #each phase we will have different probability
        super().__init__(n_arms, probabilities)
        self.t = 0
        n_phases = len(self.probabilities)
        self.phase_size = horizon/n_phases #lenght of each phases

    def round(self, pulled_arm):
        current_phase = int(self.t / self.phase_size)
        p = self.probabilities[current_phase][pulled_arm] #probability of pulled arm will be the probability of row phases and column of thr pulled arm
        self.t += 1
        reward = np.random.binomial(1, p)
        return reward
