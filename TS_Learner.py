from Learner import Learner
import numpy as np


class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms) #constructor that passing the parameters
        self.beta_parameters = np.ones((n_arms, 2)) #store the parameter value of beta distribution that we call beta parameters 

    def pull_arm(self): #select which arm to pull in each round of t, 
                        #TS algo select the arm to pull by sampling the value for each arm from beta dist and then select the arm 
                        #-associated to the beta dist that generated with the maximum value.
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])) #select the index of the maximum value 
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward
