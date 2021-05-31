import numpy as np

#we want to learn the price, we imagine each of price is bandit arm. and each price has certain conversion rate.
#in this case we have 4 arm. we can only observe the conversion rate for that specific arm that we are trying. 
#this is called limited feedback problem, because we dont not observe what user done for another prices.
#we just observe the user with specific prices.
#each arm will have unknown mean for the reward for each arm. in price problem, mean is would be the mean of conversion rate (the probability of the user that buy the product at that price)
#we model the conversion rate as bernoulli probabilities 

#why we should divide environtment and learner? to have possibilities to make plug-unplug the learner from simulation to the real environtment

class Environment():
    def __init__(self, n_arms, probabilities): #in the environment object has characterized by 2 variables n arm and prob
        self.n_arms = n_arms #pass the input with 2 parameters and initialize the parameter
        self.probabilities = probabilities # in this case we use bernouli dist, so describe 1 value of each arm

    def round(self, pulled_arm): #model the interaction between learner and environment -> this function use the chosen arm (pulled arm) as input
        reward = np.random.binomial(1, self.probabilities[pulled_arm]) # 1 mean bernauli dist and the succes prob related to the super arm that specified in the constructor
                                    #number of trial and prob of success
        return reward
