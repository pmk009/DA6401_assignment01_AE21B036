"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

class objective_function:

    def __init__(self):
        pass

    def loss(self, y, y_hat):
        pass

    def gradient(self, y, y_hat):
        pass



class Cross_Entropy(objective_function):

    def loss(self, y, y_hat):

        return -np.sum( np.multiply(y, np.log(y_hat+1e-8)), axis=1)
    
    def gradient(self, y, y_hat):
        
        return (y_hat - y)
    
