"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


# Base class to represent activations
class Activation:

    def __init__(self):
        pass

    def forward(self, ak):
        pass

    def gradient(self, ak):
        pass


class Sigmoid(Activation):

    def forward(self, ak):

        return 1/(1+np.exp(-ak))
    
    def gradient(self, ak):

        y = self.forward(ak)
        return y(1-y) 
    

class Linear(Activation):

    def forward(self, ak):
        return ak
    
    def gradient(self, ak):
        return np.ones(shape=ak.shape)
    