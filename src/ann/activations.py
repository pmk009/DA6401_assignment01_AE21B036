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
        return y*(1-y) 
    

class Linear(Activation):

    def forward(self, ak):
        return ak
    
    def gradient(self, ak):
        return np.ones(shape=ak.shape)

class Softmax(Activation):

    def forward(self, ak):

        x = ak - np.max(ak, axis = 1, keepdims=True)
        e_x = np.exp(x)
        e_x_sum = np.sum(e_x, axis=1, keepdims=True)
        
        return e_x / e_x_sum
    
    def gradient(self, ak):

        yi = self.forward(ak)
        
        return yi*(1-yi)
    

class ReLU(Activation):

    def forward(self, ak):
        
        return (ak>0)*ak
    
    def gradient(self, ak):
        
        return (ak>0.)*1.


class tanh(Activation):

    def forward(self, ak):

        return np.tanh(ak)

    def gradient(self, ak):

        return 1-np.square(self.forward(ak))