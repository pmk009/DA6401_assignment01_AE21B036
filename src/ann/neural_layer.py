"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from activations import *


class neural_layer:

    def __init__(self, nk: int=128, nk_1: int=128, activation: Activation=Sigmoid, initialization: str='random'):

        self.num_neurons = nk
        self.prev_num_neurons = nk_1 
        self.activation = activation()

        self.hk = None
        self.hk_1 = None
        self.ak = None

        self.grad_W = None
        self.grad_b = None

        if initialization not in ['random','xavier']:
            raise ValueError('initialization should be random or xavier')
        
        elif initialization == 'random':

            self.W = np.random.randn(nk_1,nk)*0.01
            self.b = np.random.randn(nk)*0.01
        else:
            pass

    def forward(self, hk_1):

        self.hk_1 = hk_1

        self.ak = np.matmul(hk_1, self.W)
        self.ak+= self.b

        self.hk = self.activation.forward(self.ak)
    

    def backward(self, del_k):

        delta = del_k * self.activation.gradient(self.ak)

        self.grad_W = self.hk_1[:, :, None] * delta[:, None, :]
        self.grad_b = delta

        return delta @ self.W.T