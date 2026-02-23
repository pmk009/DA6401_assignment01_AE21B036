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

        self.grad_W = np.zeros((self.prev_num_neurons,self.num_neurons))
        self.grad_b = np.zeros((self.num_neurons,))

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
        
        np.matmul(self.hk_1.T,delta, out = self.grad_W)
        self.grad_W /= self.hk_1.shape[0] 

        np.mean(delta, axis=0, out= self.grad_b)
        
        return delta @ self.W.T