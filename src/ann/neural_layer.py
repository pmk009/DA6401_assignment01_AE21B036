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
        self.ak = np.zeros(shape=(nk,))

        self.grad_W = np.zeros(shape=(nk_1,nk))
        self.grad_b = np.zeros(shape=(nk,))

        if initialization not in ['random','xavier']:
            raise ValueError('initialization should be random or xavier')
        
        elif initialization == 'random':

            self.W = np.random.random(size=(nk_1,nk))
            self.b = np.random.random(size=(nk,))
        else:
            pass

    def forward(self, hk_1):

        self.hk_1 = hk_1

        np.matmul(self.W.T,hk_1,out=self.ak)
        ak+= self.b

        self.hk = self.activation.forward(ak)
    

    def backward(self, del_k):

        np.multiply(del_k, self.activation.gradient(self.ak), out=self.grad_b)
        np.outer(self.hk_1, self.grad_b, out=self.grad_W)