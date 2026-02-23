"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np
import argparse
class optimizer:

    def update(self):
        pass



class SGD(optimizer):

    def __init__(self,cli_args: argparse.Namespace, Layers: list):

        self.lr = cli_args.lr
        self.weight_decay = cli_args.wd
        self.Layers = Layers
        self.num_layers = len(Layers)

    def update(self):

        for k in range(self.num_layers):
            
            L = self.Layers[k]
            L.W -= self.lr*(L.grad_W + self.weight_decay*L.W)
            L.b -= self.lr*(L.grad_b + self.weight_decay*L.b)
            