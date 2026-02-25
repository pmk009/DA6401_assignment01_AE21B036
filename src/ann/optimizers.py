"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np
import argparse
class optimizer:

    def update(self):
        pass

    def initialize_params(self):
        pass

    def log(self):
        return dict()

    def nesterov_update(self):
        pass

    def nesterov_revert(self):
        pass


class SGD(optimizer):

    def __init__(self,cli_args: argparse.Namespace, Layers: list):

        self.lr = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay
        self.Layers = Layers
        self.num_layers = len(Layers)

    def update(self):

        for k in range(self.num_layers):
            
            L = self.Layers[k]
            L.W -= self.lr*(L.grad_W + self.weight_decay*L.W)
            L.b -= self.lr*(L.grad_b + self.weight_decay*L.b)

    def log(self):
        
        return {'optimizer(sgd)/lr': self.lr, 'optimizer(sgd)/wd': self.weight_decay}


class Momentum(optimizer):

    def __init__(self, cli_args: argparse.Namespace, Layers: list, lr: float=1e-3, momentum: float=0.9):

        self.lr = lr
        self.beta = momentum
        self.weight_decay = cli_args.weight_decay

        self.Layers = Layers
        self.num_layers = len(Layers)

        self.initialize_params()

    def initialize_params(self):

        self.W_Momentums = []
        self.b_Momentums = []

        for L in self.Layers:
            self.W_Momentums.append(np.zeros(L.W.shape))
            self.b_Momentums.append(np.zeros(L.b.shape))
    
    def update(self):
        
    
        for i,L in enumerate(self.Layers):
            
            # Compute velocity update
            self.W_Momentums[i] *= self.beta
            self.b_Momentums[i] *= self.beta

            self.W_Momentums[i] -= self.lr*(L.grad_W)
            self.b_Momentums[i] -= self.lr*(L.grad_b)

            # Regularize
            L.W *= (1-self.lr*self.weight_decay)

            # Apply update
            L.W += self.W_Momentums[i]
            L.b += self.b_Momentums[i]
 
            
    
    def log(self):
        
        return {'optimizer/lr': self.lr, 'optimizer/wd': self.weight_decay, 'optimizer/beta': self.beta}


class NAG(optimizer):

    def __init__(self, cli_args: argparse.Namespace, Layers: list, lr: float=1e-3, momentum: float=0.9):

        self.lr = lr
        self.beta = momentum
        self.weight_decay = cli_args.weight_decay

        self.Layers = Layers
        self.num_layers = len(Layers)

        self.initialize_params()

    
    def initialize_params(self):
        
        self.W_Momentums = []
        self.b_Momentums = []

        for L in self.Layers:
            self.W_Momentums.append(np.zeros(L.W.shape))
            self.b_Momentums.append(np.zeros(L.b.shape))
        
    def nesterov_update(self):
        
        for i in range(self.num_layers):
            
            self.W_Momentums[i] *= self.beta
            self.b_Momentums[i] *= self.beta
            
            # Apply interim update
            self.Layers[i].W += self.W_Momentums[i]
            self.Layers[i].b += self.b_Momentums[i]

    def nesterov_revert(self):
        
        for i in range (self.num_layers):

            self.Layers[i].W -= self.W_Momentums[i]
            self.Layers[i].b -= self.b_Momentums[i]
    
    def update(self):

        for i,L in enumerate(self.Layers):
            
            # Compute velocity update
            self.W_Momentums[i] -= self.lr*(L.grad_W)
            self.b_Momentums[i] -= self.lr*(L.grad_b)

            # Regularize
            L.W *= (1-self.lr*self.weight_decay)

            # Apply update
            L.W += self.W_Momentums[i]
            L.b += self.b_Momentums[i]

    def log(self):

        return {'optimizer/lr': self.lr, 'optimizer/wd': self.weight_decay, 'optimizer/beta': self.beta}

class RMSProp(optimizer):

    def __init__(self, cli_args: argparse.Namespace, Layers: list, lr: float=1e-3, decay_rate: float=0.9):

        self.lr = lr
        self.beta = decay_rate
        self.weight_decay = cli_args.weight_decay

        self.Layers = Layers
        self.num_layers = len(Layers)

        self.initialize_params()
    
    def initialize_params(self):
        
        self.W_r = []
        self.b_r = []

        for L in self.Layers:
            self.W_r.append(np.zeros(L.W.shape))
            self.b_r.append(np.zeros(L.b.shape))
    
    def update(self):
        
        for i,L in enumerate(self.Layers):
            
            # Accumulate squared gradient
            self.W_r[i] *= self.beta
            self.b_r[i] *= self.beta

            self.W_r[i] += (1-self.beta)*np.square(L.grad_W)
            self.b_r[i] += (1-self.beta)*np.square(L.grad_b)

            # Regularize (only Weights)
            L.W *= (1-self.lr*self.weight_decay)

            # Apply update
            L.W -= self.lr * L.grad_W / (np.sqrt(self.W_r[i]) + 1e-6)
            L.b -= self.lr * L.grad_b / (np.sqrt(self.b_r[i]) + 1e-6)
    
    def log(self):
        
        return {'optimizer/lr': self.lr, 'optimizer/wd': self.weight_decay, 'optimizer/decay_rate': self.beta}