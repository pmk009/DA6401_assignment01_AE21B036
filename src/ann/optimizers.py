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
        
        return {'optimizer/lr': self.lr, 'optimizer/wd': self.weight_decay}


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
        
        return {'optimizer/lr': self.lr, 'optimizer/wd': self.weight_decay, 'optimizer/momentum_decay': self.beta}


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

        return {'optimizer/lr': self.lr, 'optimizer/wd': self.weight_decay, 'optimizer/momentum_decay': self.beta}

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
        
        return {'optimizer/lr': self.lr, 'optimizer/wd': self.weight_decay, 'optimizer/second_moment_decay': self.beta}
    

class Adam(optimizer):
    
    def __init__(
            self, 
            cli_args: argparse.Namespace, 
            Layers: list, 
            lr: float = 1e-3,
            decay_rate_1: float = 0.9,
            decay_rate_2: float = 0.999
    ):
        
        self.t = 0
        self.lr = lr
        self.beta1 = decay_rate_1
        self.beta2 = decay_rate_2
        self.weight_decay = cli_args.weight_decay

        self.Layers = Layers
        self.num_layers = len(Layers)

        self.initialize_params()

    def initialize_params(self):
        
        self.W_s = []
        self.W_r = []
        self.b_s = []
        self.b_r = []
        for L in self.Layers:
            self.W_s.append(np.zeros(shape=L.W.shape))
            self.W_r.append(np.zeros(shape=L.W.shape))
            self.b_s.append(np.zeros(shape=L.b.shape))
            self.b_r.append(np.zeros(shape=L.b.shape))
        
    def update(self):
        
        self.t += 1
        
        for i,L in enumerate(self.Layers):

            # update first moment estimate
            self.W_s[i] *= self.beta1 ; self.W_s[i] += (1-self.beta1)*L.grad_W
            self.b_s[i] *= self.beta1 ; self.b_s[i] += (1-self.beta1)*L.grad_b

            # update second mment estimate
            self.W_r[i] *= self.beta2 ; self.W_r[i] += (1-self.beta2)*np.square(L.grad_W)
            self.b_r[i] *= self.beta2 ; self.b_r[i] += (1-self.beta2)*np.square(L.grad_b)

            # correct bias in first moment
            W_s_ = self.W_s[i]/(1-self.beta1**self.t)
            b_s_ = self.b_s[i]/(1-self.beta1**self.t)

            # corrent bias in second moment
            W_r_ = self.W_r[i]/(1-self.beta2**self.t)
            b_r_ = self.b_r[i]/(1-self.beta2**self.t)



            # compute update
            del_W = - self.lr * np.multiply(W_s_, np.reciprocal(np.sqrt(W_r_)+1e-8))
            del_b = - self.lr * np.multiply(b_s_, np.reciprocal(np.sqrt(b_r_)+1e-8))

            # Regularize (only weights)
            L.W *= (1-self.lr*self.weight_decay)

            # Appky update
            L.W += del_W
            L.b += del_b
    
    def log(self):
        
        return {'optimizer/lr': self.lr, 'optimizer/wd': self.weight_decay, 'optimizer/momentum_decay': self.beta1, 'optimizer/second_moment_decay': self.beta2}
    


class NAdam(optimizer):
    
    def __init__(
            self, 
            cli_args: argparse.Namespace, 
            Layers: list, 
            lr: float = 1e-3,
            decay_rate_1: float = 0.9,
            decay_rate_2: float = 0.999
    ):
        
        self.t = 0
        self.lr = lr
        self.beta1 = decay_rate_1
        self.beta2 = decay_rate_2
        self.weight_decay = cli_args.weight_decay

        self.Layers = Layers
        self.num_layers = len(Layers)

        self.initialize_params()

    def initialize_params(self):
        
        self.W_s = []
        self.W_r = []
        self.b_s = []
        self.b_r = []
        for L in self.Layers:
            self.W_s.append(np.zeros(shape=L.W.shape))
            self.W_r.append(np.zeros(shape=L.W.shape))
            self.b_s.append(np.zeros(shape=L.b.shape))
            self.b_r.append(np.zeros(shape=L.b.shape))
        
    def update(self):
        
        self.t += 1
        
        for i,L in enumerate(self.Layers):

            # update first moment estimate
            self.W_s[i] *= self.beta1 ; self.W_s[i] += (1-self.beta1)*L.grad_W
            self.b_s[i] *= self.beta1 ; self.b_s[i] += (1-self.beta1)*L.grad_b

            # update second mment estimate
            self.W_r[i] *= self.beta2 ; self.W_r[i] += (1-self.beta2)*np.square(L.grad_W)
            self.b_r[i] *= self.beta2 ; self.b_r[i] += (1-self.beta2)*np.square(L.grad_b)

            # correct bias in first moment
            W_s_ = self.W_s[i]/(1-self.beta1**self.t)
            b_s_ = self.b_s[i]/(1-self.beta1**self.t)

            # corrent bias in second moment
            W_r_ = self.W_r[i]/(1-self.beta2**self.t)
            b_r_ = self.b_r[i]/(1-self.beta2**self.t)



            # compute update
            del_W = - self.lr/(np.sqrt(W_r_)+1e-8) * (self.beta1*W_s_+ (1-self.beta1)*L.grad_W/(1-self.beta1**self.t))
            del_b = - self.lr/(np.sqrt(b_r_)+1e-8) * (self.beta1*b_s_+ (1-self.beta1)*L.grad_b/(1-self.beta1**self.t))

            # Regularize (only weights)
            L.W *= (1-self.lr*self.weight_decay)

            # Apply update
            L.W += del_W
            L.b += del_b
    
    def log(self):
        
        return {'optimizer/lr': self.lr, 'optimizer/wd': self.weight_decay, 'optimizer/momentum_decay': self.beta1, 'optimizer/second_moment_decay': self.beta2}
            

        
