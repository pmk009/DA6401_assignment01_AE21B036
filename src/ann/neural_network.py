"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import argparse
from ann.neural_layer import *
from ann.activations import *
from ann.objective_functions import *
from ann.optimizers import *
from utils.data_loader import *
import wandb
import json

Activations = {'sigmoid': Sigmoid, 'relu':ReLU, 'tanh': Tanh, 'softmax': Softmax }
Optimizers = {'sgd': SGD, 'momentum': Momentum, 'nag': NAG, 'rmsprop': RMSProp, 'adam': Adam, 'nadam': NAdam}
objective_functions = {'cross_entropy': Cross_Entropy, "mean_squared_error": Mean_squared_Error}
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args, input_size: int=28*28, output_size: int=10, output_act: str='softmax'):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        print(cli_args)
        # Parsing the required arguments and Creating the attributes 
        self.cli_args = cli_args
        self.input_size = input_size
        self.output_size = output_size
        self.output_act_str = output_act
        self.output_act = Activations[output_act]()

        try:
            self.hidden_sizes = cli_args.hidden_size
        except:
            self.hidden_sizes = cli_args.hidden_layer_sizes
        self.num_layers = len(self.hidden_sizes)
        self.activation = Activations[cli_args.activation]
        self.weight_init = cli_args.weight_init

        # Initializing the Neural network
        self.Layers = (
            [neural_layer(self.hidden_sizes[0], self.input_size, self.activation, self.weight_init)] +
            [neural_layer(self.hidden_sizes[i+1], self.hidden_sizes[i], self.activation, self.weight_init) for i in range(self.num_layers-1)] +
            [neural_layer(self.output_size, self.hidden_sizes[-1], Linear, self.weight_init)]
        )

        # Initializing the Optimizer and Objective function
        self.optimizer = Optimizers[cli_args.optimizer](cli_args,self.Layers)
        self.objective = objective_functions[cli_args.loss]()

    
    def forward(self, X: np.ndarray , logits: bool = True):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        hk_1 = X # storing the activation of previous layer

        for k in range(self.num_layers+1):

            self.Layers[k].forward(hk_1) # Compute activation of current layer
            hk_1 = self.Layers[k].hk
        
        self.Layers[-1].hk = self.output_act.forward(self.Layers[-1].hk) # Logits to Probability
        
        return self.Layers[-1].ak if logits else self.Layers[-1].hk
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            loss_func: loss function 
            
        Returns:
            return grad_w, grad_b
        """
        
        del_k = self.objective.gradient(y_true,y_pred) # error signal with respect to post-activation stored

        for k in range(self.num_layers,-1,-1):

            del_k = self.Layers[k].backward(del_k) # Compute and store the gradients and error signal to previous layer

 
        # create explicit object arrays to avoid numpy trying to broadcast shapes
        grad_W = np.empty(self.num_layers, dtype=object)
        grad_b = np.empty(self.num_layers, dtype=object)
        for idx,i in enumerate(range(self.num_layers-1,-1,-1)):
            L = self.Layers[i]
            grad_W[idx] = L.grad_W.copy()
            grad_b[idx] = L.grad_b.copy()
            
        return grad_W, grad_b
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update() # Optimizer update

            
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int,save_path: str='', shuffle: bool=True, wandb_run: wandb.Run|None=None):
        """
        Train the network for specified epochs.
        """

        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio=0.2) # Train-Validation split
        train_dataloader = Dataloader(X_train,y_train,batch_size,shuffle,10,True)
        self.max_val_acc = 0. # recording max validation accuracy

        for e in range(epochs):

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            print(f"\nEpoch [{e+1}/{epochs}]")

            for i, (X, y) in enumerate(train_dataloader):

                self.optimizer.nesterov_update() # intermediate update for nestrov acceleration
                
                # Forward pass
                y_hat = self.forward(X, logits=False)

                # Loss and accuracy Calculation
                preds = np.argmax(y_hat, axis=1)
                true_labels = np.argmax(y, axis=1)
                total_correct += np.sum(preds == true_labels)
                total_samples += y.shape[0]
                loss = self.objective.loss(y,y_hat)
    
                # Backward pass
                grads = self.backward(y, y_hat)

                self.optimizer.nesterov_revert() # revert the intermediate update

                # Update the parameters
                self.update_weights()

                total_loss += np.sum(loss)

                print(f"  Batch [{i+1}/{len(train_dataloader)}] "
                    f"Loss: {np.mean(loss):.6f}", end="\r")

            val_loss, val_acc = self.evaluate(X_val, y_val) # Validation
            if val_acc>self.max_val_acc and save_path!='':
                self.max_val_acc = val_acc
                self.save_model(save_path, val_acc) # saving the best validation accuracy model
            avg_loss = total_loss/ total_samples
            avg_acc = total_correct/ total_samples
            if wandb_run:
                # Wandb Logging
                wandb_run.log({
                    'epoch': e+1,
                    'train/loss': avg_loss,
                    'train/acc': avg_acc,
                    'val/loss': val_loss,
                    'val/acc': val_acc}|self.optimizer.log())

            print(f"\nEpoch [{e+1}/{epochs}] "
              f"Train Loss: {avg_loss:.6f} | "
              f"Train Acc: {avg_acc:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val Acc: {val_acc:.4f}")
        

    
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the network on given data.
        """
        
        eval_dataloader = Dataloader(X,y,64,False)
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i, (X, y) in enumerate(eval_dataloader):

            y_hat = self.forward(X, logits=False)

            loss = self.objective.loss(y, y_hat)

            total_loss += np.sum(loss)

            preds = np.argmax(y_hat, axis=1)
            true_labels = np.argmax(y, axis=1)

            total_correct += np.sum(preds == true_labels)
            total_samples += y.shape[0]

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy
    

    def set_weights(self,weights):

        for i, layer in enumerate(self.Layers):
            layer.W = weights[f"W{i}"].copy()
            layer.b = weights[f"b{i}"].copy()
    
    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.Layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d
    
    def save_model(self,path: str, val_acc: float=0.0):
        """
        Save the current state of the model
        """
        save_dict = self.get_weights()

        Architecture = {
            "Activation": str(self.cli_args.activation),
            "batch_size": str(self.cli_args.batch_size),
            "Dataset": str(self.cli_args.dataset),
            "epochs": str(self.cli_args.epochs),
            "hidden sizes": ','.join(map(str, self.hidden_sizes)),
            "Num_layers": len(self.hidden_sizes),
            "Learning rate": str(self.cli_args.learning_rate),
            "Loss func": str(self.cli_args.loss),
            "Optimizer": str(self.cli_args.optimizer),
            "weight decay": str(self.cli_args.weight_decay),
            "Initialization": str(self.cli_args.weight_init)
        }


        np.save(f'{path}_model.npy', save_dict)

    @classmethod
    def load(cls, path: str):

        """
        Old load model used before skeleton update
        """
        data = np.load(path, allow_pickle=True)

        model = cls(argparse.Namespace(**data["cli_args"][0]))

        for i, layer in enumerate(model.Layers):
            layer.W = data[f"W_{i}"].copy()
            layer.b = data[f"b_{i}"].copy()

        return model
        
            


