"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import argparse
from neural_layer import *
from activations import *
from objective_functions import *
from optimizers import *
from utils.data_loader import *

Activations = {'sigmoid': Sigmoid, 'relu':ReLU}
Optimizers = {'sgd': SGD}
objective_functions = {'cross_entropy': Cross_Entropy}
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, input_size: int, output_size: int, output_act: Activation, cli_args: argparse.Namespace):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """

        self.input_size = input_size
        self.output_size = output_size
        self.output_act = output_act()

        # parsing the required arguments
        self.num_layers = cli_args.nhl 
        self.hidden_sizes = cli_args.sz
        self.activation = Activations[cli_args.a]
        self.weight_init = cli_args.w_i

        # Initializing the Neural network
        self.Layers = (
            [neural_layer(self.hidden_sizes[0], self.input_size, self.activation, self.weight_init)] +
            [neural_layer(self.hidden_sizes[i+1], self.hidden_sizes[i], self.activation, self.weight_init) for i in range(self.num_layers-1)] +
            [neural_layer(self.output_size, self.hidden_sizes[-1], Linear, self.weight_init)]
        )

        self.optimizer = Optimizers[cli_args.o](cli_args,self.Layers)
        self.objective = objective_functions[cli_args.l]()

    
    def forward(self, X: np.ndarray):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        hk_1 = X

        for k in range(self.num_layers+1):

            self.Layers[k].forward(hk_1)
            hk_1 = self.Layers[k].hk
        
        self.Layers[-1].hk = self.output_act.forward(self.Layers[-1].hk)
        
        return self.Layers[-1].hk
    
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
        loss = self.objective.loss(y_true,y_pred)
        del_k = self.objective.gradient(y_true,y_pred)

        for k in range(self.num_layers,-1,-1):

            del_k = self.Layers[k].backward(del_k)

        return np.mean(loss)        
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update()
        
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, shuffle: bool=True):
        """
        Train the network for specified epochs.
        """

        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio=0.1)
        train_dataloader = Dataloader(X_train,y_train,batch_size,shuffle,True, True)

        for e in range(epochs):

            epoch_loss = 0.0
            num_batches = 0

            print(f"\nEpoch [{e+1}/{epochs}]")

            for i, (X, y) in enumerate(train_dataloader):

                y_hat = self.forward(X)

                loss = self.backward(y, y_hat)

                self.update_weights()
                epoch_loss += loss
                num_batches += 1

                print(f"  Batch [{i+1}/{len(train_dataloader)}] "
                    f"Loss: {loss:.6f}", end="\r")

            val_loss, val_acc = self.evaluate(X_val, y_val)
            avg_loss = epoch_loss / num_batches
            print(f"\nEpoch [{e+1}/{epochs}] "
              f"Train Loss: {avg_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val Acc: {val_acc:.4f}")

    
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the network on given data.
        """
        
        eval_dataloader = Dataloader(X,y,64,False,True,True)
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i, (X, y) in enumerate(eval_dataloader):

            y_hat = self.forward(X)

            loss = self.objective.loss(y, y_hat)

            total_loss += np.sum(loss)

            preds = np.argmax(y_hat, axis=1)
            true_labels = np.argmax(y, axis=1)

            total_correct += np.sum(preds == true_labels)
            total_samples += y.shape[0]

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

