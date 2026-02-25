"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
from tensorflow.keras import datasets
from ann.neural_network import *

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], default= 'mnist', help= '\'mnist\' or \'fashion_mnist\'')
    parser.add_argument('-e', '--epochs', type= int, default= 20, help= 'Number of training epochs')
    parser.add_argument('-b', '--batch_size', type= int, default= 64, help= 'Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type= float, default= 1e-3, help= 'Learning rate for optimizer')
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='sgd', help= '\'sgd\', \'momentum\', \'nag\', \'rmsprop\', \'adam\', \'nadam\'' )
    parser.add_argument('-nhl', '--num_layers', type=int, default=2, help= 'Number of hidden layers')
    parser.add_argument('-sz', '--hidden_sizes', type=int, nargs='+', help= 'Number of neurons in each hidden layer (list of values)')
    parser.add_argument('-a', '--activation', choices= ['sigmoid', 'tanh', 'relu'], default='sigmoid', help= 'choice of sigmoid, tanh, relu')
    parser.add_argument('-l', '--loss', choices=['mean_squared_error', 'cross_entropy'], default='mean_squared_error', help= '(\'cross_entropy\', \'mse\')')
    parser.add_argument('-w_i', '--weight_init', choices= ['random', 'xavier'], default='random', help= 'choice of random or xavier')
    parser.add_argument('-wd', '--weight_decay', type= float, default=1e-3, help= 'Weight decay for L2 regularization')
    parser.add_argument('-wbp', '--wandb_project', default= '', help= 'W&B project name')
    parser.add_argument('-sp', '--save_path', default='models/model', help= 'Path to save trained model')
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    
    dataset = {'mnist': datasets.mnist, 'fashion_mnist': datasets.fashion_mnist}
    (x_train,y_train),(x_test,y_test) = dataset[args.dataset].load_data()

    input_size = int(x_train.shape[1]*x_train.shape[2])
    output_size = 10
    output_act = 'softmax' 
    NN = NeuralNetwork(input_size,output_size,output_act,args)
    print('Neural Network initialized..')

    wandb_run = None
    if args.wandb_project != '':
        wandb_run = wandb.init(
    entity="DA6401",
    project=args.wandb_project,
    config=vars(args)
    )
    
    NN.train(x_train,y_train,args.epochs,args.batch_size,wandb_run=wandb_run,save_path= args.save_path)
    
    print("Training complete!")


if __name__ == '__main__':
    main()
