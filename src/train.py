"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse

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

    parser.add_argument('-d', choices=['mnist', 'fashion_mnist'], default= 'mnist', help= '--dataset: \'mnist\' or \'fashion_mnist\'')
    parser.add_argument('-e', type= int, default= 1000, help= '--epochs: Number of training epochs')
    parser.add_argument('-b', type= int, default= 64, help= '--batch_size: Mini-batch size')
    parser.add_argument('-lr', type= float, default= 1e-3, help= '--learning_rate: Learning rate for optimizer')
    parser.add_argument('-o', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='sgd', help= '--optimizer: \'sgd\', \'momentum\', \'nag\', \'rmsprop\', \'adam\', \'nadam\'' )
    parser.add_argument('-nhl', type=int, default=2, help= '--num_layers: Number of hidden layers')
    parser.add_argument('-sz', type=int, nargs='+', help= '--hidden_size: Number of neurons in each hidden layer (list of values)')
    parser.add_argument('-a', choices= ['sigmoid', 'tanh', 'relu'], default='sigmoid', help= '--activation: choice of sigmoid, tanh, relu')
    parser.add_argument('-l', choices=['mean_squared_error', 'cross_entropy'], default='mean_squared_error', help= 'Loss function (\'cross_entropy\', \'mse\')')
    parser.add_argument('-w_i', choices= ['random', 'xavier'], default='random', help= '--weight_init: choice of random or xavier')
    parser.add_argument('-wd', type= float, default=0., help= '--weight_decay: Weight decay for L2 regularization')
    parser.add_argument('-wb_project', default= '', help= '--wandb_project: W&B project name')
    parser.add_argument('-save_path', default='', help= '--model_save_path: Path to save trained model')
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    
    print("Training complete!")


if __name__ == '__main__':
    main()
