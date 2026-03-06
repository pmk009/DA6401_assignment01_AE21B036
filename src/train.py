"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
import os
from tensorflow.keras import datasets
from ann.neural_network import *
from inference import evaluate_model

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
    parser.add_argument('-e', '--epochs', type= int, default= 30, help= 'Number of training epochs')
    parser.add_argument('-b', '--batch_size', type= int, default= 32, help= 'Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type= float, default= 0.000232944, help= 'Learning rate for optimizer')
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help= '\'sgd\', \'momentum\', \'nag\', \'rmsprop\', \'adam\', \'nadam\'' )
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help= 'Number of hidden layers')
    # parser.add_argument('-sz', '--hidden_size', type=str, default="128,128,64", help='Comma-separated number of neurons (e.g., "128,64,32")')
    parser.add_argument('-sz', '--hidden_size', type=int, default=[128,128,64], nargs='+', help= 'Number of neurons in each hidden layer (list of values)')
    parser.add_argument('-a', '--activation', choices= ['sigmoid', 'tanh', 'relu'], default='relu', help= 'choice of sigmoid, tanh, relu')
    parser.add_argument('-l', '--loss', choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy', help= '(\'cross_entropy\', \'mse\')')
    parser.add_argument('-w_i', '--weight_init', choices= ['random', 'xavier'], default='xavier', help= 'choice of random or xavier')
    parser.add_argument('-wd', '--weight_decay', type= float, default=1e-4, help= 'Weight decay for L2 regularization')
    parser.add_argument('-w_p', '--wandb_project', default= '', help= 'W&B project name')
    parser.add_argument('-sp', '--save_path', default='', help= 'Path to save trained model')
    
    return parser.parse_args()

def train(args,x_train: np.ndarray,y_train: np.ndarray, wandb_run: wandb.Run|None=None):

    # Initializing the Neural Network
    input_size = int(x_train.shape[1]*x_train.shape[2])
    output_size = 10
    output_act = 'softmax' 
    NN = NeuralNetwork(args,input_size,output_size,output_act)
    
    # Training
    NN.train(x_train,y_train,args.epochs,args.batch_size,wandb_run=wandb_run,save_path= args.save_path)

    return NN

def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # Load Dataset
    dataset = {'mnist': datasets.mnist, 'fashion_mnist': datasets.fashion_mnist}
    (x_train,y_train),(x_test,y_test) = dataset[args.dataset].load_data()

    # Wandb Run initialization
    wandb_run = None
    if args.wandb_project != '':
        wandb_run = wandb.init(
    entity="DA6401_assignment01",
    project=args.wandb_project,
    config=vars(args)
    )
    
    # Keeping track of the best model from Validation accuracy
    if os.path.exists("models/best_model.npz"):
        data = np.load("models/best_model.npz", allow_pickle=True)
        best_val_acc = float(data["val_acc"])
    else:
        best_val_acc = 0.0
    
    # Training and retreiving the model
    model = train(args,x_train,y_train, wandb_run)
    print('-'*30)
    print("Training complete....")

    # if model.max_val_acc>best_val_acc:
    
    #     model.save_model(path="models/best_model.npz", val_acc=model.max_val_acc)
    #     print("New best model Saved.")
    
    # Evaluvating the model on Test data
    print('-'*30)
    eval_dataloader = Dataloader(x_test,y_test,batch_size=128,shuffle=False)
    metrics = evaluate_model(model, eval_dataloader)
    print("\nEvaluation Results")
    print(f"Loss      : {metrics['loss']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")

    # Logging Test metrics
    if args.wandb_project != '':
        wandb_run.log({"test/loss": metrics["loss"], "test/acc": metrics["accuracy"], "val/max_acc": model.max_val_acc})



if __name__ == '__main__':
    main()
