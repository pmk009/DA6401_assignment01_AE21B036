"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
from ann.neural_network import *
from utils.data_loader import Dataloader
import os
from tensorflow.keras import datasets


def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], default= 'mnist', help= '\'mnist\' or \'fashion_mnist\'')
    parser.add_argument('-e', '--epochs', type= int, default= 30, help= 'Number of training epochs')
    parser.add_argument('-b', '--batch_size', type= int, default= 32, help= 'Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type= float, default= 0.000232944, help= 'Learning rate for optimizer')
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help= '\'sgd\', \'momentum\', \'nag\', \'rmsprop\', \'adam\', \'nadam\'' )
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help= 'Number of hidden layers')
    parser.add_argument('-sz', '--hidden_sizes', type=str, default="128,128,64", help='Comma-separated number of neurons (e.g., "128,64,32")')
    parser.add_argument('-a', '--activation', choices= ['sigmoid', 'tanh', 'relu'], default='relu', help= 'choice of sigmoid, tanh, relu')
    parser.add_argument('-l', '--loss', choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy', help= '(\'cross_entropy\', \'mse\')')
    parser.add_argument('-w_i', '--weight_init', choices= ['random', 'xavier'], default='xavier', help= 'choice of random or xavier')
    parser.add_argument('-wd', '--weight_decay', type= float, default=1e-4, help= 'Weight decay for L2 regularization')
    parser.add_argument('-w_p', '--wandb_project', default= '', help= 'W&B project name')
    parser.add_argument('-mp', '--model_path', default='best_model.npz', help= 'Path to save trained model')
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True)
    return data


def evaluate_model(model, dataloader): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    for X, y in dataloader:

        logits = model.forward(X, logits=True)
        softmax = Softmax()
        probs = softmax.forward(logits)
        probs = np.clip(probs, 1e-12, 1.0 - 1e-12)

        batch_size = X.shape[0]

        loss = -np.sum(y * np.log(probs)) / batch_size
        total_loss += loss * batch_size
        total_samples += batch_size

        all_preds.append(np.argmax(probs, axis=1))
        all_targets.append(np.argmax(y, axis=1))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    accuracy = np.mean(y_pred == y_true)

    num_classes = len(np.unique(y_true))

    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):

        tp = np.sum((y_pred == c) & (y_true == c)) # number of True Positive
        fp = np.sum((y_pred == c) & (y_true != c)) # number of False Positive
        fn = np.sum((y_pred != c) & (y_true == c)) # number of False Negative

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "logits": logits,
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1": np.mean(f1_list)
    }

def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    # Loading the Dataset    
    if args.dataset == 'mnist':
        _,(x,y) = datasets.mnist.load_data()
    elif args.dataset == 'fashion_mnist':
        _,(x,y) = datasets.fashion_mnist.load_data()

    # Loading the Neural Network
    input_size = int(x.shape[1]*x.shape[2])
    output_size = 10
    output_act = 'softmax' 
    model = NeuralNetwork(args,input_size,output_size,output_act)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    eval_dataloader = Dataloader(x,y,args.batch_size,False,10,False)
    metrics = evaluate_model(model,eval_dataloader)

    print("\nEvaluation Results")
    print(f"Loss      : {metrics['loss']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")

    return metrics
if __name__ == '__main__':
    main()
