"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
from ann.neural_network import *
from utils.data_loader import Dataloader
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
    parser.add_argument('-mp', '--model_path', default='models/model', help= 'Path to saved model')
    parser.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], default= 'mnist', help= '\'mnist\' or \'fashion_mnist\'')
    parser.add_argument('-b', '--batch_size', type= int, default= 64, help= 'batch size for evaluation')
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    return NeuralNetwork.load(path=model_path)


def evaluate_model(model, dataloader): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    eps = 1e-12

    for X, y in dataloader:

        probs = model.forward(X)
        probs = np.clip(probs, eps, 1.0 - eps)

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

        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
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
    
    model = load_model(args.model_path)
    if args.dataset == 'mnist':
        _,(x,y) = datasets.mnist.load_data()
    elif args.dataset == 'fashion_mnist':
        _,(x,y) = datasets.fashion_mnist.load_date()

    eval_dataloader = Dataloader(x,y,args.batch_size,False,augment=False)
    metrics = evaluate_model(model,eval_dataloader)

    print("\nEvaluation Results")
    print("-" * 30)
    print(f"Loss      : {metrics['loss']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print("-" * 30)
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
