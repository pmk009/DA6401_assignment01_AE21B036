"""
Separate file for Analysing Dead neuron problem.
"""


from tensorflow.keras import datasets
from ann.neural_network import *
import wandb
from utils.data_loader import *


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
    parser.add_argument('-sz', '--hidden_sizes', type=str, default="64,32", help='Comma-separated number of neurons (e.g., "128,64,32")')
    # parser.add_argument('-sz', '--hidden_sizes', type=int, nargs='+', help= 'Number of neurons in each hidden layer (list of values)')
    parser.add_argument('-a', '--activation', choices= ['sigmoid', 'tanh', 'relu'], default='sigmoid', help= 'choice of sigmoid, tanh, relu')
    parser.add_argument('-l', '--loss', choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy', help= '(\'cross_entropy\', \'mse\')')
    parser.add_argument('-w_i', '--weight_init', choices= ['random', 'xavier'], default='random', help= 'choice of random or xavier')
    parser.add_argument('-wd', '--weight_decay', type= float, default=1e-3, help= 'Weight decay for L2 regularization')
    parser.add_argument('-wbp', '--wandb_project', default= '', help= 'W&B project name')
    parser.add_argument('-sp', '--save_path', default='models/model', help= 'Path to save trained model')
    parser.add_argument('-z_i','--zero_init', type=str, default= 'false')
    return parser.parse_args()


def train_log_grad(X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int,NN, wandb_run: wandb.Run|None=None,):

    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio=0.2)
    train_dataloader = Dataloader(X_train,y_train,batch_size,True,True, True, augment=False)
    NN.max_val_acc = 0.
    iter_num = 0
    log_idxs = np.arange(0,128,np.ceil(128/5),dtype=int)
    for e in range(epochs):

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        print(f"\nEpoch [{e+1}/{epochs}]")

        for L in NN.Layers:
            num_neurons = L.grad_b.shape[0]  # assuming (batch_size, neurons)

        for i, (X, y) in enumerate(train_dataloader):
            iter_num+=1
            NN.optimizer.nesterov_update()
            y_hat = NN.forward(X)


            preds = np.argmax(y_hat, axis=1)
            true_labels = np.argmax(y, axis=1)
            total_correct += np.sum(preds == true_labels)
            total_samples += y.shape[0]

            loss = NN.backward(y, y_hat)
            wandb.log({'iter': iter_num})

            L = NN.Layers[0]
            wandb_run.log({
                    f'debug/delta_a{j}': L.grad_b[j] for j in log_idxs
            })

            NN.optimizer.nesterov_revert()

            
            NN.update_weights()
            total_loss += np.sum(loss)
            

            print(f"  Batch [{i+1}/{len(train_dataloader)}] "
                f"Loss: {np.mean(loss):.6f}", end="\r")

        val_loss, val_acc = NN.evaluate(X_val, y_val)
        avg_loss = total_loss/ total_samples
        avg_acc = total_correct/ total_samples

        
        if wandb_run:
            wandb_run.log({
                'epoch': e+1,
                'train/loss': avg_loss,
                'train/acc': avg_acc,
                'val/loss': val_loss,
                'val/acc': val_acc}|NN.optimizer.log())

        print(f"\nEpoch [{e+1}/{epochs}] "
            f"Train Loss: {avg_loss:.6f} | "
            f"Train Acc: {avg_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Acc: {val_acc:.4f}")
        

def main():


    args = parse_arguments()

    (x_train,y_train),_ = datasets.mnist.load_data()

    NN = NeuralNetwork(28*28,10,'softmax',args) 

    if args.zero_init == 'true':
        print('making parameters zero....')
        for L in NN.Layers:
            np.multiply(0,L.W,out=L.W)
            np.multiply(0,L.b,out=L.b)
    
    run = wandb.init(
    entity="DA6401_assignment01",
    project=args.wandb_project,
    config=vars(args)
    )

    train_log_grad(x_train,y_train,args.epochs,args.batch_size,NN,run)


if __name__ == '__main__':
    main()
