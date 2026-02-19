"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np

class Dataloader:

    def __init__(self,X: np.ndarray,y: np.ndarray, batch_size: int=32, shuffle: bool=True):

        # Initializing attributes
        self.X = X
        self.y = y
        
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_samples = len(self.X)
        self.indices = np.arange(self.num_samples)

    
    def __iter__(self):
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current = 0
        return self
    
    def __next__(self):
        
        # check if full dataset is iterated
        if self.current >= self.num_samples:
            raise StopIteration
        
        start = self.current
        end = start + self.batch_size

        batch_indices = self.indices[start:end]
        self.current = end

        return self.X[batch_indices],self.y[batch_indices]
    
    def __len__(self):
        
        return int(np.ceil(self.num_samples/self.batch_size))
    
