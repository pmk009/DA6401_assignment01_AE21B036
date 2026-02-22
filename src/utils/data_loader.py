"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np

import numpy as np

class Dataloader:

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 normalize: bool = True,
                 flatten: bool = False,
                 to_float32: bool = True):

        self.X = X
        self.y = y

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.normalize = normalize
        self.flatten = flatten
        self.to_float32 = to_float32

        self.num_samples = len(self.X)
        self.indices = np.arange(self.num_samples)

    def __iter__(self):

        if self.shuffle:
            np.random.shuffle(self.indices)

        self.current = 0
        return self

    def __next__(self):

        if self.current >= self.num_samples:
            raise StopIteration

        start = self.current
        end = start + self.batch_size

        batch_indices = self.indices[start:end]
        self.current = end

        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        if self.to_float32:
            X_batch = X_batch.astype(np.float32)

        if self.normalize:
            X_batch /= 255.0

        if self.flatten:
            X_batch = X_batch.reshape(X_batch.shape[0], -1)

        return X_batch, y_batch

    def __len__(self):

        return int(np.ceil(self.num_samples / self.batch_size))
    
