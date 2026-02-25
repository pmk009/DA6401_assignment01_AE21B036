"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np


def train_val_split(X, y, val_ratio=0.1, shuffle=True):
    n = len(X)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    split = int(n * (1 - val_ratio))

    train_idx = indices[:split]
    val_idx = indices[split:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

class Dataloader:

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 normalize: bool = True,
                 flatten: bool = False,
                 one_hot: bool = True,
                 num_classes: int=10,
                 to_float32: bool = True):

        self.X = X
        self.y = y

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.normalize = normalize
        self.flatten = flatten
        self.one_hot = one_hot
        self.num_classes = num_classes
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

        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices].copy()

        if self.to_float32:
            X_batch = X_batch.astype(np.float32)

        if self.normalize:
            X_batch /= 255.0

        if self.flatten:
            X_batch = X_batch.reshape(X_batch.shape[0], -1)

        if self.one_hot:
            y_batch = np.eye(self.num_classes)[y_batch]

        return X_batch, y_batch

    def __len__(self):

        return int(np.ceil(self.num_samples / self.batch_size))
    
