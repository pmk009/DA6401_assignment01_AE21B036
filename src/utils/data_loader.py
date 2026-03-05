"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np


def train_val_split(X, y, val_ratio=0.2, shuffle=True):
    """
    To split the Train data into train and validation 
    """
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
                 num_classes: int = 10,
                 augment: bool = False):

        self.X = X
        self.y = y

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.augment = augment

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

        X_batch = X_batch.astype(np.float32) # To float32 
        X_batch /= 255.0 # Normalizing the data

        if self.augment:
            X_batch = self.apply_augmentation(X_batch) # Data augmentation

        X_batch = X_batch.reshape(X_batch.shape[0], -1) # Flattening to 1D array

        y_batch = np.eye(self.num_classes)[y_batch] # One hot encoding the output

        return X_batch, y_batch

    def __len__(self):

        return int(np.ceil(self.num_samples / self.batch_size))

    def apply_augmentation(self, images):

        if np.random.rand() < 0.8:
            images = self.random_shift(images, max_shift=2)

        if np.random.rand() < 0.5:
            images = self.cutout(images, size=6)

        return images

    def random_shift(self, images, max_shift=2):

        B, H, W = images.shape
        shifted = np.zeros_like(images)

        for i in range(B):
            dx = np.random.randint(-max_shift, max_shift + 1)
            dy = np.random.randint(-max_shift, max_shift + 1)

            x1 = max(0, dx)
            x2 = min(H, H + dx)
            y1 = max(0, dy)
            y2 = min(W, W + dy)

            shifted[i, x1:x2, y1:y2] = images[i,
                                              x1 - dx:x2 - dx,
                                              y1 - dy:y2 - dy]

        return shifted

    def cutout(self, images, size=6):

        B, H, W = images.shape

        for i in range(B):
            x = np.random.randint(0, H)
            y = np.random.randint(0, W)

            x1 = np.clip(x - size // 2, 0, H)
            x2 = np.clip(x + size // 2, 0, H)
            y1 = np.clip(y - size // 2, 0, W)
            y2 = np.clip(y + size // 2, 0, W)

            images[i, x1:x2, y1:y2] = 0.0

        return images