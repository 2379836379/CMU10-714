import os
import pickle
from typing import Optional, List
import numpy as np
from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, base_folder: str, train: bool, p: Optional[int] = 0.5, transforms: Optional[List] = None):
        super().__init__(transforms=transforms)
        files = [f'data_batch_{i}' for i in range(1, 6)] if train else ['test_batch']
        xs, ys = [], []
        for name in files:
            with open(os.path.join(base_folder, name), 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            xs.append(batch[b'data'])
            ys.extend(batch[b'labels'])
        X = np.concatenate(xs, axis=0).astype(np.float32) / 255.0
        self.X = X.reshape((-1, 3, 32, 32))
        self.y = np.array(ys, dtype=np.int64)

    def __getitem__(self, index) -> object:
        X, y = self.X[index], self.y[index]
        if isinstance(index, (list, np.ndarray, slice)):
            if self.transforms is not None:
                X = np.stack([self.apply_transforms(x.transpose(1, 2, 0)).transpose(2, 0, 1) for x in X], axis=0)
            return X, y
        if self.transforms is not None:
            X = self.apply_transforms(X.transpose(1, 2, 0)).transpose(2, 0, 1)
        return X, y

    def __len__(self) -> int:
        return self.X.shape[0]
