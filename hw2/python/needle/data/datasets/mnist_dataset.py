from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as f:
            _ = int.from_bytes(f.read(4), "big")
            num_images = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            self.images = (
                np.frombuffer(f.read(), dtype=np.uint8)
                .reshape(num_images, rows * cols)
                .astype(np.float32)
                / 255.0
            )
        with gzip.open(label_filename, "rb") as f:
            _ = int.from_bytes(f.read(4), "big")
            num_labels = int.from_bytes(f.read(4), "big")
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = self.images[index]
        labels = self.labels[index]

        if self.transforms is None:
            return images, labels

        single = np.isscalar(labels)
        if single:
            img = images.reshape(28, 28, 1)
            img = self.apply_transforms(img)
            return img.reshape(-1), labels

        transformed = []
        for img in images:
            transformed.append(self.apply_transforms(img.reshape(28, 28, 1)).reshape(-1))
        return np.stack(transformed, axis=0), labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION
