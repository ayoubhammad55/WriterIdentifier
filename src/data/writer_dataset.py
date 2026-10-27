"""
PyTorch Dataset wrappers for writer identification.

WriterDataset     – loads form images, applies transforms, returns (image, label)
PairDataset       – yields (img_a, img_b, same_writer) pairs for Siamese networks
"""

import os
from typing import Callable, List, Optional, Tuple

import cv2 as cv
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object          # fallback base class

from utils.constants import IMAGE_HEIGHT, IMAGE_WIDTH


# ---------------------------------------------------------------------------
# Default image transforms (numpy -> tensor)
# ---------------------------------------------------------------------------

def _default_transform(img: np.ndarray) -> "torch.Tensor":
    """
    Resize, normalise, and convert a grayscale numpy image to a float tensor.
    Shape: (1, H, W) in [0, 1].
    """
    import torch
    import torchvision.transforms.functional as TF

    pil = Image.fromarray(img)
    pil = pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
    tensor = TF.to_tensor(pil)          # (1, H, W) float in [0,1]
    tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])   # -> [-1,1]
    return tensor


# ---------------------------------------------------------------------------
# WriterDataset
# ---------------------------------------------------------------------------

class WriterDataset(Dataset):
    """
    PyTorch Dataset for writer identification.

    Parameters
    ----------
    image_paths : list[str]   – absolute paths to form images
    labels      : list[int]   – integer writer label (0-indexed)
    transform   : callable    – image transform (numpy ndarray -> Tensor).
                                Defaults to resize + normalise.
    augment     : bool        – whether to apply random augmentations
                                (flip, brightness jitter) during training.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels:      List[int],
        transform:   Optional[Callable] = None,
        augment:     bool = False,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for WriterDataset.")

        assert len(image_paths) == len(labels), \
            "image_paths and labels must have the same length."

        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform or _default_transform
        self.augment     = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", int]:
        import torch

        path  = self.image_paths[idx]
        label = self.labels[idx]

        img = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if img is None:
            # Return a blank image with the correct shape if the file is corrupt
            img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8) * 255

        if self.augment:
            img = self._augment(img)

        tensor = self.transform(img)
        return tensor, label

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        """Lightweight augmentation: random horizontal flip + brightness jitter."""
        # Brightness jitter only; do not mirror handwritten text.

        # Brightness jitter
        factor = np.random.uniform(0.85, 1.15)
        img    = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return img

    @classmethod
    def from_loader(
        cls,
        loader,           # IAMLoader instance (already .load()ed)
        split: str,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ) -> "WriterDataset":
        """Convenience constructor: build from an IAMLoader split."""
        paths, labels = loader.get_split(split)
        return cls(paths, labels, transform=transform, augment=augment)


# ---------------------------------------------------------------------------
# PairDataset  (Siamese / contrastive)
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """
    Yields pairs of images and a binary same-writer label.

    Balanced: 50% same-writer pairs, 50% different-writer pairs.
    Pairs are sampled randomly on each epoch (deterministic per index if seeded).
    """

    def __init__(
        self,
        image_paths: List[str],
        labels:      List[int],
        num_pairs:   int = 10000,
        transform:   Optional[Callable] = None,
        seed:        int = 42,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PairDataset.")

        self.image_paths = image_paths
        self.labels      = labels
        self.num_pairs   = num_pairs
        self.transform   = transform or _default_transform

        # Group indices by label
        from collections import defaultdict
        self._by_label: dict = defaultdict(list)
        for i, lbl in enumerate(labels):
            self._by_label[lbl].append(i)
        self._label_list = list(self._by_label.keys())

        rng = np.random.default_rng(seed)
        self._pairs = self._generate_pairs(rng)

    def _generate_pairs(self, rng) -> List[Tuple[int, int, int]]:
        pairs = []
        n = self.num_pairs // 2

        # Same-writer pairs
        for _ in range(n):
            lbl  = rng.choice(self._label_list)
            pool = self._by_label[lbl]
            if len(pool) < 2:
                continue
            a, b = rng.choice(pool, size=2, replace=False)
            pairs.append((int(a), int(b), 1))

        # Different-writer pairs
        labels_arr = np.array(self._label_list)
        for _ in range(n):
            la, lb = rng.choice(labels_arr, size=2, replace=False)
            a = int(rng.choice(self._by_label[int(la)]))
            b = int(rng.choice(self._by_label[int(lb)]))
            pairs.append((a, b, 0))

        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        import torch
        ia, ib, same = self._pairs[idx]

        def _load(i):
            img = cv.imread(self.image_paths[i], cv.IMREAD_GRAYSCALE)
            if img is None:
                img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8) * 255
            return self.transform(img)

        return _load(ia), _load(ib), torch.tensor(same, dtype=torch.float32)
