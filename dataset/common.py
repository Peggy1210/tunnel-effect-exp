from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


@dataclass
class DatasetBundle:
    dataset_name: str
    train_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    input_size: Tuple[int, int, int]


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, train: bool, max_data: Optional[int] = None) -> DataLoader:

    if max_data is not None and max_data > 0:
        indices = np.random.choice(len(dataset), size=min(max_data, len(dataset)), replace=False)
        dataset = Subset(dataset, indices.tolist())

    if batch_size == -1:
        batch_size = len(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def make_transforms(
    mean,
    std,
    image_size: int = 32,
    train_aug: bool = False,
):
    train_tfms = [
        transforms.Resize((image_size, image_size)),
    ]
    if train_aug:
        train_tfms.extend([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(train_tfms), test_tfms


def subset_by_classes(dataset: Dataset,  num_classes: int):
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("This dataset wrapper expects a `targets` attribute.")

    data_indices = []
    for i in range(num_classes):
        class_indices = [idx for idx, label in enumerate(dataset.targets) if label == i]
        data_indices.extend(class_indices)

    return Subset(dataset, data_indices)