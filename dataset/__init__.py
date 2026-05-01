from typing import Optional

from .cifar10 import build_cifar10
from .cifar100 import build_cifar100
from .cinic10 import build_cinic10


def build_dataset(name: str, data_root: str, batch_size: int, num_workers: int, max_train_data: Optional[int] = None, max_test_data: Optional[int] = None):
    name = name.lower()
    if name == "cifar10":
        return build_cifar10(
            data_root,
            batch_size,
            num_workers,
            num_classes=10,
            max_train_data=max_train_data,
            max_test_data=max_test_data,
        )
    if name == "cifar100":
        return build_cifar100(
            data_root,
            batch_size,
            num_workers,
            num_classes=100,
            max_train_data=max_train_data,
            max_test_data=max_test_data,
        )
    if name == "cinic10":
        return build_cinic10(
            data_root,
            batch_size,
            num_workers,
            num_classes=10,
            max_train_data=max_train_data,
            max_test_data=max_test_data,
        )
    raise ValueError(f"Unknown dataset: {name}")


def get_num_classes(name: str) -> int:
    name = name.lower()
    if name in {"cifar10", "cinic10"}:
        return 10
    if name == "cifar100":
        return 100
    raise ValueError(f"Unknown dataset: {name}")
