from torchvision.datasets import CIFAR10

from .common import DatasetBundle, make_loader, make_transforms, subset_by_classes


CIFAR10_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR10_STD = (0.24703233, 0.24348505, 0.26158768)


def build_cifar10(data_root: str, batch_size: int, num_workers: int, num_classes: int = 10, max_train_data: int = None, max_test_data: int = None) -> DatasetBundle:
    train_tfm, test_tfm = make_transforms(CIFAR10_MEAN, CIFAR10_STD, image_size=32)

    train_dataset = CIFAR10(root=data_root, train=True, download=True, transform=train_tfm)
    test_dataset = CIFAR10(root=data_root, train=False, download=True, transform=test_tfm)

    if num_classes < 10:
        train_dataset = subset_by_classes(train_dataset, num_classes)
        test_dataset = subset_by_classes(test_dataset, num_classes)

    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loader=make_loader(train_dataset, batch_size, num_workers, train=True, max_data=max_train_data),
        test_loader=make_loader(test_dataset, batch_size, num_workers, train=False, max_data=max_test_data),
        num_classes=num_classes,
        input_size=(3, 32, 32),
        dataset_name="cifar10"
    )
