from torchvision.datasets import CIFAR100

from .common import DatasetBundle, make_loader, make_transforms, subset_by_classes


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_cifar100(data_root: str, batch_size: int, num_workers: int, num_classes: int = 100, max_train_data: int = None, max_test_data: int = None):
    train_tfm, test_tfm = make_transforms(CIFAR100_MEAN, CIFAR100_STD, image_size=32)

    train_dataset = CIFAR100(root=data_root, train=True, download=True, transform=train_tfm)
    test_dataset = CIFAR100(root=data_root, train=False, download=True, transform=test_tfm)

    if num_classes < 100:
        train_dataset = subset_by_classes(train_dataset, num_classes)
        test_dataset = subset_by_classes(test_dataset, num_classes)

    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loader=make_loader(train_dataset, batch_size, num_workers, train=True, max_data=max_train_data),
        test_loader=make_loader(test_dataset, batch_size, num_workers, train=False, max_data=max_test_data),
        num_classes=num_classes,
        input_size=(3, 32, 32),
        dataset_name="cifar100"
    )
