from pathlib import Path

from torchvision.datasets import ImageFolder

from .common import DatasetBundle, make_loader, make_transforms, subset_by_classes


CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)


def build_cinic10(data_root: str, batch_size: int, num_workers: int, num_classes: int = 10, max_train_data: int = None, max_test_data: int = None):
    train_tfm, test_tfm = make_transforms(CINIC10_MEAN, CINIC10_STD, image_size=32)

    root = Path(data_root) / "cinic10"
    train_dataset = ImageFolder(root / "train", transform=train_tfm)
    test_dataset = ImageFolder(root / "test", transform=test_tfm)

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
        dataset_name="cinic10"
    )
