from .mlp import MLPClassifier, build_mlp
from .vgg import VGGClassifier, build_vgg
from .resnet import ResNetClassifier, build_resnet


def build_model(name: str, num_classes: int):
    name = name.lower()
    if name.startswith("mlp"):
        return build_mlp(name, num_classes=num_classes)
    if name.startswith("vgg"):
        return build_vgg(name, num_classes=num_classes)
    if name.startswith("resnet"):
        return build_resnet(name, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
