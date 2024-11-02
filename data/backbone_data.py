from enum import Enum, auto


class backbone_data(Enum):
    ResNet = auto()
    MobileNetV3 = auto()
    FasterNet = auto()
    SegFormer = auto()
