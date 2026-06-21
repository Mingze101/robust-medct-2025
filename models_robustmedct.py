# models_robustmedct.py
import torch
from torch import nn
from torchvision import models

IMG_SIZE = 224  # 保持和训练脚本一致

def build_resnet18(num_classes, use_pretrained=False):
    if use_pretrained:
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
    else:
        model = models.resnet18(weights=None)

    # 1-channel 输入
    w = model.conv1.weight.data
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    model.conv1.weight.data = w.mean(dim=1, keepdim=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_resnet34(num_classes, use_pretrained=False):
    if use_pretrained:
        model = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        )
    else:
        model = models.resnet34(weights=None)

    w = model.conv1.weight.data
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    model.conv1.weight.data = w.mean(dim=1, keepdim=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
