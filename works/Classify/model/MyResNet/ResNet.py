from torchvision.models import resnet50, resnet18, resnet34
import torch.nn as nn
from model.MyResNet.MyResNet50 import MyResnet50

def CreateResNet50(classes):
    model = __change_fc(resnet50(), classes)
    return model

def CreateMyResNet50(classes):
    model = MyResnet50()
    model.fc = nn.Linear(2048, classes, bias=True)
    return model


def CreateResNet34(classes):
    return __change_fc(resnet34(), classes)


def CreateResNet18(classes):
    return __change_fc(resnet18(), classes)

def __change_fc(model, classes):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, classes)
    return model
