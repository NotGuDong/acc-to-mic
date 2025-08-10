from torchvision.models import convnext_base, convnext_tiny
import torch.nn as nn
from model.MyConvNext.MyConvNext import MyConvNeXt


def CreateConvNext_base(classes):
    model = convnext_base()
    model.classifier[2] = nn.Linear(1024, classes, bias=True)
    return model

def CreateConvNext_tiny(classes):
    model = convnext_tiny()
    model.classifier[2] = nn.Linear(768, classes, bias=True)
    return model

def CreateMyConvNext(classes):
    model = MyConvNeXt()
    model.fc = nn.Linear(768, classes, bias=True)
    return model

