import random

import torch
from torchvision.models import vgg16, vgg11, vgg13
import torch.nn as nn

def CreateVgg16(classes):
    model = changeClassifier(vgg16(), classes)
    return model


def CreateVgg13(classes):
    model = changeClassifier(vgg13(), classes)
    return model


def CreateVgg11(classes):
    model = changeClassifier(vgg11(), classes)
    return model


def changeClassifier(model, classes):
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, classes)
    return model

