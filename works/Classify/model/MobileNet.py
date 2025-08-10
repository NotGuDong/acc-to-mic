from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
import torch.nn as nn


def changeClassifier(model, classes):
    in_feature = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feature, classes)
    return model


def CreateMobileNet(classes):
    model = changeClassifier(mobilenet_v3_small(), classes)
    return model



