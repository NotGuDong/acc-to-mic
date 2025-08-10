from torchvision.models import efficientnet_b0
import torch.nn as nn


def CreateEfficientNetb0(classes):
    model = efficientnet_b0()
    in_feature = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_feature, out_features=classes)

    return model

