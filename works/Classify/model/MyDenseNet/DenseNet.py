from torchvision.models import densenet121
import torch.nn as nn


def CreateDenseNet121(classes):
    model = densenet121()
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, classes, bias=True)
    return model

def CreateDenseNet121ForRegression():
    model = densenet121()
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 1)  # 输出单个值作为体重
    return model