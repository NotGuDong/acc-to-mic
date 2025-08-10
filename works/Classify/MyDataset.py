"""
创建数据集，适用于一般分类，文件结构 data/label_name/img,每个种类单独存放一个文件夹
"""
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import os
from PIL import Image

# 语音标签
labels = {"America": 0, "answer": 1, "China": 2, "goodbye": 3, "hello": 4, "Mike": 5, "potato": 6}

train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class myDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.data_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.data_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        if self.transform is not None:
            img = self.transform(img)
        label = labels[self.label_dir]
        return img, label


def creatDataset(data_dir):
    label_dirs = os.listdir(data_dir)
    for i in range(len(label_dirs)):
        if i==0:
            MyDataset = myDataset(data_dir, label_dirs[i], train_transform)
        else:
            MyDataset += myDataset(data_dir, label_dirs[i], train_transform)
    return MyDataset

class WeightRegressionDataset(Dataset):
    def __init__(self, data_dir, transform=train_transform):
        self.data = []
        self.transform = transform

        for weight_folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, weight_folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                weight = float(weight_folder)
            except ValueError:
                continue

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.data.append((img_path, weight))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, weight = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, weight