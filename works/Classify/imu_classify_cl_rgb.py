import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import torch.nn.functional as F
import torchaudio.transforms as T
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from torchvision import models, transforms
from PIL import Image
from torchvision.models import DenseNet121_Weights

# ========== 配置 ==========
CSV_DIR = "./acc_data"
RGB_DIR = "./rgb_data"
RGB_DIR_B = "./rgb_data_B"
WAV_DIR = "./wav_data"
LABEL_FILE = "./labels.txt"
KEYWORDS = ["ZERO", "ONE", "THREE", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
SEQ_LEN = 1500
BATCH_SIZE = 16
EPOCHS_1 = 50
EPOCHS_2 = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 数据集定义 ==========
class MultiModalDataset(Dataset):
    def __init__(self, csv_dir, wav_dir, label_file, keywords, seq_len, wav_seq_len=16000, rgb_dir=None):
        self.samples = []
        self.seq_len = seq_len
        self.wav_dir = wav_dir
        self.wav_seq_len = wav_seq_len
        self.rgb_dir = rgb_dir
        self.wav_transform = T.Resample(orig_freq=16000, new_freq=16000)

        # 图像变换
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),     # 适配 DenseNet
            transforms.ToTensor(),             # [0,1] 范围
            transforms.Normalize(              # ImageNet 标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        with open(label_file, 'r') as f:
            for line in f:
                file_id, text = line.strip().split('\t')
                csv_path = os.path.join(csv_dir, f"{file_id}.csv")
                wav_path = os.path.join(wav_dir, f"{file_id}.wav")
                rgb_path = os.path.join(rgb_dir, f"{file_id}.png") if rgb_dir else None

                if not (os.path.isfile(csv_path) and os.path.isfile(wav_path) and os.path.isfile(rgb_path)):
                    continue

                text_upper = text.upper()
                label = int(any(word in text_upper for word in keywords))
                self.samples.append((csv_path, wav_path, rgb_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, wav_path, rgb_path, label = self.samples[idx]

        # -------- 振动数据保留备用（不再使用） --------
        df = pd.read_csv(csv_path, header=None)
        z_data = df.iloc[:, 3].values.astype(np.float32)
        mean, std = np.mean(z_data), np.std(z_data)
        if std < 1e-6:
            std = 1e-6
        z_data = (z_data - mean) / std
        if len(z_data) >= self.seq_len:
            z_data = z_data[:self.seq_len]
        else:
            pad = np.zeros(self.seq_len - len(z_data), dtype=np.float32)
            z_data = np.concatenate([z_data, pad])
        vib_tensor = torch.tensor(z_data).unsqueeze(0)

        # -------- 音频 --------
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            waveform = self.wav_transform(waveform)
        if waveform.shape[1] >= self.wav_seq_len:
            waveform = waveform[:, :self.wav_seq_len]
        else:
            pad_len = self.wav_seq_len - waveform.shape[1]
            pad = torch.cat([waveform, torch.zeros((waveform.shape[0], pad_len))], dim=1)

        # -------- RGB 图像 --------
        img = Image.open(rgb_path).convert("RGB")  # 转为3通道
        img_tensor = self.img_transform(img)       # [3, 224, 224]

        return img_tensor, waveform, torch.tensor(label, dtype=torch.long)

# 定义自定义数据集，传入对应rgb目录和样本列表
class CustomMultiModalDataset(MultiModalDataset):
    def __init__(self, samples, csv_dir, wav_dir, label_file, keywords, seq_len, rgb_dir):
        self.samples = []
        self.seq_len = seq_len
        self.wav_dir = wav_dir
        self.wav_seq_len = 16000
        self.rgb_dir = rgb_dir
        self.wav_transform = T.Resample(orig_freq=16000, new_freq=16000)

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 用传入的samples构造self.samples (路径 + 标签)
        for file_id, text in samples:
            csv_path = os.path.join(csv_dir, f"{file_id}.csv")
            wav_path = os.path.join(wav_dir, f"{file_id}.wav")
            rgb_path = os.path.join(rgb_dir, f"{file_id}.png") if rgb_dir else None

            if not (os.path.isfile(csv_path) and os.path.isfile(wav_path) and os.path.isfile(rgb_path)):
                continue

            text_upper = text.upper()
            label = int(any(word in text_upper for word in keywords))
            self.samples.append((csv_path, wav_path, rgb_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 与原MultiModalDataset一样的__getitem__逻辑
        csv_path, wav_path, rgb_path, label = self.samples[idx]
        # ...（保持原来__getitem__不变）
        # 振动数据
        df = pd.read_csv(csv_path, header=None)
        z_data = df.iloc[:, 3].values.astype(np.float32)
        mean, std = np.mean(z_data), np.std(z_data)
        if std < 1e-6:
            std = 1e-6
        z_data = (z_data - mean) / std
        if len(z_data) >= self.seq_len:
            z_data = z_data[:self.seq_len]
        else:
            pad = np.zeros(self.seq_len - len(z_data), dtype=np.float32)
            z_data = np.concatenate([z_data, pad])
        vib_tensor = torch.tensor(z_data).unsqueeze(0)

        # 音频
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            waveform = self.wav_transform(waveform)
        if waveform.shape[1] >= 16000:
            waveform = waveform[:, :16000]
        else:
            pad_len = 16000 - waveform.shape[1]
            waveform = torch.cat([waveform, torch.zeros((waveform.shape[0], pad_len))], dim=1)

        # RGB 图像
        img = Image.open(rgb_path).convert("RGB")
        img_tensor = self.img_transform(img)

        return img_tensor, waveform, torch.tensor(label, dtype=torch.long)

def get_samples_from_dir(rgb_dir):
    samples = []
    with open(LABEL_FILE, 'r') as f:
        for line in f:
            file_id, text = line.strip().split('\t')

            # 构造该样本对应的rgb路径
            rgb_path = os.path.join(rgb_dir, f"{file_id}.png")

            # 只保留该目录存在的样本
            if os.path.isfile(rgb_path):
                samples.append((file_id, text))
    return samples

# ========== 模型为 Wav2Vec2 + 分类器 ==========
bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec2_model = bundle.get_model()


class VibDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.feature_extractor = densenet.features  # 去掉分类头
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局池化
        self.out_dim = 1024  # DenseNet121 最终通道数

    def forward(self, x):  # 输入 x: [B, 3, H, W]
        x = self.feature_extractor(x)  # [B, 1024, H', W']
        x = self.pool(x)               # [B, 1024, 1, 1]
        x = x.view(x.size(0), -1)      # [B, 1024]
        return x


class MultiModalModel(nn.Module):
    def __init__(self, wav2vec_model, vib_model):
        super().__init__()
        self.wav2vec = wav2vec_model
        self.vib = vib_model

        # 768 (wav2vec2.0) + 1024 (DenseNet121)
        self.fc_fuse = nn.Sequential(
            nn.Linear(1792, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, vib_x, wav_x=None, use_audio=True):
        vib_feat = self.vib(vib_x)  # [B, 1024]

        if use_audio and wav_x is not None:
            if wav_x.dim() == 3:
                wav_x = wav_x.squeeze(1)
            wav_feat = self.wav2vec(wav_x)
            if isinstance(wav_feat, tuple):
                wav_feat = wav_feat[0]
            wav_feat = wav_feat.mean(dim=1)  # [B, 768]
        else:
            wav_feat = torch.zeros((vib_feat.shape[0], 768), device=vib_feat.device)

        fused = torch.cat([vib_feat, wav_feat], dim=1)  # [B, 1792]
        emb = self.fc_fuse(fused)
        logits = self.classifier(emb)
        return emb, logits

# ========== 对比损失函数 ==========
def supervised_contrastive_loss(features, labels, temperature=0.07):
    device = features.device
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    logits = similarity_matrix / temperature
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss

# ========== 训练函数 ==========
def train(model, dataloader, optimizer, criterion_ce, alpha=0.5, use_audio=True):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for vib_x, wav_x, y in dataloader:
        vib_x, y = vib_x.to(DEVICE), y.to(DEVICE)

        # 如果是 use_audio=False，wav_x 可以为空（节省显存）
        if use_audio:
            wav_x = wav_x.to(DEVICE)
        else:
            wav_x = None

        emb, out = model(vib_x, wav_x, use_audio=use_audio)

        loss_ce = criterion_ce(out, y)
        loss_contrast = supervised_contrastive_loss(emb, y)
        loss = (1 - alpha) * loss_ce + alpha * loss_contrast

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

# ========== 验证函数 ==========

def evaluate(model, dataloader, criterion, use_audio=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for vib_x, wav_x, y in dataloader:
            vib_x, y = vib_x.to(DEVICE), y.to(DEVICE)
            wav_x = wav_x.to(DEVICE) if use_audio else None
            _, out = model(vib_x, wav_x, use_audio=use_audio)
            loss = criterion(out, y)

            total_loss += loss.item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels

# ========== 主程序 ==========
def main():

    # ========== 随机取样 ==========
    # # 先读取所有标签（不含图片路径）
    # with open(LABEL_FILE, 'r') as f:
    #     all_lines = f.readlines()
    #
    # # 这里我们只需要file_id和text用于标签和样本过滤
    # all_samples = []
    # for line in all_lines:
    #     file_id, text = line.strip().split('\t')
    #     all_samples.append((file_id, text))
    #
    # # 按标签划分索引（label判断参考你的keywords逻辑）
    # labels = []
    # for file_id, text in all_samples:
    #     text_upper = text.upper()
    #     label = int(any(word in text_upper for word in KEYWORDS))
    #     labels.append(label)
    #
    # # 划分训练集和验证集索引
    # train_idx, val_idx = train_test_split(
    #     range(len(all_samples)),
    #     test_size=0.2,
    #     random_state=42,
    #     stratify=labels
    # )
    #
    # # 创建训练集和验证集的样本列表
    # train_samples = [all_samples[i] for i in train_idx]
    # val_samples = [all_samples[i] for i in val_idx]
    #
    # # 创建训练集和验证集
    # train_set = CustomMultiModalDataset(train_samples, CSV_DIR, WAV_DIR, LABEL_FILE, KEYWORDS, SEQ_LEN, RGB_DIR)
    # val_set = CustomMultiModalDataset(val_samples, CSV_DIR, WAV_DIR, LABEL_FILE, KEYWORDS, SEQ_LEN, RGB_DIR)
    #
    # # 计算训练集样本标签权重，用于加权采样
    # labels_train = [label for _, _, _, label in train_set.samples]
    # class_counts = Counter(labels_train)
    # total = len(labels_train)
    # class_weights = torch.tensor(
    #     [1.0, total / (2.0 * class_counts[1])],
    #     dtype=torch.float32
    # ).to(DEVICE)
    # sample_weights = [1.0 / class_counts[label] for label in labels_train]
    #
    # train_sampler = WeightedRandomSampler(
    #     sample_weights,
    #     num_samples=len(train_set),
    #     replacement=True
    # )
    #
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler)
    # val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # ========== 固定数据集 ==========
    # 训练集样本，从 rgb_data 目录
    train_samples = get_samples_from_dir(RGB_DIR)

    # 验证集样本，从 rgb_data_B 目录
    val_samples = get_samples_from_dir(RGB_DIR_B)

    # 创建两个数据集实例
    train_set = CustomMultiModalDataset(train_samples, CSV_DIR, WAV_DIR, LABEL_FILE, KEYWORDS, SEQ_LEN, RGB_DIR)
    val_set = CustomMultiModalDataset(val_samples, CSV_DIR, WAV_DIR, LABEL_FILE, KEYWORDS, SEQ_LEN, RGB_DIR_B)

    # 计算训练集标签和采样权重
    labels_train = [int(any(word in text.upper() for word in KEYWORDS)) for _, text in train_samples]
    class_counts = Counter(labels_train)
    total = len(labels_train)
    class_weights = torch.tensor(
        [1.0, total / (2.0 * class_counts[1])],
        dtype=torch.float32
    ).to(DEVICE)
    sample_weights = [1.0 / class_counts[label] for label in labels_train]

    train_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_set),
        replacement=True
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # ========== 训练 ==========

    vib_model = VibDenseNet()
    model = MultiModalModel(wav2vec2_model, vib_model).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0

    # ------- 阶段 1：联合训练 -------
    print("阶段 1：联合训练（使用音频和振动）")
    for epoch in range(EPOCHS_1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, alpha=0.3, use_audio=True)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, use_audio=False)
        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated and saved.")

    # ------- 阶段 2：冻结 wav2vec -------
    print("阶段 2：冻结音频模型，仅用振动微调")
    for param in model.wav2vec.parameters():
        param.requires_grad = False

    for epoch in range(EPOCHS_2):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, alpha=0.3, use_audio=False)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, use_audio=False)
        print(f"[Fine-tune Epoch {epoch + 1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated and saved (fine-tune).")

    print("训练结束，最优模型已保存为 best_model.pth")

# ========== 评估脚本 ==========
def evaluate_model(model_path):
    import pickle

    # 加载数据集和 val_idx
    dataset = MultiModalDataset(CSV_DIR, WAV_DIR, LABEL_FILE, KEYWORDS, SEQ_LEN, rgb_dir=RGB_DIR_B)
    with open("val_idx.pkl", "rb") as f:
        val_idx = pickle.load(f)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    vib_model = VibDenseNet()
    model = MultiModalModel(wav2vec2_model, vib_model).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    loader = DataLoader(val_set, batch_size=1)
    y_true = []
    y_pred = []

    with torch.no_grad():
        for vib_x, _, y in loader:
            vib_x, y = vib_x.to(DEVICE), y.to(DEVICE)
            _, out = model(vib_x, wav_x=None, use_audio=False)
            pred = out.argmax(1).item()
            y_pred.append(pred)
            y_true.append(y.item())

    print("Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["Non-Keyword", "Keyword"],
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Keyword", "Keyword"], yticklabels=["Non-Keyword", "Keyword"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
    evaluate_model("best_model.pth")