import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, List, Dict


class SpecDataset(Dataset):
    """
    初始化频谱数据集，可用于y和x的数据提取，返回的标签事实上不需要

    TODO：
        当前形状固定，实际上acc频谱数据和wav频谱数据成比例关系，
        可通过调整数据集来同时操作两个数据，不需要固定尺寸，
        也可以直接产生合并后的文件，而不需要分别读取再合并。

    参数:
        spec_dir: 频谱图目录路径
        label_dir: 标签文件目录路径
        target_size: 目标尺寸(高度, 宽度)，默认为(300,300)
    """
    def __init__(self, spec_dir: str, label_dir: str, target_size=(240, 240)):
        self.spec_dir = spec_dir
        self.spec_paths: List[str] = []
        self.label_dir = label_dir
        self.labels: Dict[str, str] = {}
        self.target_size = target_size
        self._load_spec_paths()
        self._load_labels()


    def _load_spec_paths(self) -> None:
        for root, _, files in os.walk(self.spec_dir):
            for file in files:
                if file.lower().endswith('.npy'):
                    self.spec_paths.append(os.path.join(root, file))

    def _load_labels(self) -> None:
        """从txt文件中加载标签"""
        for root, _, files in os.walk(self.label_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    label_file_path = os.path.join(root, file)
                    self._parse_label_file(label_file_path)

    def _parse_label_file(self, file_path: str) -> None:
        """解析单个标签文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    filename, label = parts
                    self.labels[filename] = label

    def __len__(self) -> int:
        return len(self.spec_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的样本

        参数:
            index: 样本索引

        返回:
            调整尺寸后的频谱tensor和空标签tensor
        """
        spec_path = self.spec_paths[index]
        spec = np.load(spec_path).astype(np.float32)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, H, W)

        # 如果需要调整尺寸
        if spec_tensor.shape[-2:] != self.target_size:
            _, orig_h, orig_w = spec_tensor.shape
            target_h, target_w = self.target_size

            # 使用双线性插值调整尺寸
            resized_spec = torch.nn.functional.interpolate(
                spec_tensor.unsqueeze(0),  # (1,1,H,W)
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (1,H',W')
        else:
            resized_spec = spec_tensor

        return resized_spec, torch.empty([0])


class GroundTruthDataset(Dataset):
    """
    用于判别器训练时的输入y|x

    特性：
    - 支持多级目录结构的mel/acc频谱图
    - 自动匹配mel/acc频谱图对
    - 从txt文件中加载标签
    - 自动处理图像尺寸不一致问题
    """

    def __init__(self, wav_mel_dir: str, acc_spec_dir: str, label_dir: str):
        """
        初始化数据集。

        参数:
            mel_spec_dir: mel频谱图根目录路径
            acc_spec_dir: acc频谱图根目录路径
            label_dir: 标签文件根目录路径
            opt: 命令行参数对象，包含n_classes等配置
        """
        self.wav_mel_dir = wav_mel_dir
        self.acc_spec_dir = acc_spec_dir
        self.label_dir = label_dir

        # 初始化数据结构
        self.mel_image_paths: List[str] = []
        self.acc_image_paths: List[str] = []
        self.labels: Dict[str, int] = {}

        # 初始化标签嵌入层
        self.label_emb = nn.Embedding(2, 2)

        # 加载所有数据
        self._load_data()

    def _load_data(self) -> None:
        """加载并匹配所有mel/acc频谱图对和标签"""
        # 加载mel频谱图路径
        self.mel_image_paths = self._load_image_paths(self.wav_mel_dir)
        # 加载acc频谱图路径
        self.acc_image_paths = self._load_image_paths(self.acc_spec_dir)

        # 确保mel和acc频谱图数量一致
        min_len = min(len(self.mel_image_paths), len(self.acc_image_paths))
        self.mel_image_paths = self.mel_image_paths[:min_len]
        self.acc_image_paths = self.acc_image_paths[:min_len]

        # 加载标签
        self._load_labels()

    def _load_image_paths(self, root_dir: str) -> List[str]:
        """递归加载指定目录下的所有图像路径"""
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                # if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                #     image_paths.append(os.path.join(root, file))
                if file.lower().endswith('.npy'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _load_labels(self) -> None:
        """从txt文件中加载标签"""
        for root, _, files in os.walk(self.label_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    label_file_path = os.path.join(root, file)
                    self._parse_label_file(label_file_path)

    def _parse_label_file(self, file_path: str) -> None:
        """解析单个标签文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    filename, label = parts
                    self.labels[filename] = int(label)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.mel_image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取指定索引的样本和标签。

        参数:
            idx: 样本索引

        返回:
            拼接后的tensor数据和标签的元组
        """
        # 获取文件路径
        mel_spec_file = self.mel_image_paths[idx]
        acc_spec_file = self.acc_image_paths[idx]

        # 从文件名中提取基础名称(不含扩展名)
        filename = os.path.splitext(os.path.basename(mel_spec_file))[0]
        label = self.labels.get(filename, 0)  # 默认返回空字符串如果找不到

        # # 加载并转换图像
        # mel_spec = Image.open(mel_spec_file).convert('L')
        # acc_spec = Image.open(acc_spec_file).convert('L')
        #
        # # 转换为tensor并归一化
        # mel_tensor = self._image_to_tensor(mel_spec)
        # acc_tensor = self._image_to_tensor(acc_spec)

        mel_tensor = torch.from_numpy(np.load(mel_spec_file)).float() / 255.0
        acc_tensor = torch.from_numpy(np.load(acc_spec_file)).float() / 255.0

        # 调整尺寸使两者一致
        mel_tensor, acc_tensor = self._align_tensor_shapes(mel_tensor, acc_tensor)

        # 拼接两个频谱图
        xy = torch.cat((mel_tensor, acc_tensor), dim=0)

        return xy, label

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为归一化的tensor"""
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # 添加通道维度
        return tensor

    def _align_tensor_shapes(self,
                             mel_tensor: torch.Tensor,
                             acc_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """调整两个tensor的尺寸使其一致"""
        if acc_tensor.shape != mel_tensor.shape:
            # 获取目标尺寸(mel_tensor的尺寸)
            _, h, w = mel_tensor.shape

            # 调整acc_tensor的尺寸
            acc_tensor = torch.nn.functional.interpolate(
                acc_tensor.unsqueeze(0),  # 添加batch维度
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # 移除batch维度

        return mel_tensor, acc_tensor