import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 添加tqdm导入

from tools.Dataset import SpecDataset
from tools.Discriminator import Discriminator
from tools.Generator import Generator

'''
参考博客：
    cGAN：https://blog.csdn.net/CXDNW/article/details/139845947
    GAN: https://zhuanlan.zhihu.com/p/628915533
'''

ACC_SPEC_DIR = ".\\dataset\\csv_to_spec_npy\\"
WAV_MEL_DIR = ".\\dataset\\wav_to_mel_npy\\"
LABEL_DIR = ".\\dataset\\train-clean-100\\"
D_PTH_SAVE_PATH = ".\\output\\discriminator.pth"
G_PTH_SAVE_PATH = ".\\output\\generator.pth"
SAVE_FREQUENCY = 1
SAMPLE_IMG_SAVE_PATH = ".\\output\\sample_img.png"
N_EPOCHS = 100
BATCH_SIZE = 4
LR = 0.0002
N_CLASSES = 2
IMG_SIZE = 30
INPUT_SIZE = (240, 240)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_data_loader():
    y_dataset = SpecDataset(ACC_SPEC_DIR, LABEL_DIR)
    x_dataset = SpecDataset(WAV_MEL_DIR, LABEL_DIR)

    # 确保两个数据集样本数相同
    min_len = min(len(y_dataset), len(x_dataset))
    y_dataset = torch.utils.data.Subset(y_dataset, range(min_len))
    x_dataset = torch.utils.data.Subset(x_dataset, range(min_len))

    y_dataset_loader = DataLoader(y_dataset, batch_size=BATCH_SIZE, shuffle=False)
    x_dataset_loader = DataLoader(x_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return y_dataset_loader, x_dataset_loader


def initialize_models():
    generator = Generator()
    discriminator = Discriminator(input_size=INPUT_SIZE)
    return generator, discriminator


def initialize_optimizers(generator, discriminator):
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR)
    return optimizer_g, optimizer_d


def extract_patches(images, patch_size=20):
    """提取所有可能的20x20 patch"""
    patches = []
    b, c, h, w = images.shape

    # 计算所有可能的起始位置
    h_indices = list(range(0, h - patch_size + 1, patch_size))
    if h % patch_size != 0:
        h_indices.append(h - patch_size)
    w_indices = list(range(0, w - patch_size + 1, patch_size))
    if w % patch_size != 0:
        w_indices.append(w - patch_size)

    # 提取所有patch
    for i in h_indices:
        for j in w_indices:
            patch = images[:, :, i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    # 合并所有patch
    if patches:
        return torch.cat(patches, dim=0)
    return None


def get_combine_spec(y_batch, x_batch):
    y_batch = y_batch.to(device)
    x_batch = x_batch.to(device)
    combine_spec = torch.cat([y_batch, x_batch], dim=1)
    return combine_spec


def train(x_data_loader, y_data_loader, generator, discriminator, optimizer_g, optimizer_d, adversarial_loss):
    d_loss_ = []
    g_loss_ = []

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # 创建tqdm进度条包装epoch循环
    epoch_bar = tqdm(range(N_EPOCHS), desc="Training Progress", unit="epoch")

    for epoch in epoch_bar:
        d_epoch_loss = 0
        g_epoch_loss = 0
        count = len(y_data_loader)

        for batch_idx, ((y_batch, _), (x_batch, _)) in enumerate(zip(y_data_loader, x_data_loader)):
            # y为z轴加速度数据的stft图，作为condition条件
            # x为真实录音环境下的mel频谱图
            # z为噪声
            # 形状应为（batch_size, channel, height, width）
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)
            z_batch = torch.randn(y_batch.shape).to(device)

            # ---------------------
            #  训练判别器
            # ---------------------
            optimizer_d.zero_grad()

            true_spec = get_combine_spec(y_batch, x_batch)
            real_predictions = discriminator(true_spec)
            real_targets = torch.ones(real_predictions.shape[0]).to(device)
            d_real_loss = adversarial_loss(real_predictions, real_targets)
            d_real_loss.backward()

            noise = get_combine_spec(y_batch, z_batch)
            gen_img = generator(noise)
            fake_spec = get_combine_spec(y_batch, gen_img)
            fake_predictions = discriminator(fake_spec.detach())
            fake_targets = torch.zeros(fake_predictions.shape[0]).to(device)
            d_fake_loss = adversarial_loss(fake_predictions, fake_targets)
            d_fake_loss.backward()

            optimizer_d.step()

            # ---------------------
            #  训练生成器
            # ---------------------
            optimizer_g.zero_grad()

            gen_spec_predictions = discriminator(fake_spec)
            fake_targets = torch.ones(gen_spec_predictions.shape[0]).to(device)
            g_loss = adversarial_loss(gen_spec_predictions, fake_targets)
            g_loss.backward()
            optimizer_g.step()

            # 累计损失
            d_loss = d_real_loss + d_fake_loss
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

        # 计算平均损失
        d_epoch_loss /= count
        g_epoch_loss /= count
        d_loss_.append(d_epoch_loss)
        g_loss_.append(g_epoch_loss)

        if (epoch + 1) % SAVE_FREQUENCY == 0:
            print("Saving model...")
            torch.save(generator.state_dict(), G_PTH_SAVE_PATH)
            torch.save(discriminator.state_dict(), D_PTH_SAVE_PATH)

        # 更新epoch进度条描述
        epoch_bar.set_postfix({
            "Epoch": f"{epoch + 1}/{N_EPOCHS}",
            "Avg D_loss": f"{d_epoch_loss:.4f}",
            "Avg G_loss": f"{g_epoch_loss:.4f}"
        })
        epoch_bar.write(
            f"[Epoch {epoch + 1}/{N_EPOCHS}] "
            f"Discriminator Loss: {d_epoch_loss:.4f} | "
            f"Generator Loss: {g_epoch_loss:.4f}"
        )

    return d_loss_, g_loss_


def test_generator(generator, condition_img, save_path=None):
    """更健壮的测试函数实现"""
    # 输入预处理
    if not isinstance(condition_img, torch.Tensor):
        condition_img = torch.from_numpy(condition_img).float()

    # 统一为4D张量(batch, channel, height, width)
    if condition_img.dim() == 2:
        condition_img = condition_img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif condition_img.dim() == 3:
        condition_img = condition_img.unsqueeze(0)  # 添加batch维度

    # 生成随机噪声
    b, c, h, w = condition_img.shape
    noise = torch.randn(b, 1, h, w).to(device)

    # 组合输入
    # input_tensor = torch.cat([noise, condition_img], dim=1)  # (b,2,H,W)
    input_tensor = get_combine_spec(condition_img, noise)

    # 生成
    generator.eval()
    with torch.no_grad():
        output = generator(input_tensor)

    # 输出后处理
    assert output.dim() == 4
    assert output.shape[1] == 1
    img_data = output.squeeze(0).squeeze(0).cpu().numpy()

    if save_path is not None:
        img_data = np.clip(img_data, 0, 1)
        plt.imsave(save_path, img_data, cmap='viridis')
    return img_data


def plot_losses(d_loss_, g_loss_):
    x = [epoch + 1 for epoch in range(len(d_loss_))]
    plt.figure()
    plt.plot(x, g_loss_, 'r')
    plt.plot(x, d_loss_, 'b')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['G_loss', 'D_loss'])
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    args = parser.parse_args()

    if args.test:
        print("\n=== 运行生成器测试 ===")
        # 这里需要实际加载预训练模型
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(G_PTH_SAVE_PATH))

        # 创建示例输入(实际使用时替换为真实数据)
        example_input = np.load(".\\dataset\\csv_to_spec_npy\\103\\1240_001\\103-1240-0000.npy")

        # 运行测试
        output = test_generator(generator, example_input, SAMPLE_IMG_SAVE_PATH)
        print(f"测试完成，输出形状: {output.shape}\n图像已保存至: {SAMPLE_IMG_SAVE_PATH}")

    else:
        print("=== 开始加载数据 ===")
        y_data_loader, x_data_loader = build_data_loader()
        print(f"数据加载完成，共加载 {len(y_data_loader)} 个样本")

        # 添加模型初始化提示
        print("\n=== 初始化模型 ===")
        generator, discriminator = initialize_models()
        print("生成器(Generator)和判别器(Discriminator)初始化完成")

        # 添加优化器初始化提示
        optimizer_g, optimizer_d = initialize_optimizers(generator, discriminator)
        print("优化器初始化完成 (学习率: {})".format(LR))

        # 添加损失函数提示
        print("\n=== 初始化损失函数 ===")
        adversarial_loss = torch.nn.MSELoss()
        print("使用MSE损失函数进行训练")

        # 添加训练开始提示
        print("\n=== 开始训练 ===")
        print(f"训练参数: Epochs={N_EPOCHS}, Batch Size={BATCH_SIZE}, 设备={device}")

        # 训练过程
        d_loss_, g_loss_ = train(
            x_data_loader, y_data_loader, generator, discriminator,
            optimizer_g, optimizer_d, adversarial_loss
        )

        # 添加训练结束提示
        print("\n=== 训练完成 ===")
        print(f"总训练轮次: {N_EPOCHS}")
        print(f"最终损失 - 判别器: {d_loss_[-1]:.4f}, 生成器: {g_loss_[-1]:.4f}")

        # 添加绘图提示
        print("\n=== 绘制损失曲线 ===")
        plot_losses(d_loss_, g_loss_)
        print("损失曲线已保存/显示")


if __name__ == "__main__":
    main()
