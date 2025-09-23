# README.md

## 项目介绍

本项目为论文AccEar_Accelerometer_Acoustic_Eavesdropping_with_Unconstrained_Vocabulary的复现


## 文件介绍

- dataset.zip：需要解压到 ./dataset/ 目录下
- dataset/：用于模型训练的数据集
    - csv_to_spec_npy：处理过后的acc的z轴数据文件的stft的Zxx的绝对值
    - wav_to_mel_npy：处理过后的wav文件的梅尔频谱的stft的Zxx的绝对值
    - train-clean-100：原始wav文件
    - patch000_LibriSpeech：原始acc的数据文件
- output/：存储训练过程中生成器和判别器的参数，存储测试生成器时得到的图片数据
- tools/：涉及到的python文件，unet.py没有用到，仅作参考
- acc_ear.py：训练与测试代码
- preprocess.py：处理原始wav文件和原始acc文件的代码，目前已经处理过

## 代码介绍

```python
# acc_ear.py
# 参数定义，可修改

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
```

**任务介绍**

*Task* 1：模型训练函数
    - 相关函数：`train()`
    - 使用方法：`python acc_ear.py`

*Task* 2：单独测试生成器
    - 相关函数：`test_generator()`
    - 使用方法：`python acc_ear.py --test`


## 优化方式

1. 处理过后的ndarray数据长宽不同，理论上acc数据和wav数据对应的ndarray长宽有一定比例关系，目前读取时统一插值到240*240的形状。可以优化，不损失原有信息，仅插值数据较少的acc数据。
2. 理论上处理后的acc数据文件和wav数据文件是一一对应的关系，代码中使用两个数据集对象存储不同的数据文件，使用dataloader包装时不进行打乱，以此来实现对应关系。可以优化数据集类，用一个数据集处理两个数据文件的关系。
3. 训练时参考论文，将输入的图片按照30*30的批次进行判别，用结果的平均值代表图片判断结果。目前的处理方法效率较低，可以进一步改进，参考tools/Discriminator.py。
4. 目前训练使用的损失函数为MSE，均方根误差，和原论文不完全相似。可以进一步设计损失函数

## 参考网址

- 博客 [使用 Hugging Face Transformers 对英文 ASR 进行 Wav2Vec2 微调](https://hugging-face.cn/blog/fine-tune-wav2vec2-english)
- 使用的预训练模型为 [Hugging Face](https://huggingface.co/) 官网的 [*facebook/wav2vec2-base*](https://huggingface.co/models) 模型，需要翻墙，windows下模型会下载到c盘的用户文件夹下
