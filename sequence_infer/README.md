# README.md

## 项目介绍

本项目使用hugging face的Wav2Vec2模型进行序列识别任务，理想情况下，可以识别任何句子

- 输入：使用400Hz手机收集到的csv文件
- 输出：31个字符拼接而成的英文句子
- 31个标签字符：['-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z', '[UNK]', '[PAD]']

其中，"|"表示单词结束

## 文件介绍

- dataset/：用于模型训练的数据集，all文件夹包含所有数据
- output/：代码需要的文件和输出文件
- patch000_LibriSpeech/：手机收集到的原始文件，已经和all文件夹同步
- Wav2Vec2_Train.py：训练与处理代码

## 代码介绍

```python
# 宏定义
DATASET_FOLDER = '.\\dataset\\'
SINGLE_TEST_WAV_PATH = ".\\dataset\\test\\1040\\133433_001\\1040-133433-0000.wav"
TRAINING_SAVE_PATH = '.\\output'
MODEL_SAVE_PATH = '.\\output\\model'
PROCESSER_SAVE_PATH = '.\\output\\processor'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ORIGINAL_SAMPLE_RATE = 417
SAMPLE_RATE = 400
```

```python
# 训练参数，根据配置自行选择
    training_args = TrainingArguments(
        output_dir='./data/audios/output',
        group_by_length=True,
        per_device_train_batch_size=4,
        num_train_epochs=5,
        fp16=True,
        gradient_checkpointing=False,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        gradient_accumulation_steps=2,
    )
```

**任务介绍**

*Task* 1：模型训练函数

    main()

*Task* 2：单独推断某一个句子

    single_infer()

*Task* 3：将原始传感器数据递归转换为wav文件

    csv2wav_recursive(
        base_folder='.',
        input_folder='patch000_LibriSpeech\\train-clean\\train-clean-100',
        output_folder='dataset\\all'
    )

*Task* 4：展示某一个音频文件的频谱图

    show_wav('.\\dataset\\all\\103\\1240_001\\103-1240-0000.wav')
    plt.show()

## 使用方法

1. 安装下载torch, torchvision, transformer, librosa, jiwer等一系列包

2. 在dataset中，将适量的all中的数据放到train和test文件夹中

3. 在 *Wav2Vec2_Train.py* 中, 在最底部的 `if __name__ == "__main__":` 块处，先只留`main()`函数，运行，进行模型训练

4. 模型训练存在训练慢等问题时查看模型训练参数，进行修改

5. 训练完成后，查看控制台输出的评估分数，其中，**wer** 表示词错误率，越小越好

6. 在 *Wav2Vec2_Train.py* 中, 在最底部的 `if __name__ == "__main__":` 块处，只留`single_infer()`函数，运行，进行模型在单个csv文件上的测试

7. 运行完成后，查看控制台输出的字符串序列，是否和理论标签一致

## 参考网址

- 博客 [使用 Hugging Face Transformers 对英文 ASR 进行 Wav2Vec2 微调](https://hugging-face.cn/blog/fine-tune-wav2vec2-english)
- 使用的预训练模型为 [Hugging Face](https://huggingface.co/) 官网的 [*facebook/wav2vec2-base*](https://huggingface.co/models) 模型，需要翻墙，windows下模型会下载到c盘的用户文件夹下
