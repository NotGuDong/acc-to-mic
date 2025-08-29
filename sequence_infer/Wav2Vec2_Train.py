import json
import os
import numpy as np
import pandas as pd
import jiwer
import librosa
import torch
import torchaudio
import torch.utils.data
from dataclasses import dataclass
from typing import Optional, List, Dict, Union
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample
from torchaudio.transforms import Resample
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, \
    Wav2Vec2Processor, TrainingArguments, Wav2Vec2ForCTC, Trainer

DATASET_FOLDER = '.\\dataset\\'
SINGLE_TEST_WAV_PATH = ".\\dataset\\test\\1040\\133433_001\\1040-133433-0000.wav"
TRAINING_SAVE_PATH = '.\\output'
MODEL_SAVE_PATH = '.\\output\\model'
PROCESSER_SAVE_PATH = '.\\output\\processor'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ORIGINAL_SAMPLE_RATE = 417
SAMPLE_RATE = 400


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, tokenizer):
        super().__init__()
        self.data_folder = data_folder
        self.values = []
        self.tokenizer = tokenizer
        self.audio_paths = []
        self.transcripts = []
        self.resampler = Resample(orig_freq=SAMPLE_RATE, new_freq=16000)
        self.load_data()

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        return self.values[index]
        # waveform, sr = torchaudio.load(self.audio_paths[index])
        # if sr != 16000:
        #     waveform = self.resampler(waveform)
        # transcript = self.transcripts[index].upper()
        # tokens = self.tokenizer(transcript)
        #
        # tokens['input_ids'] = torch.tensor(tokens['input_ids'])
        # tokens['attention_mask'] = torch.tensor(tokens['attention_mask'])
        #
        # input_values = waveform.squeeze()
        # attention_mask = tokens['attention_mask'].squeeze()
        # labels = tokens['input_ids']
        # return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}

    def load_data(self):
        patch1_folders = os.listdir(self.data_folder)
        for patch1_folder in patch1_folders:
            patch2_folders = os.listdir(os.path.join(self.data_folder, patch1_folder))
            for patch2_folder in patch2_folders:
                label_file = patch1_folder + '-' + patch2_folder.split('_')[0] + '.trans.txt'
                transcripts_map = {}
                with open(label_file, "r", encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(" ", 1)
                        file_name = parts[0]
                        label = parts[1]
                        label = label.replace(' ', '|') + '|'
                        transcripts_map[file_name] = label

                data_files = os.listdir(os.path.join(self.data_folder, patch1_folder, patch2_folder))
                for data_file in data_files:
                    if not data_file.endswith('.wav'):
                        continue

                    file_path = os.path.join(self.data_folder, patch1_folder, patch2_folder, data_file)
                    transcript = transcripts_map[data_file.split('.', maxsplit=1)[0]].upper()
                    # self.audio_paths.append(file_path)
                    # self.transcripts.append(transcript)

                    waveform, sr = torchaudio.load(file_path)
                    if sr != 16000:
                        waveform = self.resampler(waveform)

                    tokens = self.tokenizer(transcript)

                    tokens['input_ids'] = torch.tensor(tokens['input_ids'])
                    tokens['attention_mask'] = torch.tensor(tokens['attention_mask'])

                    input_values = waveform.squeeze()
                    attention_mask = tokens['attention_mask'].squeeze()
                    labels = tokens['input_ids']
                    self.values.append({"input_values": input_values, "attention_mask": attention_mask, "labels": labels})

        print(f"Loaded {len(self.values)} audio files")


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def single_infer():
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_SAVE_PATH).to(DEVICE)
    processor = Wav2Vec2Processor.from_pretrained(PROCESSER_SAVE_PATH)
    model.eval()

    waveform, sr = torchaudio.load(SINGLE_TEST_WAV_PATH)
    if sr != 16000:
        waveform = Resample(orig_freq=SAMPLE_RATE, new_freq=16000)(waveform)
    waveform = waveform.squeeze()

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding="longest").to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast():
        logits = model(**inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)
    print(processor.decode(pred_ids[0]))


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {DEVICE}")

    # 1. 准备Tokenizer
    labels = ['-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P',
              'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z', '[UNK]', '[PAD]']
    vocab_dict = {v: k for k, v in enumerate(labels)}
    print(f"vocab_list: {labels}\nvocab_dict size: {len(vocab_dict)}")

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')

    # 2. 准备特征提取器
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )

    # 3. 组合分词器
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        # wer = wer_metric.compute(predictions=pred_str, references=label_str)
        wer = jiwer.wer(pred_str, label_str)

        return {"wer": wer}

    # 4. 准备数据整理器
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # 5. 准备评估指标
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir=TRAINING_SAVE_PATH,
        group_by_length=True,
        per_device_train_batch_size=4,
        # evaluation_strategy="steps",
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

    train_dataset = AudioDataset(data_folder=os.path.join(DATASET_FOLDER, 'train'),
                                 tokenizer=tokenizer)
    test_dataset = AudioDataset(data_folder=os.path.join(DATASET_FOLDER, 'test'),
                                tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    results = trainer.evaluate()
    print("评估结果：")
    for key, value in results.items():
        print(f"{key}: {value}")

    processor.save_pretrained("./output/processor")
    model.save_pretrained("./output/model")


def csv2wav_recursive(base_folder, input_folder, output_folder):
    full_input_folder = os.path.join(base_folder, input_folder)
    full_output_folder = os.path.join(base_folder, output_folder)
    for root, dirs, files in os.walk(full_input_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                relative_path = root[len(full_input_folder) + 1:]
                wav_path = os.path.join(full_output_folder, relative_path, file.replace(".csv", ".wav"))

                df = pd.read_csv(csv_path)
                signal_np = df.iloc[:, 3].values

                new_num_samples = round(len(signal_np) * (SAMPLE_RATE / ORIGINAL_SAMPLE_RATE))
                resampled_signal = resample(signal_np, new_num_samples)

                min_value = np.min(resampled_signal)
                max_value = np.max(resampled_signal)
                scaled_signal_np = (resampled_signal - min_value) * (32767 - (-32768)) / (max_value - min_value) + (
                    -32768)
                scaled_signal_np = scaled_signal_np.astype(np.int16)

                os.makedirs(os.path.join(full_output_folder, relative_path), exist_ok=True)
                wavfile.write(wav_path, SAMPLE_RATE, scaled_signal_np)


def show_wav(wav_path):
    # 加载音频文件
    y, sr = librosa.load(wav_path)

    # 生成波形图
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')

    # 生成频谱图
    plt.subplot(1, 2, 2)
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    plt.tight_layout()
    fig.show()


if __name__ == "__main__":
    # Task 1：模型训练函数
    main()

    # Task 2：单独推断某一个句子
    # single_infer()

    # Task 3：将原始传感器数据递归转换为wav文件
    # csv2wav_recursive(
    #     base_folder='.',
    #     input_folder='patch000_LibriSpeech\\train-clean\\train-clean-100',
    #     output_folder='dataset\\all'
    # )

    # Task 4：展示某一个音频文件的频谱图
    # show_wav('.\\dataset\\all\\103\\1240_001\\103-1240-0000.wav')
    # plt.show()

