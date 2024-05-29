import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from transformers import Wav2Vec2Model
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset

# 定义设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# 数据预处理
class AudioPreprocessor:
    def __init__(self, model_type='wav2vec2', max_length=16000):
        if model_type == 'wav2vec2':
            self.transform = nn.Sequential(
                MelSpectrogram(sample_rate=16000, n_mels=64),
                AmplitudeToDB()
            )
        self.model_type = model_type
        self.max_length = max_length

    def __call__(self, audio):
        print("Original audio shape:", audio.shape)
        if self.model_type == 'wav2vec2':
            audio = self.transform(audio)
        print("Transformed audio shape:", audio.shape)
        if audio.shape[2] > self.max_length:
            audio = audio[:, :, :self.max_length]
        else:
            padding = self.max_length - audio.shape[2]
            audio = nn.functional.pad(audio, (0, padding))
        audio = audio.squeeze(0)  # 移除单一的批处理维度，确保形状为 [feature_size, time_steps]
        print("Padded/Trimmed audio shape:", audio.shape)
        return audio


def pad_sequence(batch):
    audios = [item[0] for item in batch]
    emotion_labels = torch.tensor([item[1] for item in batch])
    dimension_labels = torch.stack([item[2] for item in batch])

    # 获取所有音频样本的最大长度
    max_length = max(audio.shape[1] for audio in audios)
    print("Max length in batch:", max_length)
    padded_audios = [nn.functional.pad(audio, (0, max_length - audio.shape[1])) for audio in audios]
    padded_audios = torch.stack(padded_audios)

    # 确保输出为 [batch_size, feature_size, time_steps]
    print("Padded audios shape:", padded_audios.shape)
    print("Emotion labels shape:", emotion_labels.shape)
    print("Dimension labels shape:", dimension_labels.shape)

    return padded_audios, emotion_labels, dimension_labels


# 多视角学习模型
class MultiViewModel(nn.Module):
    def __init__(self, model_type='wav2vec2'):
        super(MultiViewModel, self).__init__()
        if model_type == 'wav2vec2':
            self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
        )
        self.emotion_head = nn.Linear(256, 5)
        self.dimensions_head = nn.Linear(256, 3)


    def forward(self, audio):
        print("Input audio shape:", audio.shape)
        x = self.encoder(audio).last_hidden_state
        print("Encoded audio shape:", x.shape)
        x = self.projector(x)
        print("Projected audio shape:", x.shape)
        emotions = self.emotion_head(x)
        dimensions = self.dimensions_head(x)
        print("Emotions shape:", emotions.shape)
        print("Dimensions shape:", dimensions.shape)
        return emotions, dimensions


# 训练和评估
def train_and_evaluate(model, train_loader, val_loader, epochs=20, lr=1e-4):
    criterion_classification = nn.CrossEntropyLoss()
    criterion_regression = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            audio, emotion_labels, dimension_labels = batch
            audio, emotion_labels, dimension_labels = audio.to(device), emotion_labels.to(device), dimension_labels.to(
                device)
            optimizer.zero_grad()
            emotions, dimensions = model(audio)
            loss = (
                    criterion_classification(emotions, emotion_labels) +
                    criterion_regression(dimensions, dimension_labels)
            )
            loss.backward()
            optimizer.step()

        # 验证模型
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in val_loader:
                audio, emotion_labels, dimension_labels = batch
                audio, emotion_labels, dimension_labels = audio.to(device), emotion_labels.to(
                    device), dimension_labels.to(device)
                emotions, dimensions = model(audio)
                loss = (
                        criterion_classification(emotions, emotion_labels) +
                        criterion_regression(dimensions, dimension_labels)
                )
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Validation Loss: {total_loss / len(val_loader)}")


# 数据集类
class EmotionDataset(Dataset):
    def __init__(self, csv_file, audio_dir, model_type='wav2vec2', max_length=16000):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.preprocessor = AudioPreprocessor(model_type=model_type, max_length=max_length)

        # 将标签转换为数值表示
        self.label_to_idx = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'fea': 4, 'dis': 5, 'sur': 6, 'exc': 7, 'fru': 8}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 1] + '.wav')
        audio, sample_rate = torchaudio.load(audio_path)
        print("Loaded audio shape:", audio.shape)
        audio = self.preprocessor(audio)

        # 提取标签
        emotion_label = self.label_to_idx[self.data.iloc[idx, 2]]
        valence = torch.tensor(self.data.iloc[idx, 3], dtype=torch.float32)
        arousal = torch.tensor(self.data.iloc[idx, 4], dtype=torch.float32)
        dominance = torch.tensor(self.data.iloc[idx, 5], dtype=torch.float32)

        print("Final audio shape:", audio.shape)
        print("Emotion label:", emotion_label)
        print("Valence, Arousal, Dominance:", valence, arousal, dominance)

        return audio, emotion_label, torch.tensor([valence, arousal, dominance])


# 示例数据
csv_file = "data.csv"
audio_dir = "./data/iemocap"
max_length = 16000  # 根据你的音频长度进行调整
train_dataset = EmotionDataset(csv_file, audio_dir, max_length=max_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_sequence)
val_dataset = EmotionDataset(csv_file, audio_dir, max_length=max_length)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_sequence)

# 训练模型
model = MultiViewModel(model_type='wav2vec2').to(device)
train_and_evaluate(model, train_loader, val_loader)
