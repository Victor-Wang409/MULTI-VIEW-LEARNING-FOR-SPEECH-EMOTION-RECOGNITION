import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import Wav2Vec2Model
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 数据预处理函数
def preprocess_audio(audio_path, max_length=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    else:
        pad_amount = max_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    return waveform.squeeze()



# 自定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, csv_file, audio_dir, max_length=16000):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.label_map = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'exc': 4}  # 根据实际标签进行调整

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = f"{self.audio_dir}/{self.data.iloc[idx, 1]}.wav"
        label_emotion = self.label_map[self.data.iloc[idx, 2]]
        label_sentiment = 1 if label_emotion in [1, 4] else (0 if label_emotion == 0 else 2)  # 简单假设：快乐和兴奋为积极，其他为消极，中性为中立
        valence = self.data.iloc[idx, 3]
        arousal = self.data.iloc[idx, 4]
        dominance = self.data.iloc[idx, 5]

        audio = preprocess_audio(audio_path, self.max_length)
        labels_emotion = torch.tensor(label_emotion, dtype=torch.long)
        labels_sentiment = torch.tensor(label_sentiment, dtype=torch.long)
        labels_dimension = torch.tensor([valence, arousal, dominance], dtype=torch.float)

        return audio, labels_emotion, labels_sentiment, labels_dimension

# 加载数据集
train_dataset = EmotionDataset(csv_file='./data.csv', audio_dir='./iemocap')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 模拟验证和测试集加载
valid_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

# 定义模型
class MultiViewModel(nn.Module):
    def __init__(self, audio_encoder, hidden_dim):
        super(MultiViewModel, self).__init__()
        self.audio_encoder = audio_encoder
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.emotion_head = nn.Linear(hidden_dim, 5)  # 假设有5种情感分类
        self.sentiment_head = nn.Linear(hidden_dim, 3)  # 3种情感评分
        self.dimension_head = nn.Linear(hidden_dim, 3)  # Valence, Arousal, Dominance

    def forward(self, x):
        x = self.audio_encoder(x).last_hidden_state
        x = self.projection(x[:, 0, :])
        x = self.relu(x)
        emotion_output = self.emotion_head(x)
        sentiment_output = self.sentiment_head(x)
        dimension_output = self.dimension_head(x)
        return emotion_output, sentiment_output, dimension_output

# 加载预训练的Wav2Vec2模型并冻结其参数
wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
for param in wav2vec2.parameters():
    param.requires_grad = False

# 实例化模型
model = MultiViewModel(wav2vec2, hidden_dim=768)

# 定义损失函数和优化器
def ccc_loss(pred, target):
    mean_pred = torch.mean(pred)
    mean_target = torch.mean(target)
    covariance = torch.mean((pred - mean_pred) * (target - mean_target))
    var_pred = torch.mean((pred - mean_pred) ** 2)
    var_target = torch.mean((target - mean_target) ** 2)
    ccc = (2 * covariance) / (var_pred + var_target + (mean_pred - mean_target) ** 2)
    return 1 - ccc  # 1 - ccc 为了使其成为一个可以优化的损失

def CCC(pred, target):
    mean_pred = torch.mean(pred, dim=0)
    mean_target = torch.mean(target, dim=0)
    covariance = torch.mean((pred - mean_pred) * (target - mean_target), dim=0)
    var_pred = torch.mean((pred - mean_pred) ** 2, dim=0)
    var_target = torch.mean((target - mean_target) ** 2, dim=0)
    ccc = (2 * covariance) / (var_pred + var_target + (mean_pred - mean_target) ** 2)
    return ccc  # 返回平均损失和每个维度的CCC值

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练
def train(model, train_loader, valid_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                inputs, labels_emotion, labels_sentiment, labels_dimension = batch
                inputs = inputs.to(device)
                labels_emotion = labels_emotion.to(device)
                labels_sentiment = labels_sentiment.to(device)
                labels_dimension = labels_dimension.to(device)

                optimizer.zero_grad()
                output_emotion, output_sentiment, output_dimension = model(inputs)
                loss_emotion = nn.CrossEntropyLoss()(output_emotion, labels_emotion)
                loss_sentiment = nn.CrossEntropyLoss()(output_sentiment, labels_sentiment)
                loss_dimension = ccc_loss(output_dimension, labels_dimension)
                loss = loss_emotion + loss_sentiment + loss_dimension
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss/len(train_loader))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

        model.eval()
        valid_loss = 0
        total_ccc = torch.zeros(3, device=device)
        with tqdm(valid_loader, unit="batch") as vepoch:
            for batch in vepoch:
                inputs, labels_emotion, labels_sentiment, labels_dimension = batch
                inputs = inputs.to(device)
                labels_emotion = labels_emotion.to(device)
                labels_sentiment = labels_sentiment.to(device)
                labels_dimension = labels_dimension.to(device)

                output_emotion, output_sentiment, output_dimension = model(inputs)
                loss_emotion = nn.CrossEntropyLoss()(output_emotion, labels_emotion)
                loss_sentiment = nn.CrossEntropyLoss()(output_sentiment, labels_sentiment)
                loss_dimension = ccc_loss(output_dimension, labels_dimension)
                ccc = CCC(output_dimension, labels_dimension)
                loss = loss_emotion + loss_sentiment + loss_dimension
                valid_loss += loss.item()
                total_ccc += ccc.detach()
                vepoch.set_postfix(loss=valid_loss/len(valid_loader), ccc=total_ccc.cpu().numpy()/(len(vepoch)))
        print(f'Validation Loss: {valid_loss/len(valid_loader)}')
        print(f'Validation CCC (Valence, Arousal, Dominance): {tuple(total_ccc.cpu().numpy()/len(valid_loader))}')

# 训练模型
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print(torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))
model.to(device)
train(model, train_loader, valid_loader, epochs=10)