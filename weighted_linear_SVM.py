import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier

# 假设你已经有了处理好的情绪数据集，并加载为 NumPy 数组
# X = data[:, :-1]  # 特征：包含你的特征数据
# y = data[:, -1]  # 标签：包含 'ang', 'hap', 'sad', 'neu'

# 或者从CSV文件加载
import pandas as pd
data = pd.read_csv('data.csv')
# 确保你的数据中情绪标签列命名为 'emotion'
X = data[['valence', 'arousal', 'dominance']].values
y = data['Label'].values

# 数据预处理：标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=40)

# 根据论文中的描述设置gamma
gamma_value = 1 / X_train.shape[1]  # feature dimension is 3, so gamma = 1/3

# 创建一对一的SVM模型
model = OneVsOneClassifier(svm.SVC(kernel='rbf', gamma=gamma_value, class_weight='balanced'))

# 训练模型
model.fit(X_train, y_train)


# 预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 注意：实际代码中需要将 'feature1', 'feature2', 'feature3' 和 'emotion' 替换为你的数据集中的相应列名
