import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import ast

# 基础统计计算函数
def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_variance(numbers):
    mean = calculate_mean(numbers)
    return sum((x - mean) ** 2 for x in numbers) / len(numbers)

def calculate_std_deviation(numbers):
    return calculate_variance(numbers) ** 0.5

# 解析CSV文件数据
def parse_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # 跳过头部
        for row in reader:
            vad = ast.literal_eval(row[6])
            data.append([row[1], row[2], row[3], row[4], row[5], vad])
    return data

# 绘制椭球体函数
def plot_ellipsoid(ax, center, radii, color, label):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.3, label=label+' Ellipsoid')

# 主功能函数
def process_emotion_data(file_path):
    data = parse_csv(file_path)
    emotions = ['neu', 'ang', 'sad', 'hap']
    emotion_data = {emotion: [] for emotion in emotions}

    # 分类并收集情感数据
    for row in data:
        emotion = row[2]
        if emotion in emotions:
            emotion_data[emotion].append(row[5])

    # 创建图形和3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 颜色映射
    colors = {'neu': 'green', 'ang': 'red', 'sad': 'blue', 'hap': 'yellow'}

    # 对每种情感计算平均值和标准差，并绘制椭球体和散点图
    for emotion, points in emotion_data.items():
        if points:
            v, a, d = zip(*points)
            center = (calculate_mean(v), calculate_mean(a), calculate_mean(d))
            radii = (calculate_std_deviation(v), calculate_std_deviation(a), calculate_std_deviation(d))
            plot_ellipsoid(ax, center, radii, colors[emotion], emotion)
            ax.scatter(v, a, d, color=colors[emotion], s=5, label=emotion+' Points')

    # 设置坐标轴标签和标题
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_title("3D Scatter Plot of VAD Values with Emotion Ellipsoids and Points")
    ax.legend(ncol=2, loc='upper right')

    # 显示图形
    plt.show()

# 调用主功能函数
file_path = 'data/iemocap_new.csv'  # 替换为你的CSV文件的实际路径
process_emotion_data(file_path)
