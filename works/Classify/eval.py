import numpy
import numpy as np

from model.MyDenseNet.DenseNet import *
from model.MyVAE.VAE import *
from torch.utils.data import DataLoader
from MyDataset import creatDataset
from model.vgg import CreateVgg16
import torch
from config.config import configs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# GPU或者CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 种类
classes = configs["train_config"]["classes"]
# 语音标签
labels = {"America": 0, "answer": 1, "China": 2, "goodbye": 3, "hello": 4, "Mike": 5, "potato": 6}

test_num = [0 for i in range(classes)]

# 模型建立
if configs["model_config"]["model_name"] == 'densenet121':
    model = CreateDenseNet121(classes).to(device)
    print("--- models : CreateDenseNet121 ---")

if configs["model_config"]["model_name"] == 'VAE':
    model = VAE(classes).to(device)
    print("--- models : VAE ---")

# model.load_state_dict(torch.load(configs["data_config"]["best_model_weights"], map_location=torch.device('cpu')))

# 测试集地址
test_dir = configs["data_config"]["val_dir"]

test_dataset = creatDataset(test_dir)  # + val_dataset2
test_loader = DataLoader(test_dataset, batch_size=configs["train_config"]["batch_size"], shuffle=True)


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


correct = 0
total = 0
print("---begin---")
model.eval()
# 选择的模型权重
model.load_state_dict(torch.load(configs["data_config"]["weights"]))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# 混淆矩阵
conf_matrix = torch.zeros(classes, classes)
t_sne_data = []
t_sne_label = []

# 正确和错误情况下的置信度
error_conf = []
correct_conf = []

# 不同类别准确率
label_acc = [0 for i in range(classes)]

for i, data in enumerate(test_loader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    for label in labels:
        test_num[label.item()] += 1

    outputs = model(inputs)

    out, label = outputs.cpu(), labels.cpu()
    t_sne_data += out.tolist()
    t_sne_label += label.tolist()

    conf_matrix = confusion_matrix(outputs, labels, conf_matrix)

    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum().item()
    predicted_label = predicted.tolist()
    labels_label = labels.tolist()
    for i in range(len(predicted_label)):
        confidence = softmax(outputs.data.tolist()[i])
        if predicted_label[i] != labels_label[i]:
            print(" ")
            print("predict:", label_KEY[predicted_label[i]])
            print(confidence[predicted_label[i]])
            print("label:", label_KEY[labels_label[i]])
            print(confidence[labels_label[i]])
            error_conf.append(confidence[predicted_label[i]])
        else:
            correct_conf.append(confidence[predicted_label[i]])
            label_acc[predicted_label[i]] += 1
    total += labels.size(0)
acc = correct / total
print("test_acc:", acc)

print("error:")
print("average confidence:", np.sum(error_conf) / len(error_conf))
print("over 0.99:", len([item for item in error_conf if item > 0.99]) / len(error_conf))

print("correct:")
print("average confidence:", np.sum(correct_conf) / len(correct_conf))
print("over 0.99:", len([item for item in correct_conf if item > 0.99]) / len(correct_conf))


def plot_matrix(conf_matrix):
    acc_matrix = np.zeros_like(conf_matrix, dtype=float)
    for x in range(classes):
        for y in range(classes):
            if test_num[x] > 0:  # 防止除以零
                acc_matrix[y, x] = conf_matrix[y, x] / test_num[x]
            else:
                acc_matrix[y, x] = 0.0  # 如果某个类没有测试数据，准确率设为0
    acc_matrix = acc_matrix.T
    plt.imshow(acc_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    thresh = 0.5

    for x in range(classes):
        for y in range(classes):
            acc = acc_matrix[y, x]
            if acc == 0 or acc == 1:
                plt.text(x, y, int(acc), va='center', ha='center',
                         color="white" if acc > thresh else "black")
            else:
                plt.text(x, y, "{:.2f}".format(acc), va='center', ha='center',
                         color="white" if acc > thresh else "black")
    plt.tight_layout()
    plt.yticks(range(classes), label_KEY)
    plt.xticks(range(classes), label_KEY)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


color_map = ['#FF0000', '#FFA500', '#008000', '#00FFFF', '#FFFF00', '#0000FF', '#800080', '#FFC0CB', '#808080',
             '#800000', '#FF00FF', '#00FF00', '#008080', '#000080', '#FFD700', '#FF6347', '#00CED1', '#FF69B4',
             '#4B0082', '#7CFC00', '#F08080', '#ADD8E6', '#DDA0DD', '#B0C4DE', '#00FF7F', '#FA8072']


def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_map[label[i]])
    for i, item in enumerate(set(label)):
        plt.annotate(label_KEY[item], (data[label == item, 0].mean(), data[label == item, 1].mean()),
                     color=color_map[i])

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return fig


def plot_sne():
    data, label = t_sne_data, t_sne_label
    n_samples, n_features = data.shape
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    result_2D = tsne_2D.fit_transform(data)

    fig1 = plot_embedding_2D(result_2D, label, 't-SNE')  # 将二维数据用plt绘制出来

print(label_acc)
# 每个类别准确率
label_acc = [round(item / test_num[index] * 100, 2) for index, item in enumerate(label_acc)]
plt.bar(label_KEY, label_acc, color='#4995C6')
for i in range(len(label_KEY)):
    plt.text(label_KEY[i], label_acc[i], label_acc[i], ha='center', va='bottom')
plt.ylabel('Accuracy (%)')
plt.xlabel('key')
plt.show()

# t-SNE图
t_sne_data = np.array(t_sne_data)
t_sne_label = np.array(t_sne_label)
plot_sne()


# 混淆矩阵
plot_matrix(conf_matrix)
