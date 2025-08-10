
from model.MyDenseNet.DenseNet import *
from model.MyVAE.VAE import *
from model.MyResNet.ResNet import *
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from MyDataset import creatDataset
from tqdm import tqdm
from model.vgg import *
from config.config import configs


# 单GPU或者CPU
print(torch.cuda.is_available())
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# 训练参数
epochs = configs["train_config"]["epochs"]
lr = configs["train_config"]["learining_rate"]
# 是否继续之前的训练
subsequent_training = configs["train_config"]["subsequent_training"]
# 种类
classes = configs["train_config"]["classes"]
batch_size = configs["train_config"]["batch_size"]
# 数据地址
data_dir = configs["data_config"]["train_dir"]
val_dir = configs["data_config"]["val_dir"]
# checkpoint
checkpoint_path = configs["data_config"]["checkpoint_path"]

#
# train_Dataset2 = creatDataset('dataset/train')

# 创建训练集和验证集
train_Dataset = creatDataset(data_dir) #+ train_Dataset2
val_dataset = creatDataset(val_dir)
train_loader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# 模型创建
if configs["model_config"]["model_name"] == 'densenet121':
    model = CreateDenseNet121(classes).to(device)
    print("--- models : densenet121 ---")


# 损失函数
if configs["train_config"]["loss"] == "CrossEntropyLoss":
    loss = nn.CrossEntropyLoss()
    print("--- loss : CrossEntropy ---")
if configs["train_config"]["loss"] == "BCELoss":
    loss = nn.BCELoss()
    loss.size_average = False
    print("--- loss : BCE ---")
if configs["train_config"]["loss"] == "MSELoss":
    loss = nn.MSELoss()
    print("--- loss : MSE ---")

# 优化器
if configs["train_config"]["optimizer"] == "SGD":
    optimizer = SGD(model.parameters(), lr=configs["train_config"]["learining_rate"], momentum=configs["train_config"]["momentum"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    print("--- optimizer : SGD ---")
elif configs["train_config"]["optimizer"] == "Adam":
    optimizer = Adam(model.parameters(), lr=configs["train_config"]["learining_rate"])
    print("--- optimizer : Adam ---")

start_epoch = 0
if subsequent_training:  # 如果是断点继续上次训练
    try:
        checkpoints = torch.load(os.path.join(checkpoint_path, configs["data_config"]["checkpoint"]))
        start_epoch = checkpoints['epoch']
        optimizer.load_state_dict(checkpoints['optimizer'])
        model.load_state_dict(checkpoints['models'])
        print('--- 继续上次训练 ---')
    except:
        print("--- start train ---")

best_acc = 0.0
best_loss = 99999
best_model_weights = model.state_dict()
writer = SummaryWriter(configs["data_config"]["log_dir"])
for epoch in range(start_epoch, epochs):
    training_loss = 0.  # 训练集loss
    training_acc = 0.   # 训练集准确率
    train_correct = 0
    train_total = 0

    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in loop:
        # 原本的
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = loss(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
        training_loss += batch_loss.item()

        loop.set_description(f'Epoch[{epoch}/{epochs}]')
        loop.set_postfix(loss=training_loss/train_total, acc=train_correct/train_total)

    print("\rEpoch:{}/{},training_loss:{},training_acc:{}".format(epoch+1, epochs, training_loss/train_total, train_correct/train_total))
    training_acc = train_correct / train_total
    training_loss = training_loss / train_total
    if training_loss <= best_loss:
        best_loss = training_loss
        best_model_weights = model.state_dict()
        torch.save(best_model_weights, configs["data_config"]["best_model_weights"])
        print("\rbest.pth保存成功")
    writer.add_scalar(tag='train_loss', scalar_value=training_loss/len(train_loader), global_step= epoch + 1)
    writer.add_scalar(tag='train_acc', scalar_value=training_acc, global_step= epoch + 1)

    val_acc = 0.
    val_correct = 0
    val_total = 0
    model.eval()
    print("\r---测试集---")
    loop = tqdm(enumerate(val_loader), total=len(val_loader))
    for j, val_data in loop:
        val_input, val_label = val_data
        val_input = val_input.to(device)
        val_label = val_label.to(device)
        val_pred = model(val_input)

        _, val_predicted = torch.max(val_pred.data, 1)
        val_correct += (val_predicted == val_label).sum().item()
        val_total += val_label.size(0)

    val_acc = val_correct / val_total
    print('val_acc:{}'.format(val_acc))
    writer.add_scalar(tag='val_acc', scalar_value=val_acc, global_step=epoch + 1)

    checkpoint = {
        'epoch': epoch + 1,
        'optimizer': optimizer.state_dict(),
        'models': model.state_dict()
    }
    torch.save(checkpoint, os.path.join(checkpoint_path,configs["data_config"]["checkpoint"]))
    print('\r---保存各参数完成---')

    torch.save(best_model_weights, configs["data_config"]["last_model_weights"])




