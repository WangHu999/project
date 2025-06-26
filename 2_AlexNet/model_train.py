import copy
import time

from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader, random_split
from model import AlexNet
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd


def train_val_data_process():
    data = FashionMNIST(root='./data',
                        train=True,
                        transform=Compose([Resize(size=227), ToTensor()]),
                        download=True)

    train_data, val_data = random_split(data, [round(0.8 * len(data)), round(0.2 * len(data))])
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=1)

    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=64,
                                shuffle=True,
                                num_workers=1)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 将模型放到训练设备中
    model.to(device)
    # 构建损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 构建优化器对象
    optimizer = Adam(model.parameters(), lr=1e-3)
    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # 初始化参数
        # 训练集损失
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集损失
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入训练设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失
            loss = criterion(output, b_y)
            # 梯度清0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 对损失进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum((pre_lab == b_y))
            # 当前训练样本的数量
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入训练设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为评估模式
            model.eval()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失
            loss = criterion(output, b_y)

            # 对损失进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y)
            # 当前训练样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} train loss: {:.4f} train acc: {:.4f}".format(epoch + 1, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss: {:.4f} valid acc: {:.4f}".format(epoch + 1, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    torch.save(best_model_wts, './path/best_model_weights.pth')
    train_process = pd.DataFrame(data={'epoch': list(range(num_epochs)),
                                       'train_loss_all': train_loss_all,
                                       'train_acc_all': train_acc_all,
                                       'val_loss_all': val_loss_all,
                                       'val_acc_all': val_acc_all})

    return train_process


def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集与验证集的损失和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process['train_loss_all'], 'ro-', label='Train loss')
    plt.plot(train_process['epoch'], train_process['val_loss_all'], 'bs-', label='Val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_acc_all'], 'ro-', label='Train acc')
    plt.plot(train_process['epoch'], train_process['val_acc_all'], 'bs-', label='Val acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载所需要的模型
    AlexNet = AlexNet()
    # 加载数据集
    train_dataloader, val_dataloader = train_val_data_process()
    # 模型训练
    train_process = train_model_process(AlexNet, train_dataloader, val_dataloader, 20)
    # 绘图
    matplot_acc_loss(train_process)
