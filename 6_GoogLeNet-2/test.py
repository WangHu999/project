import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        # 初始化se_block类，继承自nn.Module
        # channel参数是输入特征图的通道数
        # ratio参数是一个缩放比例，默认值为16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应平均池化，将输入特征图的每个通道池化到1x1大小
        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channel, channel // ratio, bias=False),
                # 线性层，将通道数从channel降到channel // ratio，去掉偏置项
                nn.ReLU(inplace=True),
                # ReLU激活函数，就地操作
                nn.Linear(channel // ratio, channel, bias=False),
                # 线性层，将通道数从channel // ratio恢复到channel，去掉偏置项
                nn.Sigmoid()
                # Sigmoid激活函数，将输出压缩到(0,1)范围内
        )

    def forward(self, x):
        # 前向传播函数，定义输入数据如何通过网络层传播
        b, c, _, _ = x.size()

        y = self.avg_pool(x)
        # 对x进行自适应平均池化，然后重塑为(b, c)的形状
        y = self.fc(y)
        # 将y传入全连接层，输出形状为(b, c)，然后再重塑为(b, c, 1, 1)
        return x * y
        # 将输入x与y按元素相乘，得到加权后的输出
