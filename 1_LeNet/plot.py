from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

# transforms.Resize(224) 用于调整图像的尺寸,它将图像的高度和宽度统一调整为 224 像素
# 许多流行的神经网络架构在设计时考虑的输入图像尺寸是 224x224 像素
train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                          download=True
                          )

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

b_x = 0.0
b_y = 0.0
# 获取一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

batch_x = b_x.squeeze().numpy()  # 去除维度为1的数据，并转成numpy数据
batch_y = b_y.numpy()

print(batch_x.shape, batch_y.shape)
print(batch_x, batch_y)
class_label = train_data.classes
# print(class_label)

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for i in np.arange(len(batch_y)):
    plt.subplot(4, 16, i + 1)
    plt.imshow(batch_x[i, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[i]], size=10)
    plt.axis('off')
    plt.tight_layout()
plt.show()



