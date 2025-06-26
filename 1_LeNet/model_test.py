import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
from model import LeNet


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=Compose([Resize(size=28), ToTensor()]),
                             download=True)

    test_dataloader = DataLoader(test_data,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 将模型放入GPU
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征值放入GPU
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 设置成评估模式
            model.eval()

            # 输出预测值
            output = model(test_data_x)
            # 输出最大行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y).double().item()
            # 对所有的测试样本进行累加
            test_num += test_data_y.size(0)

    # 计算准确率
    test_acc = test_corrects / test_num
    print("测试集的准确率为:%.4f" % test_acc)
    print("测试集的准确率为:{:.4f}".format(test_acc))


if __name__ == '__main__':
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load('./path/best_model_weights.pth'))
    # 加载数据测试集
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)
