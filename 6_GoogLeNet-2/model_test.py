import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from model import GoogLeNet, Inception
from PIL import Image


def test_data_process():
    # 定义数据集路径
    ROOT = r'./data/test'
    # 定义归一化方法
    normalize = transforms.Normalize(mean=[0.162, 0.151, 0.138], std=[0.058, 0.052, 0.047])
    # 定义数据集处理方法变量
    test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # 加载数据集
    data = ImageFolder(root=ROOT, transform=test_transforms)

    test_dataloader = DataLoader(data,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('./path/best_model_weights.pth'))
    # 加载数据测试集
    # test_dataloader = test_data_process()
    # test_model_process(model, test_dataloader)

    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    classes = ['苹果', '香蕉', '葡萄', '橘子', '梨子']
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         # 设置模型为验证模型
    #         model.eval()
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output, dim=1)
    #         result = pre_lab.item()
    #         label = b_y.item()
    #         print("预测值：", classes[result], "------", "真实值：", classes[label])

    # 单个水果预测
    image = Image.open('./simple/pe.jfif')
    # 定义归一化方法
    normalize = transforms.Normalize(mean=[0.22890568, 0.19639583, 0.1433638],
                                     std=[0.09950783, 0.07997292, 0.06596899])
    # 定义数据集处理方法
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          normalize])
    image = test_transforms(image)
    print(image.shape)

    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_label = torch.argmax(output, dim=1)
        result = pre_label.item()

        print('预测:', classes[result])
