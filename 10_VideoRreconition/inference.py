import numpy as np
import torch
import cv2
from C3D_model import C3D
from torch.autograd import Variable
import torch.nn as nn

# 剪切视频帧
def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def inference():
    # 定义模型训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载数据集标签
    with open("./data/labels.txt", 'r') as f :
        class_names = f.readlines()
        # print(class_names)
        f.close()

    # 加载模型，并将模型参数加载到模型中
    # 加载模型，并将模型参数加载到模型中
    model = C3D(num_classes=101)
    checkpoint = torch.load('model_result/models/C3D_epoch-29.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # 将模型放入到设备中，并设置验证模式
    model.to(device)
    model.eval()

    video = './video/v_YoYo_g01_c01.avi'
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read() # 读取最后返回False和空
        if not retaining and frame is None:
            continue
        temp_ = center_crop(cv2.resize(frame, (171, 128)))
        temp = temp_ - np.array([[[90.0, 98.0, 102.0]]])  # 归一化
        clip.append(temp)

        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)  # 扩展维度1(批次) (1, 16, 112, 112, 3)
            inputs = np.transpose(inputs,(0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = Variable(inputs, requires_grad=False).to(device) # 允许梯度下降
            # inputs.requires_grad_(True)

            with torch.no_grad():
                outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)


            clip.pop(0)

        frame = cv2.resize(frame, (640, 480))

        cv2.imshow('result', frame)
        cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    inference()
