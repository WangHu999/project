# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def fitness(x):
    # 计算模型的适应度作为指标的加权组合
    # x: 输入数组，形状为(n, 4)，其中 n 是样本数量，包含 [P, R, mAP@0.5, mAP@0.5:0.95]

    # 定义每个指标的权重
    w = [0.0, 0.0, 0.1, 0.9]  # 权重分别为 [P, R, mAP@0.5, mAP@0.5:0.95]

    # 计算加权和并返回
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ 计算平均精度，给定召回率和精度曲线。
    来源: https://github.com/rafaelpadilla/Object-Detection-Metrics.

    参数:
        tp: 真实正例 (nparray, nx1 或 nx10)。
        conf: 目标存在的置信度值，范围在0到1 (nparray)。
        pred_cls: 预测的物体类别 (nparray)。
        target_cls: 真实物体类别 (nparray)。
        plot: 是否绘制精度-召回曲线。
        save_dir: 绘图保存目录。

    返回:
        计算出的平均精度。
    """

    # 按置信度排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 查找唯一的类别
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # 类别数量

    # 创建精度-召回曲线并为每个类别计算AP
    px, py = np.linspace(0, 1, 1000), []  # 用于绘图
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # 标签数量
        n_p = i.sum()  # 预测数量

        if n_p == 0 or n_l == 0:
            continue
        else:
            # 累计FP和TP
            fpc = (1 - tp[i]).cumsum(0)  # 假阳性累计
            tpc = tp[i].cumsum(0)  # 真阳性累计

            # 召回率
            recall = tpc / (n_l + 1e-16)  # 召回曲线
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # 反向插值

            # 精度
            precision = tpc / (tpc + fpc)  # 精度曲线
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # 在pr_score下的精度

            # 从召回-精度曲线计算AP
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # 在mAP@0.5下的精度

    # 计算F1（精度和召回的调和平均）
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # 找到最大F1对应的索引
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')  # 返回精度、召回率、AP、F1和唯一类别


def compute_ap(recall, precision):
    """ 计算平均精度，给定召回率和精度曲线
    # 参数
        recall:    召回率曲线 (list)
        precision: 精度曲线 (list)
    # 返回
        平均精度, 精度曲线, 召回率曲线
    """

    # 在开头和结尾附加哨兵值
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # 计算精度包络线
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 积分计算曲线下面积
    method = 'interp'  # 方法: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101点插值 (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # 积分
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # x轴（召回率）变化的点
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 曲线下面积

    return ap, mpre, mrec  # 返回平均精度、精度曲线和召回率曲线



class ConfusionMatrix:
    # 更新版本的混淆矩阵类，来自于 https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))  # 初始化混淆矩阵
        self.nc = nc  # 类别数量
        self.conf = conf  # 置信度阈值
        self.iou_thres = iou_thres  # IOU阈值

    def process_batch(self, detections, labels):
        """
        计算箱体的交并比（Jaccard index）。
        两个箱体集合都应该是 (x1, y1, x2, y2) 格式。
        参数:
            detections (Array[N, 6])：检测结果，包括 x1, y1, x2, y2, 置信度, 类别
            labels (Array[M, 5])：真实标签，包括 类别, x1, y1, x2, y2
        返回：
            None，更新混淆矩阵
        """
        detections = detections[detections[:, 4] > self.conf]  # 过滤低置信度的检测结果
        gt_classes = labels[:, 0].int()  # 获取真实标签的类别
        detection_classes = detections[:, 5].int()  # 获取检测结果的类别
        iou = box_iou(labels[:, 1:], detections[:, :4])  # 计算IOU

        x = torch.where(iou > self.iou_thres)  # 找到高于阈值的IOU索引
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按照IOU排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 去重
                matches = matches[matches[:, 2].argsort()[::-1]]  # 再次排序
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 再次去重
        else:
            matches = np.zeros((0, 3))  # 如果没有匹配，则返回空数组

        n = matches.shape[0] > 0  # 检查是否有匹配
        m0, m1, _ = matches.transpose().astype(np.int16)  # 提取匹配结果的索引
        for i, gc in enumerate(gt_classes):
            j = m0 == i  # 检查每个真实类别的匹配情况
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # 正确匹配
            else:
                self.matrix[self.nc, gc] += 1  # 背景错误匹配

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # 背景漏检

    def matrix(self):
        return self.matrix  # 返回混淆矩阵

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn
            import matplotlib.pyplot as plt
            import warnings

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # 规范化列
            array[array < 0.005] = np.nan  # 不进行标注的值

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # 标签大小
            labels = (0 < len(names) < 99) and len(names) == self.nc  # 应用名称到标签
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')  # x轴标签
            fig.axes[0].set_ylabel('Predicted')  # y轴标签
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)  # 保存混淆矩阵图
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')  # 捕获绘制错误

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))  # 打印混淆矩阵



def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # 返回box1与box2的IoU。box1是4个值，box2是nx4的数组
    box2 = box2.T

    # 获取边界框的坐标
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # 从xywh转为xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # 交集面积
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # 并集面积
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 最小包围框的宽度
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 最小包围框的高度
        if CIoU or DIoU:  # 距离或完全IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # 最小包围框对角线的平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 中心距离的平方
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # 最小包围框面积
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # 返回IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    返回边界框的交并比（Jaccard指数）。
    两组边界框预计采用 (x1, y1, x2, y2) 格式。
    参数：
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    返回：
        iou (Tensor[N, M]): 包含 boxes1 和 boxes2 中每个元素的成对 IoU 值的 NxM 矩阵
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # 交集计算
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ 返回 box1 相对于 box2 的交集比率。
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # 获取边界框的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # 计算交集面积
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # 计算 box2 的面积
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # 返回 box1 相对于 box2 的交集比率
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # 计算宽高 (width-height) 的交并比 (IoU) 矩阵。
    # wh1 是形状为 (n, 2) 的张量，表示 n 个宽高框。
    # wh2 是形状为 (m, 2) 的张量，表示 m 个宽高框。

    wh1 = wh1[:, None]  # 将 wh1 扩展维度，变为形状 [N, 1, 2]，以便进行广播操作。
    wh2 = wh2[None]  # 将 wh2 扩展维度，变为形状 [1, M, 2]，以便进行广播操作。

    # 计算交集区域：对每个宽高对取最小值并计算面积，结果为形状 [N, M]。
    inter = torch.min(wh1, wh2).prod(2)  # 对最后一维 (宽和高) 进行乘积运算，得到交集面积。

    # 计算并返回 IoU：交集面积除以并集面积。
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # IoU = 交集 / (框1面积 + 框2面积 - 交集面积)


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    """
    绘制精确率-召回率 (Precision-Recall) 曲线。

    参数：
        px (array-like): 召回率 (recall) 值。
        py (list of arrays): 每个类的精确率 (precision) 值列表。
        ap (array-like): 平均精确率 (average precision) 值，形状为 (num_classes, 2)。
        save_dir (str): 保存图像的路径，默认为 'pr_curve.png'。
        names (tuple): 类别名称，若少于 21 个，则在图例中显示。

    返回：
        None
    """

    # 创建图形和坐标轴
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    # 将精确率数据堆叠在一起以便处理
    py = np.stack(py, axis=1)

    # 如果类别数小于 21，逐类绘制曲线
    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # 绘制每个类的 (recall, precision) 曲线
    else:
        # 绘制所有类的精确率平均值曲线
        ax.plot(px, py, linewidth=1, color='grey')  # 绘制所有类的 (recall, precision) 曲线

    # 绘制所有类的平均精确率曲线
    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())

    # 设置坐标轴标签
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    # 设置坐标轴范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 添加图例
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # 保存图像
    fig.savefig(Path(save_dir), dpi=250)

    # 关闭图形，释放内存
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    """
    绘制指标-置信度 (Metric-Confidence) 曲线。

    参数：
        px (array-like): 置信度 (confidence) 值。
        py (list of arrays): 每个类的指标值列表。
        save_dir (str): 保存图像的路径，默认为 'mc_curve.png'。
        names (tuple): 类别名称，若少于 21 个，则在图例中显示。
        xlabel (str): X 轴标签，默认为 'Confidence'。
        ylabel (str): Y 轴标签，默认为 'Metric'。

    返回：
        None
    """

    # 创建图形和坐标轴
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    # 如果类别数小于 21，逐类绘制曲线
    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # 绘制每个类的 (confidence, metric) 曲线
    else:
        # 绘制所有类的指标值曲线
        ax.plot(px, py.T, linewidth=1, color='grey')  # 绘制所有类的 (confidence, metric) 曲线

    # 计算所有类的平均指标值
    y = py.mean(0)

    # 绘制所有类的平均指标曲线
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')

    # 设置坐标轴标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 设置坐标轴范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 添加图例
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # 保存图像
    fig.savefig(Path(save_dir), dpi=250)

    # 关闭图形，释放内存
    plt.close()