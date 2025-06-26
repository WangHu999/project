# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, \
    xywh2xyxy, xywhn2xyxy, xyxy2xywhn, xyn2xy
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'  # 提供的帮助文档链接
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # 可接受的图像文件后缀
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # 可接受的视频文件后缀
NUM_THREADS = min(8, os.cpu_count())  # 使用的多线程数量，最多为8个或CPU核心数量的最小值

# 获取Exif中的方向标签
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':  # 查找“Orientation”标签
        break  # 找到后退出循环


def get_hash(paths):
    """
    返回文件或目录路径列表的单个哈希值

    参数:
        paths: 文件或目录的路径列表

    返回:
        h: 计算出的哈希值
    """
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # 计算所有路径的大小总和
    h = hashlib.md5(str(size).encode())  # 对大小进行MD5哈希
    h.update(''.join(paths).encode())  # 对路径进行MD5哈希
    return h.hexdigest()  # 返回最终的哈希值


def exif_size(img):
    """
    返回经过Exif校正的PIL图像大小

    参数:
        img: PIL图像对象

    返回:
        s: 校正后的图像大小元组 (宽度, 高度)
    """
    s = img.size  # 获取图像的原始大小 (宽度, 高度)
    try:
        # 从图像的Exif信息中获取方向值
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # 如果旋转值为6（顺时针90度）
            s = (s[1], s[0])  # 交换宽高
        elif rotation == 8:  # 如果旋转值为8（逆时针90度）
            s = (s[1], s[0])  # 交换宽高
    except:
        pass  # 如果没有Exif信息或发生错误，保持原始大小

    return s  # 返回校正后的大小



def exif_transpose(image):
    """
    根据图像的EXIF方向标签对PIL图像进行转置。
    来源：https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: 需要转置的图像。
    :return: 处理后的图像。
    """
    exif = image.getexif()  # 获取图像的EXIF信息
    orientation = exif.get(0x0112, 1)  # 获取方向标签（默认为1，即无旋转）

    if orientation > 1:  # 如果方向标签大于1，说明图像需要旋转或翻转
        # 根据方向标签选择相应的转置方法
        method = {2: Image.FLIP_LEFT_RIGHT,    # 水平翻转
                  3: Image.ROTATE_180,       # 180度旋转
                  4: Image.FLIP_TOP_BOTTOM,   # 垂直翻转
                  5: Image.TRANSPOSE,         # 左上到右下对角线翻转
                  6: Image.ROTATE_270,       # 顺时针旋转270度
                  7: Image.TRANSVERSE,       # 右上到左下对角线翻转
                  8: Image.ROTATE_90,        # 顺时针旋转90度
                  }.get(orientation)  # 根据方向标签获取转置方法

        if method is not None:  # 如果找到了对应的转置方法
            image = image.transpose(method)  # 对图像进行转置
            del exif[0x0112]  # 删除方向标签，因为已处理
            image.info["exif"] = exif.tobytes()  # 更新图像的EXIF信息

    return image  # 返回处理后的图像



def create_dataloader(
    path, imgsz, batch_size, stride, single_cls=False, hyp=None,
    augment=False, cache=False, pad=0.0, rect=False, rank=-1,
    workers=8, image_weights=False, quad=False, prefix=''
):
    # 确保只有第一个进程在 DDP 中首先处理数据集，其他进程可以使用缓存
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path, imgsz, batch_size,
            augment=augment,  # 是否进行图像增强
            hyp=hyp,  # 增强的超参数
            rect=rect,  # 是否进行矩形训练
            cache_images=cache,  # 是否缓存图像
            single_cls=single_cls,  # 是否为单类别检测
            stride=int(stride),  # 步幅
            pad=pad,  # 填充
            image_weights=image_weights,  # 是否使用图像加权
            prefix=prefix  # 日志前缀
        )

    batch_size = min(batch_size, len(dataset))  # 确保批大小不超过数据集大小
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # 计算工作线程数
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None  # 分布式采样器
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader  # 选择加载器

    # 使用 torch.utils.data.DataLoader() 如果数据集属性在训练期间会更新，否则使用 InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn
    )
    return dataloader, dataset  # 返回数据加载器和数据集



class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    一个可重复使用工作线程的Dataloader

    采用与普通DataLoader相同的语法。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化InfiniteDataLoader实例。

        Args:
            *args: 传递给父类DataLoader的参数。
            **kwargs: 传递给父类DataLoader的关键字参数。
        """
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))  # 将batch_sampler替换为_repeat_sampler
        self.iterator = super().__iter__()  # 获取父类的迭代器

    def __len__(self):
        """
        返回样本的数量。

        Returns:
            int: batch_sampler中的样本数量。
        """
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """
        迭代器方法，支持无限迭代。

        Yields:
            返回每次迭代的下一个数据批次。
        """
        for i in range(len(self)):
            yield next(self.iterator)  # 从父类迭代器中获取下一个批次


class _RepeatSampler(object):
    """
    一个无限重复的Sampler

    Args:
        sampler (Sampler): 用于生成样本的原始采样器。
    """

    def __init__(self, sampler):
        """
        初始化_repeat_sampler实例。

        Args:
            sampler (Sampler): 传入的样本采样器。
        """
        self.sampler = sampler  # 保存传入的采样器

    def __iter__(self):
        """
        无限迭代器方法，支持对样本的无限重复采样。

        Yields:
            从sampler中生成的样本。
        """
        while True:
            yield from iter(self.sampler)  # 无限返回采样器中的样本



class LoadImages:
    # YOLOv5 图像/视频数据加载器，示例用法：`python detect.py --source image.jpg/vid.mp4`

    def __init__(self, path, img_size=640, stride=32, auto=True):
        # 将路径转换为操作系统无关的绝对路径
        p = str(Path(path).resolve())

        # 检查路径是否包含通配符，并使用 glob 模块查找匹配的文件
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # 使用通配符查找文件
        # 如果路径是目录，获取目录下所有文件
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # 从目录中获取所有文件
        # 如果路径是文件，直接将其加入文件列表
        elif os.path.isfile(p):
            files = [p]  # 文件路径
        else:
            # 如果路径不存在，则抛出异常
            raise Exception(f'ERROR: {p} does not exist')

        # 将文件分为图像和视频
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]  # 图像文件
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]  # 视频文件
        ni, nv = len(images), len(videos)  # 图像和视频的数量

        self.img_size = img_size  # 设置图像大小
        self.stride = stride  # 设置步幅
        self.files = images + videos  # 合并所有文件
        self.nf = ni + nv  # 文件总数量
        self.video_flag = [False] * ni + [True] * nv  # 标记哪些是图像，哪些是视频
        self.mode = 'image'  # 初始模式设置为图像
        self.auto = auto  # 是否自动调整大小的标志

        # 如果有视频文件，则初始化第一个视频
        if any(videos):
            self.new_video(videos[0])  # 初始化第一个视频
        else:
            self.cap = None  # 没有视频捕获对象

        # 确保至少有一个文件存在
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        # 初始化迭代计数器
        self.count = 0
        return self  # 返回自身以支持迭代

    def __next__(self):
        # 检查是否已处理完所有文件
        if self.count == self.nf:
            raise StopIteration  # 如果所有文件已处理，停止迭代

        path = self.files[self.count]  # 获取当前文件的路径

        if self.video_flag[self.count]:
            # 处理视频文件加载
            self.mode = 'video'
            ret_val, img0 = self.cap.read()  # 从视频捕获对象读取一帧
            if not ret_val:
                # 如果读取失败，更新计数并释放当前视频捕获对象
                self.count += 1
                self.cap.release()  # 释放当前视频捕获对象
                if self.count == self.nf:  # 如果已达到最后一个视频
                    raise StopIteration  # 停止迭代
                else:
                    path = self.files[self.count]  # 获取下一个文件的路径
                    self.new_video(path)  # 加载下一个视频
                    ret_val, img0 = self.cap.read()  # 再次尝试读取帧

            self.frame += 1  # 更新当前帧计数
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # 处理图像文件加载
            self.count += 1  # 更新计数器
            img0 = cv2.imread(path)  # 使用 OpenCV 读取图像（BGR 格式）
            assert img0 is not None, 'Image Not Found ' + path  # 确保图像成功加载
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # 进行图像的填充调整
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # 将图像转换为模型所需的格式
        img = img.transpose((2, 0, 1))[::-1]  # 将 HWC 转换为 CHW，并将 BGR 转换为 RGB
        img = np.ascontiguousarray(img)  # 确保数组是连续的

        # 返回路径、处理后的图像、原始图像和视频捕获对象
        return path, img, img0, self.cap

    def new_video(self, path):
        # 为新的视频重置帧计数
        self.frame = 0
        self.cap = cv2.VideoCapture(path)  # 初始化视频捕获对象
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频中的总帧数

    def __len__(self):
        # 返回文件的数量
        return self.nf  # 文件数量


class LoadWebcam:  # 用于推理
    """
    YOLOv5本地网络摄像头数据加载器，例如：`python detect.py --source 0`
    """

    def __init__(self, pipe='0', img_size=640, stride=32):
        """
        初始化LoadWebcam实例。

        Args:
            pipe (str): 摄像头的输入源，可以是数字（摄像头ID）或字符串（视频文件路径）。
            img_size (int): 输入图像的大小，默认为640。
            stride (int): 图像处理的步幅，默认为32。
        """
        self.img_size = img_size  # 设置图像大小
        self.stride = stride  # 设置步幅
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe  # 解析输入源
        self.cap = cv2.VideoCapture(self.pipe)  # 创建视频捕捉对象
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 设置缓冲区大小

    def __iter__(self):
        """
        初始化迭代器。

        Returns:
            self: 返回自身以支持迭代。
        """
        self.count = -1  # 计数器初始化
        return self

    def __next__(self):
        """
        获取下一个帧的数据。

        Returns:
            tuple: 包含图像路径、处理后的图像、原始图像和None。

        Raises:
            StopIteration: 如果用户按下'q'键退出。
        """
        self.count += 1  # 递增计数器
        if cv2.waitKey(1) == ord('q'):  # 检测到'q'键则退出
            self.cap.release()  # 释放摄像头
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
            raise StopIteration  # 引发停止迭代异常

        # 读取帧
        ret_val, img0 = self.cap.read()  # 从摄像头读取图像
        img0 = cv2.flip(img0, 1)  # 左右翻转图像

        # 检查读取结果
        assert ret_val, f'Camera Error {self.pipe}'  # 确保成功读取
        img_path = 'webcam.jpg'  # 图像路径
        print(f'webcam {self.count}: ', end='')  # 打印当前帧计数

        # 填充缩放
        img = letterbox(img0, self.img_size, stride=self.stride)[0]  # 将图像调整为目标大小

        # 转换图像格式
        img = img.transpose((2, 0, 1))[::-1]  # HWC到CHW，并将BGR转换为RGB
        img = np.ascontiguousarray(img)  # 确保数组是连续的

        return img_path, img, img0, None  # 返回图像路径、处理后的图像、原始图像和占位符

    def __len__(self):
        """
        返回数据集的长度。

        Returns:
            int: 由于这是无限加载器，因此返回0。
        """
        return 0  # 无限加载器长度为0



class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        # 初始化 LoadStreams 类
        self.mode = 'stream'  # 设置模式为流
        self.img_size = img_size  # 设置图像大小
        self.stride = stride  # 设置步幅

        # 检查 sources 是否为文件，如果是，则读取文件内容
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]  # 如果不是文件，直接将其包装成列表

        n = len(sources)  # 获取源数量
        # 初始化图像、帧率、帧数和线程的列表
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # 清理源名称以便后续使用
        self.auto = auto  # 是否自动调整

        for i, s in enumerate(sources):  # 遍历源列表
            # 启动线程从视频流中读取帧
            print(f'{i + 1}/{n}: {s}... ', end='')  # 显示当前源索引和源地址
            if 'youtube.com/' in s or 'youtu.be/' in s:  # 如果源是 YouTube 视频
                check_requirements(('pafy', 'youtube_dl'))  # 检查依赖
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # 获取最佳 YouTube URL
            s = eval(s) if s.isnumeric() else s  # 如果是数字，则将其作为本地摄像头索引
            cap = cv2.VideoCapture(s)  # 打开视频流
            assert cap.isOpened(), f'Failed to open {s}'  # 确保视频流已成功打开

            # 获取视频流的宽度和高度
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 获取帧率，默认为 30 FPS
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0
            # 获取帧数，默认为无限流
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')

            _, self.imgs[i] = cap.read()  # 确保读取第一帧
            # 创建线程以异步更新帧
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")  # 输出成功信息
            self.threads[i].start()  # 启动线程
        print('')  # 换行

        # 检查图像形状的一致性
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # 如果所有形状相等，则进行矩形推理
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')  # 输出警告

    def update(self, i, cap, stream):
        # 在守护线程中读取流 `i` 的帧
        n, f, read = 0, self.frames[i], 1  # 帧数，帧数组，每 'read' 帧进行一次推理
        while cap.isOpened() and n < f:  # 当视频流仍然打开且未达到帧数限制时
            n += 1
            # cap.grab()  # 抓取帧
            cap.grab()  # 抓取下一帧，不返回图像
            if n % read == 0:  # 每隔一定帧数读取一次图像
                success, im = cap.retrieve()  # 获取图像
                if success:
                    self.imgs[i] = im  # 更新图像
                else:
                    print('WARNING: Video stream unresponsive, please check your IP camera connection.')  # 输出警告
                    self.imgs[i] *= 0  # 如果读取失败，设置图像为零
                    cap.open(stream)  # 重新打开流，如果信号丢失
            time.sleep(1 / self.fps[i])  # 等待时间，确保帧率

    def __iter__(self):
        self.count = -1  # 计数器初始化
        return self  # 返回迭代器对象

    def __next__(self):
        self.count += 1  # 计数器递增
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # 如果所有线程都不再存活，或者按下 'q' 键
            cv2.destroyAllWindows()  # 关闭所有窗口
            raise StopIteration  # 停止迭代

        # Letterbox
        img0 = self.imgs.copy()  # 复制当前图像
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]  # 对每幅图像进行 Letterbox 操作

        # Stack
        img = np.stack(img, 0)  # 将所有图像堆叠为一个批次

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR 转 RGB，BHWC 转 BCHW
        img = np.ascontiguousarray(img)  # 确保返回的数组是连续的

        return self.sources, img, img0, None  # 返回源地址、处理后的图像、原始图像和 None

    def __len__(self):
        return len(self.sources)  # 返回源的数量，便于迭代



def img2label_paths(img_paths):
    """
    将图像路径转换为标签路径。

    Args:
        img_paths (list): 包含图像文件路径的列表。

    Returns:
        list: 对应于输入图像路径的标签文件路径列表。
    """
    # 定义图像和标签路径的子字符串
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # '/images/' 和 '/labels/' 子字符串

    # 遍历每个图像路径，将其转换为标签路径
    return [
        sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths
    ]



class LoadImagesAndLabels(Dataset):
    # YOLOv5 的训练/验证数据加载器，用于加载图像及其标签
    cache_version = 0.5  # 数据集标签的缓存版本

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        # 初始化数据加载器的参数
        self.img_size = img_size  # 图像大小
        self.augment = augment  # 是否使用数据增强
        self.hyp = hyp  # 超参数
        self.image_weights = image_weights  # 是否使用图像权重
        self.rect = False if image_weights else rect  # 如果使用图像权重，则不使用矩形训练
        self.mosaic = self.augment and not self.rect  # 是否使用马赛克增强（仅在训练时）
        self.mosaic_border = [-img_size // 2, -img_size // 2]  # 马赛克增强的边界
        self.stride = stride  # 网络的步幅
        self.path = path  # 图像路径
        self.albumentations = Albumentations() if augment else None  # 初始化数据增强工具

        try:
            f = []  # 图像文件列表
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # 处理路径，适应不同操作系统
                if p.is_dir():  # 如果是目录
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)  # 递归查找图像文件
                elif p.is_file():  # 如果是文件
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()  # 读取文件内容
                        parent = str(p.parent) + os.sep  # 父目录路径
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # 更新路径
                else:
                    raise Exception(f'{prefix}{p} does not exist')  # 报错：路径不存在
            # 筛选并排序图像文件
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS])
            assert self.img_files, f'{prefix}No images found'  # 确保找到图像
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')  # 报错：加载数据时出错

        # 检查缓存
        self.label_files = img2label_paths(self.img_files)  # 获取标签文件路径
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # 设置缓存路径
        try:
            # 尝试加载缓存
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # 加载缓存字典
            assert cache['version'] == self.cache_version  # 确保缓存版本一致
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # 确保缓存哈希一致
        except:
            # 如果缓存不存在，则创建新的缓存
            cache, exists = self.cache_labels(cache_path, prefix), False  # 创建新的缓存

        # 显示缓存信息
        nf, nm, ne, nc, n = cache.pop('results')  # 提取缓存统计信息
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # 显示缓存结果
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))  # 显示警告信息
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'  # 确保找到标签

        # 读取缓存
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # 移除不需要的项
        labels, shapes, self.segments = zip(*cache.values())  # 解压标签、形状和分段信息
        self.labels = list(labels)  # 标签
        self.shapes = np.array(shapes, dtype=np.float64)  # 图像形状
        self.img_files = list(cache.keys())  # 更新图像文件列表
        self.label_files = img2label_paths(cache.keys())  # 更新标签文件列表
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0  # 如果是单类任务，则将所有标签的类编号设置为0

        n = len(shapes)  # 图像数量
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # 批次索引
        nb = bi[-1] + 1  # 批次数量
        self.batch = bi  # 图像的批次索引
        self.n = n  # 总图像数量
        self.indices = range(n)  # 索引范围

        # 矩形训练
        if self.rect:
            # 根据宽高比排序
            s = self.shapes  # 图像的宽高
            ar = s[:, 1] / s[:, 0]  # 计算宽高比
            irect = ar.argsort()  # 获取排序索引
            self.img_files = [self.img_files[i] for i in irect]  # 按宽高比排序图像文件
            self.label_files = [self.label_files[i] for i in irect]  # 按宽高比排序标签文件
            self.labels = [self.labels[i] for i in irect]  # 按宽高比排序标签
            self.shapes = s[irect]  # 更新图像形状
            ar = ar[irect]  # 更新宽高比

            # 设置训练图像的形状
            shapes = [[1, 1]] * nb  # 初始化形状列表
            for i in range(nb):
                ari = ar[bi == i]  # 获取当前批次的宽高比
                mini, maxi = ari.min(), ari.max()  # 获取最小和最大宽高比
                if maxi < 1:
                    shapes[i] = [maxi, 1]  # 如果最大宽高比小于1，设置形状为 [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]  # 如果最小宽高比大于1，设置形状为 [1, 1/mini]

            # 计算批次形状，向上取整并进行步幅调整
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # 将图像缓存到内存以加速训练（警告：大型数据集可能会超出系统内存）
        self.imgs, self.img_npy = [None] * n, [None] * n  # 初始化图像和缓存路径
        if cache_images:
            if cache_images == 'disk':
                # 如果选择将图像缓存到磁盘
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')  # 缓存目录
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]  # 缓存文件路径
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)  # 创建缓存目录
            gb = 0  # 缓存图像的大小（以GB为单位）
            self.img_hw0, self.img_hw = [None] * n, [None] * n  # 初始化原始和调整后图像大小
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 多线程加载图像
            pbar = tqdm(enumerate(results), total=n)  # 初始化进度条
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])  # 保存缓存图像
                    gb += self.img_npy[i].stat().st_size  # 更新缓存大小
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # 加载图像及其大小
                    gb += self.imgs[i].nbytes  # 更新缓存大小
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'  # 更新进度描述
            pbar.close()  # 关闭进度条

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # 缓存数据集标签，检查图像并读取形状
        x = {}  # 初始化字典用于存储图像、标签、形状和段落信息
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 统计缺失、找到、空、损坏的数量和消息
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."  # 描述信息
        with Pool(NUM_THREADS) as pool:  # 创建线程池
            # 使用进度条显示图像和标签验证的进度
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                # 更新计数器
                nm += nm_f  # 更新缺失的标签数量
                nf += nf_f  # 更新找到的标签数量
                ne += ne_f  # 更新空标签数量
                nc += nc_f  # 更新损坏标签数量
                if im_file:  # 如果找到了图像文件
                    x[im_file] = [l, shape, segments]  # 将图像文件、标签、形状和段落存储在字典中
                if msg:  # 如果有消息
                    msgs.append(msg)  # 收集消息
                # 更新进度条描述
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()  # 关闭进度条
        if msgs:  # 如果有警告消息
            logging.info('\n'.join(msgs))  # 记录警告消息
        if nf == 0:  # 如果没有找到任何标签
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')  # 发出警告

        # 生成缓存数据
        x['hash'] = get_hash(self.label_files + self.img_files)  # 计算哈希值
        x['results'] = nf, nm, ne, nc, len(self.img_files)  # 保存统计结果
        x['msgs'] = msgs  # 保存警告信息
        x['version'] = self.cache_version  # 缓存版本

        try:
            np.save(path, x)  # 保存缓存以便下次使用
            path.with_suffix('.cache.npy').rename(path)  # 移除 .npy 后缀并重命名
            logging.info(f'{prefix}New cache created: {path}')  # 记录新缓存创建的信息
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # 记录目录不可写的警告

        return x  # 返回缓存数据

    def __len__(self):
        # 返回图像文件的数量
        return len(self.img_files)

    def __getitem__(self, index):
        # 根据索引获取图像和标签
        index = self.indices[index]  # 线性、打乱或根据图像权重获取索引

        hyp = self.hyp  # 超参数
        mosaic = self.mosaic and random.random() < hyp['mosaic']  # 决定是否使用马赛克增强
        if mosaic:
            # 加载马赛克图像
            img, labels = load_mosaic(self, index)  # 加载马赛克图像和标签
            shapes = None  # 不存储形状信息

            # MixUp 增强
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # 加载单个图像
            img, (h0, w0), (h, w) = load_image(self, index)  # 加载图像及其原始和调整后的尺寸

            # 信纸框处理
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # 最终信纸框形状
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)  # 调整图像形状
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # 为 COCO mAP 重标定存储形状信息

            labels = self.labels[index].copy()  # 获取标签
            if labels.size:  # 如果有标签，将标准化的 xywh 转换为像素的 xyxy 格式
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:  # 如果需要增强
                img, labels = random_perspective(img, labels,  # 随机透视变换
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # 标签数量
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True,
                                        eps=1E-3)  # 转换标签为标准格式

        if self.augment:
            # 使用 Albumentations 进行数据增强
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # 更新标签数量

            # HSV 颜色空间增强
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # 上下翻转
            if random.random() < hyp['flipud']:
                img = np.flipud(img)  # 翻转图像
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]  # 更新标签坐标

            # 左右翻转
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)  # 翻转图像
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]  # 更新标签坐标

            # Cutouts 增强（可选）
            # labels = cutout(img, labels, p=0.5)

        # 创建输出标签
        labels_out = torch.zeros((nl, 6))  # 初始化标签输出
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)  # 将标签转为 PyTorch 张量

        # 转换图像格式
        img = img.transpose((2, 0, 1))[::-1]  # HWC 转为 CHW，同时从 BGR 转为 RGB
        img = np.ascontiguousarray(img)  # 确保图像是连续的内存块

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes  # 返回图像、标签、文件名和形状信息

    @staticmethod
    def collate_fn(batch):
        # 从批次中提取图像、标签、路径和形状
        img, label, path, shapes = zip(*batch)  # 进行转置
        for i, l in enumerate(label):
            l[:, 0] = i  # 为每个标签添加目标图像索引，便于后续处理
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes  # 返回堆叠的图像、合并的标签、路径和形状

    @staticmethod
    def collate_fn4(batch):
        # 从批次中提取图像、标签、路径和形状
        img, label, path, shapes = zip(*batch)  # 进行转置
        n = len(shapes) // 4  # 计算每组图像的数量
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]  # 初始化输出列表

        # 定义用于图像处理的张量
        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])  # 偏移量
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])  # 偏移量
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # 缩放因子
        for i in range(n):  # 遍历每组图像
            i *= 4  # 计算索引
            if random.random() < 0.5:  # 随机决定图像处理方式
                # 通过插值扩大图像
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]  # 直接使用标签
            else:
                # 拼接四个图像
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                # 拼接并调整标签
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)  # 添加处理后的图像
            label4.append(l)  # 添加处理后的标签

        for i, l in enumerate(label4):
            l[:, 0] = i  # 为每个标签添加目标图像索引

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4  # 返回堆叠的图像、合并的标签、路径和形状


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # 从数据集中加载索引 'i' 的图像，返回图像、原始高宽和调整后的高宽
    im = self.imgs[i]  # 尝试从缓存中获取图像
    if im is None:  # 如果图像没有被缓存到内存中
        npy = self.img_npy[i]  # 获取对应的 .npy 文件路径
        if npy and npy.exists():  # 如果 .npy 文件存在
            im = np.load(npy)  # 加载 .npy 文件中的图像数据
        else:  # 否则读取图像文件
            path = self.img_files[i]  # 获取图像文件的路径
            im = cv2.imread(path)  # 使用 OpenCV 读取图像 (BGR 格式)
            assert im is not None, 'Image Not Found ' + path  # 确保图像被正确加载
        h0, w0 = im.shape[:2]  # 获取原始图像的高和宽
        r = self.img_size / max(h0, w0)  # 计算缩放比例
        if r != 1:  # 如果图像大小不等
            # 根据缩放比例调整图像大小
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # 返回调整后的图像、原始高宽和调整后的高宽
    else:
        # 如果图像已缓存，直接返回缓存的图像和高宽信息
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # 返回图像、原始高宽和调整后的高宽


def load_mosaic(self, index):
    # YOLOv5 4-mosaic 加载器。加载1张图像和3张随机图像形成一个4图像拼接
    labels4, segments4 = [], []  # 用于存储拼接后的标签和分段
    s = self.img_size  # 定义拼接图像的大小
    # 随机生成拼接中心点的坐标
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
    # 随机选择3个额外的图像索引
    indices = [index] + random.choices(self.indices, k=3)
    random.shuffle(indices)  # 随机打乱索引顺序

    for i, index in enumerate(indices):
        # 加载图像
        img, _, (h, w) = load_image(self, index)

        # 根据索引放置图像
        if i == 0:  # 左上角
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # 创建一个基于114的空白拼接图像
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # 大图像的坐标
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # 小图像的坐标
        elif i == 1:  # 右上角
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # 左下角
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # 右下角
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将小图像放置到拼接图像的指定位置
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b  # 水平偏移量
        padh = y1a - y1b  # 垂直偏移量

        # 处理标签
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:  # 如果有标签
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # 将归一化的xywh转换为像素xyxy格式
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]  # 转换分段坐标
        labels4.append(labels)  # 添加标签
        segments4.extend(segments)  # 添加分段

    # 合并和裁剪标签
    labels4 = np.concatenate(labels4, 0)  # 合并所有标签
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # 裁剪坐标，以免超出范围

    # 数据增强
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # 随机透视变换

    return img4, labels4  # 返回拼接图像和标签



def load_mosaic9(self, index):
    # YOLOv5 9-mosaic 加载器。加载1张图像和8张随机图像形成一个9图像拼接
    labels9, segments9 = [], []  # 用于存储拼接后的标签和分段
    s = self.img_size  # 定义拼接图像的大小
    # 随机选择8个额外的图像索引
    indices = [index] + random.choices(self.indices, k=8)
    random.shuffle(indices)  # 随机打乱索引顺序

    for i, index in enumerate(indices):
        # 加载图像
        img, _, (h, w) = load_image(self, index)

        # 根据索引放置图像
        if i == 0:  # 中心
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # 创建一个基于114的空白拼接图像
            h0, w0 = h, w
            c = s, s, s + w, s + h  # (xmin, ymin, xmax, ymax) 坐标
        elif i == 1:  # 顶部
            c = s, s - h, s + w, s
        elif i == 2:  # 右上角
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # 右侧
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # 右下角
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # 底部
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # 左下角
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # 左侧
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # 左上角
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]  # 记录偏移量
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # 计算坐标，确保不超出边界

        # 处理标签
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:  # 如果有标签
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # 将归一化的xywh转换为像素xyxy格式
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]  # 转换分段坐标
        labels9.append(labels)  # 添加标签
        segments9.extend(segments)  # 添加分段

        # 将小图像放置到拼接图像的指定位置
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # 记录前一张图像的高度和宽度

    # 随机偏移中心点
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # 随机生成拼接中心点的坐标
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]  # 以中心点裁剪拼接图像

    # 合并和裁剪标签
    labels9 = np.concatenate(labels9, 0)  # 合并所有标签
    labels9[:, [1, 3]] -= xc  # 更新标签的x坐标
    labels9[:, [2, 4]] -= yc  # 更新标签的y坐标
    c = np.array([xc, yc])  # 记录中心点
    segments9 = [x - c for x in segments9]  # 更新分段坐标

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # 裁剪坐标，以免超出范围

    # 数据增强
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # 随机透视变换

    return img9, labels9  # 返回拼接图像和标签


def create_folder(path='./new'):
    # 创建文件夹函数
    # 参数:
    # path (str): 要创建的文件夹路径，默认为 './new'

    if os.path.exists(path):
        # 检查指定路径是否已存在
        shutil.rmtree(path)  # 如果存在，删除输出文件夹及其内容
    os.makedirs(path)  # 创建新的输出文件夹


def flatten_recursive(path='../datasets/coco128'):
    # 将递归目录展平，将所有文件移到顶层目录
    # 参数:
    # path (str): 要展平的目录路径，默认为 '../datasets/coco128'

    new_path = Path(path + '_flat')  # 创建新的路径，用于存放展平后的文件
    create_folder(new_path)  # 调用 create_folder 函数创建新文件夹

    # 使用 tqdm 进度条遍历指定路径下的所有文件
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        # 复制每个文件到新目录
        shutil.copyfile(file, new_path / Path(file).name)  # 通过 Path(file).name 获取文件名


def extract_boxes(path='../datasets/coco128'):
    # 将检测数据集转换为分类数据集，每个类一个目录
    # 参数:
    # path (str): 数据集的路径，默认为 '../datasets/coco128'

    path = Path(path)  # 将路径转换为 Path 对象
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # 删除已有的 'classifier' 目录

    files = list(path.rglob('*.*'))  # 递归查找所有文件
    n = len(files)  # 文件总数

    for im_file in tqdm(files, total=n):  # 遍历每个文件并显示进度条
        if im_file.suffix[1:] in IMG_FORMATS:  # 如果是图像文件
            im = cv2.imread(str(im_file))[..., ::-1]  # 读取图像并转换 BGR 到 RGB
            h, w = im.shape[:2]  # 获取图像的高度和宽度

            # 获取对应的标签文件
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():  # 检查标签文件是否存在
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # 读取标签

                for j, x in enumerate(lb):  # 遍历每个标签
                    c = int(x[0])  # 获取类别
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # 新文件名

                    if not f.parent.is_dir():  # 创建类目录
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # 计算边界框
                    b[2:] = b[2:] * 1.2 + 3  # 扩大边界框
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)  # 将框从 xywh 转换为 xyxy 格式

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # 限制框的 x 坐标在图像范围内
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)  # 限制框的 y 坐标在图像范围内

                    # 保存剪裁后的图像，并检查写入是否成功
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """
    自动将数据集分割为训练/验证/测试集，并保存路径下的 autosplit_*.txt 文件
    使用方法: from utils.datasets import *; autosplit()

    参数:
        path:            图像目录的路径
        weights:         训练、验证、测试的权重 (列表或元组)
        annotated_only:  仅使用带注释的图像
    """
    path = Path(path)  # 将路径转换为 Path 对象
    # 获取所有图像文件
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in IMG_FORMATS], [])  # 仅图像文件
    n = len(files)  # 文件总数
    random.seed(0)  # 设置随机种子以确保可重复性
    # 根据权重随机分配每个图像到训练、验证、测试集
    indices = random.choices([0, 1, 2], weights=weights, k=n)

    # 定义三个 txt 文件名
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']
    # 删除已有的 txt 文件
    [(path.parent / x).unlink(missing_ok=True) for x in txt]

    # 打印分割信息
    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)

    # 遍历每个图像和其对应的索引
    for i, img in tqdm(zip(indices, files), total=n):
        # 如果只使用带注释的图像，检查对应的标签文件是否存在
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            # 将图像路径写入对应的 txt 文件
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # 添加图像到 txt 文件


def verify_image_label(args):
    # 验证一对图像和标签
    im_file, lb_file, prefix = args  # 解包参数
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # 计数（缺失、找到、空、损坏）、消息、段落
    try:
        # 验证图像
        im = Image.open(im_file)  # 打开图像文件
        im.verify()  # 使用 PIL 验证图像完整性
        shape = exif_size(im)  # 获取图像大小
        # 确保图像大小大于 10 像素
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        # 确保图像格式有效
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            # 检查 JPEG 文件是否损坏
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)  # 定位到文件尾部倒数第二个字节
                if f.read() != b'\xff\xd9':  # 检查 JPEG 文件尾部
                    # 重新保存图像以修复损坏
                    Image.open(im_file).save(im_file, format='JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: corrupt JPEG restored and saved {im_file}'

        # 验证标签
        if os.path.isfile(lb_file):
            nf = 1  # 标签文件存在
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]  # 读取标签
                if any([len(x) > 8 for x in l]):  # 检查是否为分段
                    classes = np.array([x[0] for x in l], dtype=np.float32)  # 类别
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                    # 组合类别和边界框
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l = np.array(l, dtype=np.float32)  # 转换为浮点型数组
            if len(l):
                # 验证标签的格式和有效性
                assert l.shape[1] == 5, 'labels require 5 columns each'  # 每个标签必须有 5 列
                assert (l >= 0).all(), 'negative labels'  # 确保没有负值标签
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'  # 确保坐标归一化
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'  # 确保没有重复标签
            else:
                ne = 1  # 标签为空
                l = np.zeros((0, 5), dtype=np.float32)  # 返回空标签
        else:
            nm = 1  # 标签缺失
            l = np.zeros((0, 5), dtype=np.float32)  # 返回空标签
        return im_file, l, shape, segments, nm, nf, ne, nc, msg  # 返回结果
    except Exception as e:
        nc = 1  # 设置损坏计数
        msg = f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'  # 错误消息
        return [None, None, None, None, nm, nf, ne, nc, msg]  # 返回错误结果


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ 返回数据集统计字典，包括每个分割的图像和实例计数
    用法1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    用法2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    参数
        path:           data.yaml 或包含 data.yaml 的 data.zip 的路径
        autodownload:   如果未在本地找到数据集，则尝试下载
        verbose:        打印统计字典
    """

    def round_labels(labels):
        # 更新标签为整数类和 6 位小数的浮点数
        return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]

    def unzip(path):
        # 解压 data.zip TODO: 约束：path/to/abc.zip 必须解压到 'path/to/abc/'
        if str(path).endswith('.zip'):  # 如果路径是 data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # 解压缩
            dir = path.with_suffix('')  # 数据集目录 = zip 名称
            return True, str(dir), next(dir.rglob('*.yaml'))  # 返回压缩状态、数据目录和 yaml 路径
        else:  # 如果路径是 data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB 操作，用于调整单个图像 'f' 的大小并以降低质量保存在 /dataset-hub 中以供网页/应用查看
        f_new = im_dir / Path(f).name  # dataset-hub 图像文件名
        try:  # 使用 PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # 比例
            if r < 1.0:  # 图像太大
                im = im.resize((int(im.width * r), int(im.height * r)))  # 调整图像大小
            im.save(f_new, quality=75)  # 保存
        except Exception as e:  # 使用 OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # 比例
            if r < 1.0:  # 图像太大
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_LINEAR)  # 调整图像大小
            cv2.imwrite(str(f_new), im)  # 保存调整后的图像

    zipped, data_dir, yaml_path = unzip(Path(path))  # 解压或获取路径
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # 读取数据字典
        if zipped:
            data['path'] = data_dir  # 如果解压缩，更新路径
    check_dataset(data, autodownload)  # 检查并下载数据集（如果缺失）
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))  # hub 目录
    stats = {'nc': data['nc'], 'names': data['names']}  # 统计字典

    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # 如果没有测试集
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # 加载数据集
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))  # 统计每个类的实例
        x = np.array(x)  # 转换为数组，形状(128x80)
        stats[split] = {
            'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},  # 实例统计
            'image_stats': {
                'total': dataset.n,  # 图像总数
                'unlabelled': int(np.all(x == 0, 1).sum()),  # 未标记图像数
                'per_class': (x > 0).sum(0).tolist()  # 每类的标记图像数
            },
            'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in zip(dataset.img_files, dataset.labels)]  # 标签信息
        }

        if hub:
            im_dir = hub_dir / 'images'  # hub 图像目录
            im_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass  # 执行 HUB 操作

    # 性能分析
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)  # 保存为 npy 格式
            t2 = time.time()
            x = np.load(file, allow_pickle=True)  # 加载 npy 文件
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # 保存为 JSON 格式
            t2 = time.time()
            with open(file, 'r') as f:
                x = json.load(f)  # 加载 JSON 文件
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # 保存、打印和返回
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # 保存统计信息
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))  # 打印统计信息
    return stats  # 返回统计信息

