import os

print(os.listdir(r"C:\Data\Code_Data\carvana-image-masking-challenge"))

"""
1.解压缩train.zip,train_masks.zip
2.生成train,train_masks两个文件夹(分别有5088张图片)
3.分别从train,train_masks划分图片生成train(4600张图片),val,train_masks(4600张图片),val_masks
"""
import zipfile
import shutil

DATASET_DIR = r'C:\Data\Code_Data\carvana-image-masking-challenge\\'
WORKING_DIR = r'C:\Code\Python\cv\12_Unet\dataset\\'


# 解压缩
def unzip(DATASET_DIR, WORKING_DIR):
    if len(os.listdir(WORKING_DIR)) <= 1:
        with zipfile.ZipFile(DATASET_DIR + 'train.zip', 'r') as zip_file:
            zip_file.extractall(WORKING_DIR)

        with zipfile.ZipFile(DATASET_DIR + 'train_masks.zip', 'r') as zip_file:
            zip_file.extractall(WORKING_DIR)

        print(
            len(os.listdir(WORKING_DIR + 'train')),
            len(os.listdir(WORKING_DIR + 'train_masks'))
        )


# 数据集的划分
def data_split(WORKING_DIR):
    train_dir = WORKING_DIR + 'train/'
    val_dir = WORKING_DIR + 'val/'
    os.mkdir(val_dir)
    for file in sorted(os.listdir(train_dir))[4600:]:
        shutil.move(train_dir + file, val_dir)

    masks_dir = WORKING_DIR + 'train_masks/'
    val_masks_dir = WORKING_DIR + 'val_masks/'
    os.mkdir(val_masks_dir)
    for file in sorted(os.listdir(masks_dir))[4600:]:
        shutil.move(masks_dir + file, val_masks_dir)


if __name__ == '__main__':
    #  解压缩
    unzip(DATASET_DIR, WORKING_DIR)
    # 数据集划分
    data_split(WORKING_DIR)
