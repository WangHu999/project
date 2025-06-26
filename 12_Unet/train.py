import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import Unet
from utils import get_dataloader

import numpy as np
import random

# Hyper Parameter
LEARNING_RATE = 1e-8
BATCH_SIZE = 8
NUM_EPOCHS = 6
LEARNING_RATE_DECAY = 0
PIN_MEMORY = True
# LOAD_MODEL = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using device:', DEVICE)

TRAIN_IMG_PATH = r'./dataset/train'
TRAIN_MASK_PATH = r'./dataset/train_masks'
VAL_IMG_PATH = r'./dataset/val'
VAL_MASK_PATH = r'./dataset/val_masks'

IMAGE_HEIGHT = 320
IMG_WIDTH = 480
NUM_WORKERS = 8

train_losses = []
val_acc = []
val_dice = []

# 设置随机种子
seed = random.randint(1, 100)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def train_fn(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    for index, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.unsqueeze(1).float().to(DEVICE)

        with torch.cuda.amp.autocast(enabled=True):
            predict = model(data)
            loss = loss_fn(predict, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(DEVICE)
            y = y.unsqueeze(1).float().to(DEVICE)
            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == y).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2 * (predictions * y).sum()) / (2 * (predictions * y).sum()
                                                           + ((predictions * y) < 1).sum())
    accuracy = round(float(num_correct / num_pixels), 4)
    dice = round(float(dice_score / len(loader)), 4)

    print(f"Got {num_correct} / {num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice Score: {dice_score} / {len(loader)}")

    model.train()
    return accuracy, dice


def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=35, p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ], )

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMG_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ], )

    train_loader, val_loader = get_dataloader(TRAIN_IMG_PATH, TRAIN_MASK_PATH,
                                              VAL_IMG_PATH, VAL_MASK_PATH,
                                              train_transform, val_transform,
                                              BATCH_SIZE, num_workers=NUM_WORKERS,
                                              pin_memory=PIN_MEMORY)

    model = Unet(in_channel=3, out_channel=1).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for index in range(NUM_EPOCHS):
        print("Current Epoch: ", index)
        train_loss = train_fn(train_loader, model, loss_fn, optimizer, scaler)
        train_losses.append(train_loss)

        accuracy, dice = check_accuracy(val_loader, model, device=DEVICE)
        val_acc.append(accuracy)
        val_dice.append(dice)
        print(f"accuracy:{accuracy}")
        print(f"dice score:{dice}")


if __name__ == "__main__":
    main()
