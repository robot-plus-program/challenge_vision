import os
from datetime import datetime

import natsort
import cv2
import numpy as np
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from centerpt import CenterPointNet


LR = 0.01
EPOCH = 200
BATCH_SIZE = 2
IMAGE_SIZE = [128, 128]
LR_SCHEDULER = True
LR_REDUCE_EPOCH = 10
LR_REDUCE_GAMMA = 0.8

SAVE = True

TRAIN_PATH = 'dataset/train'
VALID_PATH = 'dataset/valid'

# BASELINE = 'efficientnet'
BASELINE = 'mobilenet'
PRETRAINED = 'checkpoint/20240321_epoch28_lr0.0069.pth'

def main():
    today = datetime.now().strftime("%Y%m%d")

    model = CenterPointNet(baseline=BASELINE, pretrained=PRETRAINED)
    model = model.get_model()
    model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    train_set = CenterPTDataset(TRAIN_PATH)
    valid_set  = CenterPTDataset(VALID_PATH)

    train_loader = DataLoader(train_set,
                              batch_size=BATCH_SIZE,
                              num_workers=8,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=BATCH_SIZE,
                              num_workers=8,
                              shuffle=False)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=LR_REDUCE_EPOCH * len(train_loader),
                                             gamma=LR_REDUCE_GAMMA)

    best_epoch = 0
    best_mLoss = 10

    for e in range(EPOCH):
        # Train
        model.train()

        mLoss = AverageMeter()

        progressbar = tqdm(train_loader)
        for inputs, target in progressbar:
            inputs = inputs.cuda()
            target = target.cuda()

            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(target, output)
            loss.backward()

            optimizer.step()

            mLoss.update(loss.item())

            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            if cur_lr > 0.0001 and LR_SCHEDULER:
                lr_scheduler.step()

            progressbar.set_description(f"[Train] [{e+1:2d}/{EPOCH}] mLoss: {mLoss.avg:.4f} \tLR: {cur_lr:.6f}\t")
            progressbar.update()

        # Valid
        model.eval()

        mLoss = AverageMeter()

        progressbar = tqdm(valid_loader)
        for inputs, target in progressbar:
            inputs = inputs.cuda()
            target = target.cuda()

            output = model(inputs)
            loss = criterion(target, output)

            mLoss.update(loss.item())

            progressbar.set_description(f"[Valid] [{e+1:2d}/{EPOCH}] mLoss: {mLoss.avg:.4f}\t\t\t\t\t")
            progressbar.update()

        # Save
        if mLoss.avg < best_mLoss and mLoss.avg < 0.03:
            best_mLoss = mLoss.avg
            best_epoch = e+1

            if SAVE:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint', exist_ok=True)

                save_name = f"{today}_{BASELINE}_epoch{best_epoch}_lr{best_mLoss:.4f}"

                torch.save(model.state_dict(),
                           f"checkpoint/{save_name}.pth")
                # torch.jit.save(torch.jit.script(model),
                #                f"checkpoint/{save_name}.pt")

                print(f"Weight saved. [{save_name}]")


class CenterPTDataset(Dataset):
    def __init__(self, path):
        self.image_path = f"{path}/image"
        self.label_path = f"{path}/label"

        il = natsort.natsorted(os.listdir(self.image_path))
        ll = natsort.natsorted(os.listdir(self.label_path))

        self.data_list = []

        for name in il:
            name = name.split('.')[0]
            try:
                image = self.open_image(il[il.index(f"{name}.png")])
                label = self.open_label(ll[ll.index(f"{name}.txt")])

                if np.any(image) and label:
                    self.data_list.append([image, label])

            except:
                pass

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),  # MobileNetV3의 입력 크기에 맞게 조정
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        image = self.data_list[idx][0]
        label = self.data_list[idx][1]

        h, w = image.shape[:2]
        x, y = label

        x = (2 * x / w) - 1
        y = (2 * y / h) - 1
        # x /= w
        # y /= h

        label = torch.FloatTensor([x, y])
        image = self.T(image.copy())

        return image, label

    def open_image(self, name):
        try:
            path = f"{self.image_path}/{name}"
            return cv2.imread(path)[:, :, ::-1]
        except:
            return False

    def open_label(self, name):
        try:
            path = f"{self.label_path}/{name}"
            with open(path, 'r') as f:
                data = f.readline()
            if data:
                data = [int(i) for i in data.split(',')]
                return data
            else:
                return False
        except:
            return False


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()
