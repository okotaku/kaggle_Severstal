# ===============
# best_ckpt=
# ===============
import os
import gc
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import segmentation_models_pytorch as smp
from apex import amp
from contextlib import contextmanager
from albumentations import *
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import sys
sys.path.append("../severstal-src/")
from util import seed_torch, search_threshold
from logger import setup_logger, LOGGER



import warnings
warnings.filterwarnings('ignore')

# ===============
# Constants
# ===============
IMG_DIR = "../input/train_images/"
LOGGER_PATH = "log.txt"
FOLD_PATH = "../input/severstal_folds01.csv"
ID_COLUMNS = "ImageId"
N_CLASSES = 4


# ===============
# Settings
# ===============
SEED = np.random.randint(100000)
device = "cuda:0"
IMG_SIZE = (1600, 256)
CLR_CYCLE = 3
BATCH_SIZE = 32
EPOCHS = 137
FOLD_ID = 0
EXP_ID = "exp78_unet_resnet"
CLASSIFICATION = True
base_ckpt = 10
base_model = None
base_model = "models/{}_fold{}_latest.pth".format(EXP_ID, FOLD_ID)

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)
LOGGER.info("seed={}".format(SEED))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


class SeverDataset(Dataset):

    def __init__(self,
                 df,
                 img_dir,
                 img_size,
                 n_classes,
                 crop_rate=1.0,
                 id_colname="ImageId",
                 transforms=None,
                 means=[0.485, 0.456, 0.406],
                 stds=[0.229, 0.224, 0.225],
                 class_y=None,
                 cut_h=False,
                 crop_320=False,
                 gamma=None,
                 meaning=False
                 ):
        self.df = df
        self.img_dir = img_dir
        self.img_size = img_size
        self.transforms = transforms
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.id_colname = id_colname
        self.mask_colname = ["EncodedPixels_{}".format(i) for i in range(1, n_classes+1)]
        self.n_classes = n_classes
        self.crop_rate = crop_rate
        self.class_y = class_y
        self.cut_h = cut_h
        self.crop_320 = crop_320
        self.gamma = gamma
        self.meaning = meaning

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_path = os.path.join(self.img_dir, img_id)

        img = cv2.imread(img_path)
        if self.meaning is not None:
            img = (img - np.mean(img)) / np.std(img) * 32 + 100
            img = img.astype("uint8")

        if self.gamma is not None:
            lookUpTable = np.empty((1, 256), np.uint8)
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, self.gamma) * 255.0, 0, 255)
            img = cv2.LUT(img, lookUpTable)

        img = cv2.resize(img, self.img_size)

        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))

        return torch.from_numpy(img), img_id


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        model = models.resnet34(pretrained=True)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        out = x.squeeze()

        return out


def main(seed):
    with timer('load data'):
        df = pd.read_csv(FOLD_PATH)
        y1 = (df.EncodedPixels_1 != "-1").astype("float32").values.reshape(-1, 1)
        y2 = (df.EncodedPixels_2 != "-1").astype("float32").values.reshape(-1, 1)
        y3 = (df.EncodedPixels_3 != "-1").astype("float32").values.reshape(-1, 1)
        y4 = (df.EncodedPixels_4 != "-1").astype("float32").values.reshape(-1, 1)
        y = np.concatenate([y1, y2, y3, y4], axis=1)

    with timer('preprocessing'):
        train_augmentation = None
        train_dataset = SeverDataset(df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                    transforms=train_augmentation, crop_rate=1.0, class_y=y, meaning=None)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        del df, train_dataset
        gc.collect()

    with timer('create model'):
        model = ResNet()
        model.to(device)
        model.eval()

    with timer('train'):
        features = []
        ids = []
        for image, ids_ in train_loader:
            image = image.cuda()
            output = model(image)
            features.append(output.cpu().data.numpy().astype("float32"))
            ids.extend(ids_)

        np.save("ids.npy", np.array(ids))
        np.save("features.npy", np.array(features))


if __name__ == '__main__':
    main(SEED)
