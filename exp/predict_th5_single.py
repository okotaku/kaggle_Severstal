import os
import gc
import cv2
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import segmentation_models_pytorch as smp
from apex import amp
from tqdm import tqdm
from contextlib import contextmanager
from albumentations import *
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append("../severstal-src/")
from util import seed_torch, search_threshold, rle2mask
from logger import setup_logger, LOGGER
from trainer import predict, predict_dsv
from metric import dice
sys.path.append("../")
import segmentation_models_pytorch as smp
import segmentation_models_pytorch2 as smp_old

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
EPOCHS = 71
FOLD_ID = 0
EXP_ID = "exp77_unet_resnet"
CLASSIFICATION = True
GAMMA = 0.8
base_ckpt = 14
#base_model = None
base_model = "models/{}_fold{}_ckpt{}.pth".format(EXP_ID, FOLD_ID, base_ckpt)
if N_CLASSES == 4:
    ths = [0.5, 0.5, 0.5, 0.5]
elif N_CLASSES == 3:
    ths = [0.5, 0.5, 0.5]

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)
LOGGER.info("seed={}".format(SEED))


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
                 cut_h=False
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
        self.gamma = GAMMA

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_path = os.path.join(self.img_dir, img_id)

        img = cv2.imread(img_path)

        if self.gamma is not None:
            lookUpTable = np.empty((1, 256), np.uint8)
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, self.gamma) * 255.0, 0, 255)
            img = cv2.LUT(img, lookUpTable)

        w, h, _ = img.shape
        mask = np.zeros((w, h, self.n_classes))
        for i, encoded in enumerate(cur_idx_row[self.mask_colname]):
            if encoded in "-1":
                continue
            else:
                mask[:, :, i] = rle2mask(encoded, (w, h))

        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        mask[mask != 0] = 1

        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        if self.class_y is None:
            return torch.Tensor(img), torch.Tensor(mask)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def main(seed):
    with timer('load data'):
        df = pd.read_csv(FOLD_PATH)
        if N_CLASSES == 3:
            df.drop("EncodedPixels_2", axis=1, inplace=True)
            df = df.rename(columns={"EncodedPixels_3": "EncodedPixels_2"})
            df = df.rename(columns={"EncodedPixels_4": "EncodedPixels_3"})

    with timer('preprocessing'):
        val_df = df[df.fold_id == FOLD_ID]

        val_augmentation = None
        val_dataset = SeverDataset(val_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                  transforms=val_augmentation)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        del val_df, df, val_dataset
        gc.collect()

    with timer('create model'):
        model = smp.Unet('resnet34', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
                         decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
                         classification=CLASSIFICATION, attention_type="cbam", center=True)
        model.load_state_dict(torch.load(base_model))
        model.to(device)
        model.eval()

        criterion = torch.nn.BCEWithLogitsLoss()

    with timer('predict'):
        valid_loss, y_pred, y_true, cls = predict(model, val_loader, criterion, device, classification=CLASSIFICATION)
        LOGGER.info('Mean valid loss: {}'.format(round(valid_loss, 5)))

        scores = []
        all_scores = []
        for i, th in enumerate(ths):
            sum_val_preds = np.sum(y_pred[:, i, :, :].reshape(len(y_pred), -1) > th, axis=1)

            best = 0
            for n_th, remove_mask_pixel in enumerate([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]):
                val_preds_ = copy.deepcopy(y_pred[:, i, :, :])
                val_preds_[sum_val_preds < remove_mask_pixel] = 0
                scores_ = []
                all_scores_ = []
                for y_val_, y_pred_ in zip(y_true[:, i, :, :], val_preds_):
                    score = dice(y_val_, y_pred_ > 0.5)
                    if np.isnan(score):
                        scores_.append(1)
                    else:
                        scores_.append(score)
                LOGGER.info('dice={} on {}'.format(np.mean(scores_), remove_mask_pixel))
                if np.mean(scores_) >= best:
                    best = np.mean(scores_)
                all_scores_.append(np.mean(scores_))
            scores.append(best)
            all_scores.append(all_scores_)

        LOGGER.info('holdout dice={}'.format(np.mean(scores)))
        np.save("all_scores_fold{}.npy".format(FOLD_ID), np.array(all_scores))


if __name__ == '__main__':
    main(SEED)
