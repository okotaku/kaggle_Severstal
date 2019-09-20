# ===============
# best_ckpt=17
# 2019-09-19 08:38:04,590 - INFO - Mean train loss: 0.00853
# 2019-09-19 08:39:09,219 - INFO - Mean valid loss: 0.00827
# 2019-09-19 08:40:13,552 - INFO - Mean EMA valid loss: 0.00779
# 2019-09-19 23:47:45,337 - INFO - dice=0.9628946998017844 on 800
# 2019-09-19 23:49:46,463 - INFO - dice=0.9909279885615222 on 1200
# 2019-09-19 23:51:16,791 - INFO - dice=0.8709502685862224 on 800
# 2019-09-19 23:52:47,977 - INFO - dice=0.9852685557191396 on 800
# 2019-09-20 12:42:39,467 - INFO - dice=0.9636942515879636 on 0.62
# 2019-09-20 12:43:53,722 - INFO - dice=0.9902655938929043 on 0.02
# 2019-09-20 12:49:10,817 - INFO - dice=0.8713122114503271 on 0.18
# 2019-09-20 12:55:34,150 - INFO - dice=0.9852305690939597 on 0.22
# best_ckpt=17, fold=1
# 2019-09-20 07:10:23,185 - INFO - Mean train loss: 0.00763
# 2019-09-20 07:11:23,938 - INFO - Mean valid loss: 0.00859
# 2019-09-20 07:12:24,977 - INFO - Mean EMA valid loss: 0.00856
# best_ckpt=20, fold=2
# 2019-09-20 03:03:52,071 - INFO - Mean train loss: 0.00889
# 2019-09-20 03:04:53,016 - INFO - Mean valid loss: 0.00904
# 2019-09-20 03:05:54,019 - INFO - Mean EMA valid loss: 0.00809
# best_ckpt=12, fold=3
# 2019-09-20 00:01:36,664 - INFO - Mean train loss: 0.00988
# 2019-09-20 00:02:39,863 - INFO - Mean valid loss: 0.0092
# 2019-09-20 00:03:43,701 - INFO - Mean EMA valid loss: 0.00867
# best_ckpt=15, fold=4
# 2019-09-20 02:08:53,682 - INFO - Mean train loss: 0.0091
# 2019-09-20 02:09:55,802 - INFO - Mean valid loss: 0.00941
# 2019-09-20 02:10:57,574 - INFO - Mean EMA valid loss: 0.00802
# ===============
import os
import gc
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import segmentation_models_pytorch as smp
from apex import amp
from contextlib import contextmanager
from albumentations import *
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append("../severstal-src/")
from util import seed_torch, search_threshold
from losses import FocalLovaszLoss
from datasets import SeverDataset, MaskProbSampler
from logger import setup_logger, LOGGER
from trainer import train_one_epoch, validate
from scheduler import GradualWarmupScheduler
sys.path.append("../")
import segmentation_models_pytorch as smp
from sync_batchnorm import convert_model


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
device = "cuda"
IMG_SIZE = (1600, 256)
CLR_CYCLE = 3
BATCH_SIZE = 32
EPOCHS = 125
FOLD_ID = 1
EXP_ID = "exp55_unet_resnet"
CLASSIFICATION = True
EMA = True
EMA_START = 6
base_ckpt = 17
base_model = None
base_model_ema = None
base_model = "models/{}_fold{}_latest.pth".format(EXP_ID, FOLD_ID)
base_model_ema = "models/{}_fold{}_latest_ema.pth".format(EXP_ID, FOLD_ID)

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)
LOGGER.info("seed={}".format(SEED))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def main(seed):
    with timer('load data'):
        df = pd.read_csv(FOLD_PATH)
        y1 = (df.EncodedPixels_1 != "-1").astype("float32").values.reshape(-1, 1)
        y2 = (df.EncodedPixels_2 != "-1").astype("float32").values.reshape(-1, 1)
        y3 = (df.EncodedPixels_3 != "-1").astype("float32").values.reshape(-1, 1)
        y4 = (df.EncodedPixels_4 != "-1").astype("float32").values.reshape(-1, 1)
        y = np.concatenate([y1, y2, y3, y4], axis=1)

    with timer('preprocessing'):
        train_df, val_df = df[df.fold_id != FOLD_ID], df[df.fold_id == FOLD_ID]
        y_train, y_val = y[df.fold_id != FOLD_ID], y[df.fold_id == FOLD_ID]

        train_augmentation = Compose([
            Flip(p=0.5),
            OneOf([
                GridDistortion(p=0.5),
                OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)
            ], p=0.5),
            OneOf([
                RandomGamma(gamma_limit=(100,140), p=0.5),
                RandomBrightnessContrast(p=0.5),
                RandomBrightness(p=0.5),
                RandomContrast(p=0.5)
            ], p=0.5),
            OneOf([
                GaussNoise(p=0.5),
                Cutout(num_holes=10, max_h_size=10, max_w_size=20, p=0.5)
            ], p=0.5),
            ShiftScaleRotate(rotate_limit=20, p=0.5),
        ])
        val_augmentation = None

        train_dataset = SeverDataset(train_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                    transforms=train_augmentation, crop_rate=1.0, class_y=y_train)
        val_dataset = SeverDataset(val_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                  transforms=val_augmentation)
        train_sampler = MaskProbSampler(train_df, demand_non_empty_proba=0.6)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        del train_df, val_df, df, train_dataset, val_dataset
        gc.collect()

    with timer('create model'):
        model = smp.Unet('resnet34', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
                         decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
                         classification=CLASSIFICATION, attention_type="cbam")
        model = convert_model(model)
        if base_model is not None:
            model.load_state_dict(torch.load(base_model))
        model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam([
            {'params': model.decoder.parameters(), 'lr': 3e-3},
            {'params': model.encoder.parameters(), 'lr': 3e-4},
        ])
        if base_model is None:
            scheduler_cosine = CosineAnnealingLR(optimizer, T_max=CLR_CYCLE, eta_min=3e-5)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=CLR_CYCLE*2, after_scheduler=scheduler_cosine)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=CLR_CYCLE, eta_min=3e-5)


        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

        if EMA:
            ema_model = copy.deepcopy(model)
            if base_model_ema is not None:
                ema_model.load_state_dict(torch.load(base_model_ema))
            ema_model.to(device)
        else:
            ema_model = None
        model = torch.nn.DataParallel(model)
        ema_model = torch.nn.DataParallel(ema_model)

    with timer('train'):
        train_losses = []
        valid_losses = []

        best_model_loss = 999
        best_model_ema_loss = 999
        best_model_ep = 0
        ema_decay = 0
        checkpoint = base_ckpt+1

        for epoch in range(102, EPOCHS + 1):
            seed = seed + epoch
            seed_torch(seed)

            if epoch >= EMA_START:
                ema_decay = 0.99

            LOGGER.info("Starting {} epoch...".format(epoch))
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, cutmix_prob=0.0,
                                      classification=CLASSIFICATION, ema_model=ema_model, ema_decay=ema_decay)
            train_losses.append(tr_loss)
            LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

            valid_loss = validate(model, val_loader, criterion, device, classification=CLASSIFICATION)
            valid_losses.append(valid_loss)
            LOGGER.info('Mean valid loss: {}'.format(round(valid_loss, 5)))

            if EMA and epoch >= EMA_START:
                ema_valid_loss = validate(ema_model, val_loader, criterion, device, classification=CLASSIFICATION)
                LOGGER.info('Mean EMA valid loss: {}'.format(round(ema_valid_loss, 5)))

                if ema_valid_loss < best_model_ema_loss:
                    torch.save(ema_model.module.state_dict(),
                               'models/{}_fold{}_ckpt{}_ema.pth'.format(EXP_ID, FOLD_ID, checkpoint))
                    best_model_ema_loss = ema_valid_loss

            scheduler.step()

            if valid_loss < best_model_loss:
                torch.save(model.module.state_dict(), 'models/{}_fold{}_ckpt{}.pth'.format(EXP_ID, FOLD_ID, checkpoint))
                best_model_loss = valid_loss
                best_model_ep = epoch
                #np.save("val_pred.npy", val_pred)

            if epoch % (CLR_CYCLE * 2) == CLR_CYCLE * 2 - 1:
                torch.save(model.module.state_dict(), 'models/{}_fold{}_latest.pth'.format(EXP_ID, FOLD_ID))
                LOGGER.info('Best valid loss: {} on epoch={}'.format(round(best_model_loss, 5), best_model_ep))
                if EMA:
                    torch.save(ema_model.module.state_dict(), 'models/{}_fold{}_latest_ema.pth'.format(EXP_ID, FOLD_ID))
                    LOGGER.info('Best ema valid loss: {}'.format(round(best_model_ema_loss, 5)))
                checkpoint += 1
                best_model_loss = 999

            #del val_pred
            gc.collect()

    LOGGER.info('Best valid loss: {} on epoch={}'.format(round(best_model_loss, 5), best_model_ep))

    xs = list(range(1, len(train_losses) + 1))
    plt.plot(xs, train_losses, label='Train loss')
    plt.plot(xs, valid_losses, label='Val loss')
    plt.legend()
    plt.xticks(xs)
    plt.xlabel('Epochs')
    plt.savefig("loss.png")


if __name__ == '__main__':
    main(SEED)
