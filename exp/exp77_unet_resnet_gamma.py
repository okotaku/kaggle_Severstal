# ===============
# best_ckpt=13, 14, fold=0, GAMMA=0.8
# 2019-09-29 05:41:50,708 - INFO - Mean train loss: 0.00827
# 2019-09-29 05:42:56,282 - INFO - Mean valid loss: 0.00768
# 2019-09-29 06:22:44,157 - INFO - Mean train loss: 0.00825
# 2019-09-29 06:23:50,598 - INFO - Mean valid loss: 0.00768
# 2019-09-30 01:49:15,063 - INFO - dice=0.9634999144968512 on 200
# 2019-09-30 01:49:33,908 - INFO - dice=0.9649331310000543 on 400
# 2019-09-30 01:49:52,702 - INFO - dice=0.9642013761031197 on 600
# 2019-09-30 01:50:11,515 - INFO - dice=0.9625960736019388 on 800
# 2019-09-30 01:50:30,312 - INFO - dice=0.9609646822098031 on 1000
# 2019-09-30 01:50:49,098 - INFO - dice=0.9581844338278953 on 1200
# 2019-09-30 01:51:07,929 - INFO - dice=0.9562934791187989 on 1400
# 2019-09-30 01:51:26,821 - INFO - dice=0.9550476565643492 on 1600
# 2019-09-30 01:51:45,597 - INFO - dice=0.9536821359785382 on 1800
# 2019-09-30 01:52:20,590 - INFO - dice=0.9881141291421964 on 200
# 2019-09-30 01:52:39,387 - INFO - dice=0.988909674090486 on 400
# 2019-09-30 01:52:58,144 - INFO - dice=0.9906736179941543 on 600
# 2019-09-30 01:53:16,936 - INFO - dice=0.9910713904682992 on 800
# 2019-09-30 01:53:35,837 - INFO - dice=0.9909922843482397 on 1000
# 2019-09-30 01:53:54,674 - INFO - dice=0.9913900568223846 on 1200
# 2019-09-30 01:54:13,522 - INFO - dice=0.990575638908663 on 1400
# 2019-09-30 01:54:32,367 - INFO - dice=0.989630436082064 on 1600
# 2019-09-30 01:54:51,199 - INFO - dice=0.989630436082064 on 1800
# 2019-09-30 01:55:26,237 - INFO - dice=0.8596845045258205 on 200
# 2019-09-30 01:55:44,916 - INFO - dice=0.8610052372448306 on 400
# 2019-09-30 01:56:03,563 - INFO - dice=0.8645342584097025 on 600
# 2019-09-30 01:56:22,196 - INFO - dice=0.8647812432362187 on 800
# 2019-09-30 01:56:40,980 - INFO - dice=0.8649114159436079 on 1000
# 2019-09-30 01:56:59,736 - INFO - dice=0.8645497446484278 on 1200
# 2019-09-30 01:57:18,424 - INFO - dice=0.8640880117047903 on 1400
# 2019-09-30 01:57:37,165 - INFO - dice=0.8651323516763354 on 1600
# 2019-09-30 01:57:55,841 - INFO - dice=0.8621210334182574 on 1800
# 2019-09-30 01:58:30,891 - INFO - dice=0.9852043297166853 on 200
# 2019-09-30 01:58:49,686 - INFO - dice=0.98560210219083 on 400
# 2019-09-30 01:59:08,538 - INFO - dice=0.98560210219083 on 600
# 2019-09-30 01:59:27,365 - INFO - dice=0.98560210219083 on 800
# 2019-09-30 01:59:46,206 - INFO - dice=0.98560210219083 on 1000
# 2019-09-30 02:00:05,031 - INFO - dice=0.98560210219083 on 1200
# 2019-09-30 02:00:23,860 - INFO - dice=0.98560210219083 on 1400
# 2019-09-30 02:00:42,676 - INFO - dice=0.98560210219083 on 1600
# 2019-09-30 02:01:01,502 - INFO - dice=0.9859998746649747 on 1800
# 2019-09-30 02:01:01,504 - INFO - holdout dice=0.9518638535409372
# 2019-10-02 01:21:35,811 - INFO - dice=0.9684700476579294 on 0.58
# 2019-10-02 01:51:25,794 - INFO - dice=0.9913900568223846 on 0.0
# 2019-10-02 03:04:01,673 - INFO - dice=0.8718925026751589 on 0.47000000000000003
# 2019-10-02 03:32:40,265 - INFO - dice=0.986167751879789 on 0.2
# best_ckpt=18, fold=0, GAMMA=0.7
# 2019-10-01 01:04:15,468 - INFO - Mean train loss: 0.00855
# 2019-10-01 01:06:41,910 - INFO - Mean valid loss: 0.0086
# 2019-10-01 01:06:41,910 - INFO - Mean valid score: 0.9568000000000001
# ===============
import os
import gc
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
device = "cuda:0"
IMG_SIZE = (1600, 256)
CLR_CYCLE = 3
BATCH_SIZE = 32
EPOCHS = 137
FOLD_ID = 1
GAMMA = 0.8
EXP_ID = "exp77_unet_resnet"
CLASSIFICATION = True
base_ckpt = 8
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
            ], p=0.5),
            OneOf([
                GaussNoise(p=0.5),
            ], p=0.5),
            ShiftScaleRotate(rotate_limit=20, p=0.5),
        ])
        val_augmentation = None

        train_dataset = SeverDataset(train_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                    transforms=train_augmentation, crop_rate=1.0, class_y=y_train, gamma=GAMMA)
        val_dataset = SeverDataset(val_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                  transforms=val_augmentation, gamma=GAMMA)
        train_sampler = MaskProbSampler(train_df, demand_non_empty_proba=0.6)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        del train_df, val_df, df, train_dataset, val_dataset
        gc.collect()

    with timer('create model'):
        model = smp.Unet('resnet34', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
                         decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
                         classification=CLASSIFICATION, attention_type="cbam", center=True)
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
        model = torch.nn.DataParallel(model)

    with timer('train'):
        train_losses = []
        valid_losses = []

        best_model_loss = 999
        best_model_ep = 0
        best_model_score = 0
        checkpoint = base_ckpt+1

        for epoch in range(48, EPOCHS + 1):
            seed = seed + epoch
            seed_torch(seed)

            LOGGER.info("Starting {} epoch...".format(epoch))
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, cutmix_prob=0.0,
                                      classification=CLASSIFICATION)
            train_losses.append(tr_loss)
            LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

            valid_loss, val_score = validate(model, val_loader, criterion, device, classification=CLASSIFICATION)
            valid_losses.append(valid_loss)
            LOGGER.info('Mean valid loss: {}'.format(round(valid_loss, 5)))
            LOGGER.info('Mean valid score: {}'.format(round(val_score, 5)))

            scheduler.step()

            if val_score > best_model_score:
                torch.save(model.module.state_dict(), 'models/{}_fold{}_ckpt{}_score.pth'.format(EXP_ID, FOLD_ID, checkpoint))
                best_model_score = val_score
                best_model_ep_score = epoch

            if valid_loss < best_model_loss:
                torch.save(model.module.state_dict(), 'models/{}_fold{}_ckpt{}.pth'.format(EXP_ID, FOLD_ID, checkpoint))
                best_model_loss = valid_loss
                best_model_ep = epoch

            if epoch % (CLR_CYCLE * 2) == CLR_CYCLE * 2 - 1:
                torch.save(model.module.state_dict(), 'models/{}_fold{}_latest.pth'.format(EXP_ID, FOLD_ID))
                LOGGER.info('Best valid loss: {} on epoch={}'.format(round(best_model_loss, 5), best_model_ep))
                LOGGER.info('Best valid score: {} on epoch={}'.format(round(best_model_score, 5), best_model_ep_score))
                checkpoint += 1
                best_model_loss = 999
                best_model_score = 0

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
