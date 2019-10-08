# ===============
# best_ckpt=11, fold=0
# 2019-10-07 14:41:05,834 - INFO - Mean train loss: 0.00799
# 2019-10-07 14:42:53,281 - INFO - Mean valid loss: 0.00803
# 2019-10-07 14:42:53,281 - INFO - Mean valid score: 0.9503400000000001
# 2019-10-07 10:00:13,706 - INFO - dice=0.9627098122452976 on 100
# 2019-10-07 10:00:34,837 - INFO - dice=0.9651321032226555 on 200
# 2019-10-07 10:00:55,895 - INFO - dice=0.9657563374317423 on 300
# 2019-10-07 10:01:16,753 - INFO - dice=0.9651275302477957 on 400
# 2019-10-07 10:01:37,509 - INFO - dice=0.9640796422626003 on 600
# 2019-10-07 10:01:58,348 - INFO - dice=0.9612654116985395 on 800
# 2019-10-07 10:02:22,550 - INFO - dice=0.8626393496611545 on 100
# 2019-10-07 10:02:46,423 - INFO - dice=0.8644371004668896 on 200
# 2019-10-07 10:03:10,608 - INFO - dice=0.8669505740210292 on 300
# 2019-10-07 10:03:34,885 - INFO - dice=0.8690772348924043 on 400
# 2019-10-07 10:03:58,948 - INFO - dice=0.8711994049195169 on 600
# 2019-10-07 10:04:22,742 - INFO - dice=0.8710416153462411 on 800
# 2019-10-07 10:04:46,617 - INFO - dice=0.8697053545989428 on 1000
# 2019-10-07 10:05:10,531 - INFO - dice=0.8676218779226781 on 1200
# 2019-10-07 10:05:31,321 - INFO - dice=0.9839468584407572 on 100
# 2019-10-07 10:05:52,127 - INFO - dice=0.983951911493907 on 200
# 2019-10-07 10:06:12,884 - INFO - dice=0.9843528548875485 on 300
# 2019-10-07 10:06:33,635 - INFO - dice=0.9843449573602008 on 400
# 2019-10-07 10:06:54,288 - INFO - dice=0.9846955183633653 on 600
# 2019-10-07 10:07:14,940 - INFO - dice=0.9850973161011053 on 800
# 2019-10-07 10:07:35,584 - INFO - dice=0.9851126809073929 on 1000
# 2019-10-07 10:07:56,278 - INFO - dice=0.9854855914091328 on 1200
# 2019-10-07 10:08:16,898 - INFO - dice=0.9855252354653347 on 1400
# 2019-10-07 10:08:37,669 - INFO - dice=0.9855658091429101 on 1600
# 2019-10-07 10:08:58,378 - INFO - dice=0.9854938840821189 on 1800
# best_ckpt=14, fold=1
# 2019-10-07 12:14:21,251 - INFO - Mean train loss: 0.00749
# 2019-10-07 12:16:30,290 - INFO - Mean valid loss: 0.00873
# 2019-10-07 12:16:30,291 - INFO - Mean valid score: 0.8947900000000001
# best_ckpt=14, fold=2
# 2019-10-07 10:18:41,873 - INFO - Mean train loss: 0.00773
# 2019-10-07 10:20:26,477 - INFO - Mean valid loss: 0.00848
# 2019-10-07 10:20:26,477 - INFO - Mean valid score: 0.95749
# best_ckpt=13, fold=3
# 2019-10-08 04:36:07,176 - INFO - Mean train loss: 0.00896
# 2019-10-08 04:38:13,457 - INFO - Mean valid loss: 0.00883
# 2019-10-08 04:38:13,457 - INFO - Mean valid score: 0.9564
# best_ckpt=13, fold=4
# 2019-10-07 22:16:44,299 - INFO - Mean train loss: 0.00754
# 2019-10-07 22:18:52,193 - INFO - Mean valid loss: 0.00808
# 2019-10-07 22:18:52,193 - INFO - Mean valid score: 0.95435
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
import segmentation_models_pytorch2 as smp_old
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
EPOCHS = 101
FOLD_ID = 3
EXP_ID = "exp86_unet_resnet"
CLASSIFICATION = True
base_ckpt = 5
base_model = None
#base_model = "models/{}_fold{}_latest.pth".format(EXP_ID, FOLD_ID)
PSEUDO_PATH = "../input/severstal_pseudo02.csv"

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
        df = df.append(pd.read_csv(PSEUDO_PATH)).reset_index(drop=True)
        for c in ["EncodedPixels_1", "EncodedPixels_2", "EncodedPixels_3", "EncodedPixels_4"]:
            df[c] = df[c].astype(str)
        df["fold_id"] = df["fold_id"].fillna(FOLD_ID+1)
        y = (df.sum_target != 0).astype("float32").values

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
                                    transforms=train_augmentation, crop_rate=1.0, class_y=y_train)
        val_dataset = SeverDataset(val_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                  transforms=val_augmentation)
        train_sampler = MaskProbSampler(train_df, demand_non_empty_proba=0.6)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        del train_df, val_df, df, train_dataset, val_dataset
        gc.collect()

    with timer('create model'):
        #model = smp.Unet('resnet34', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
        #                 decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
        #                 classification=CLASSIFICATION, attention_type="cbam", center=True)
        model = smp_old.Unet('resnet34', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
                         decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
                         classification=CLASSIFICATION)
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
        checkpoint = base_ckpt+1

        for epoch in range(30, EPOCHS + 1):
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

            if valid_loss < best_model_loss:
                torch.save(model.module.state_dict(), 'models/{}_fold{}_ckpt{}.pth'.format(EXP_ID, FOLD_ID, checkpoint))
                best_model_loss = valid_loss
                best_model_ep = epoch
                #np.save("val_pred.npy", val_pred)

            if epoch % (CLR_CYCLE * 2) == CLR_CYCLE * 2 - 1:
                torch.save(model.module.state_dict(), 'models/{}_fold{}_latest.pth'.format(EXP_ID, FOLD_ID))
                LOGGER.info('Best valid loss: {} on epoch={}'.format(round(best_model_loss, 5), best_model_ep))
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
