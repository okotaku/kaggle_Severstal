# ===============
# best_ckpt=11, fold=0
# 2019-09-22 20:21:17,224 - INFO - Mean train loss: 0.00841
# 2019-09-22 20:22:24,301 - INFO - Mean valid loss: 0.00855
# 2019-09-22 20:23:32,791 - INFO - Mean EMA valid loss: 0.00803
# 2019-09-23 06:08:22,377 - INFO - dice=0.9667818835644745 on 800
# 2019-09-23 06:09:54,324 - INFO - dice=0.9899488094674251 on 800
# 2019-09-23 06:11:25,569 - INFO - dice=0.8673755849027167 on 800
# 2019-09-23 06:12:25,661 - INFO - dice=0.9847439999198341 on 400
# 2019-09-23 13:12:00,649 - INFO - dice=0.9691685184093431 on 0.17
# 2019-09-23 13:40:14,458 - INFO - dice=0.99034658194157 on 0.08
# 2019-09-23 14:08:25,024 - INFO - dice=0.8754584861686093 on 0.65
# 2019-09-23 14:36:54,637 - INFO - dice=0.9859557605689044 on 0.42
# best_ckpt=17, fold=1
# 2019-09-23 01:15:26,062 - INFO - Mean train loss: 0.00707
# 2019-09-23 01:16:32,374 - INFO - Mean valid loss: 0.00871
# 2019-09-23 01:17:39,554 - INFO - Mean EMA valid loss: 0.00833
# best_ckpt=13, fold=2
# 2019-09-22 21:40:58,876 - INFO - Mean train loss: 0.00805
# 2019-09-22 21:42:04,469 - INFO - Mean valid loss: 0.00859
# 2019-09-22 21:43:08,734 - INFO - Mean EMA valid loss: 0.008121
# best_ckpt=13, fold=3
# 2019-09-22 21:43:32,186 - INFO - Mean train loss: 0.00722
# 2019-09-22 21:44:33,931 - INFO - Mean valid loss: 0.00869
# 2019-09-22 21:45:36,636 - INFO - Mean EMA valid loss: 0.00859
# best_ckpt=9, fold=4
# 2019-09-23 02:31:39,970 - INFO - Mean train loss: 0.00865
# 2019-09-23 02:32:47,933 - INFO - Mean valid loss: 0.0085
# 2019-09-23 02:33:57,067 - INFO - Mean EMA valid loss: 0.00809
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
EPOCHS = 119
FOLD_ID = 1
EXP_ID = "exp61_unet_resnet"
CLASSIFICATION = True
EMA = True
EMA_START = 6
base_ckpt = 0
base_model = None
base_model_ema = None
#base_model = "models/{}_fold{}_latest.pth".format(EXP_ID, FOLD_ID)
#base_model_ema = "models/{}_fold{}_latest_ema.pth".format(EXP_ID, FOLD_ID)

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
        LOGGER.info(len(df))
        df = df[df.sum_target != 0].reset_index(drop=True)
        LOGGER.info(len(df))
        y1 = (df.EncodedPixels_1 != "-1").astype("float32").values.reshape(-1, 1)
        y2 = (df.EncodedPixels_2 != "-1").astype("float32").values.reshape(-1, 1)
        y3 = (df.EncodedPixels_3 != "-1").astype("float32").values.reshape(-1, 1)
        y4 = (df.EncodedPixels_4 != "-1").astype("float32").values.reshape(-1, 1)
        y = np.concatenate([y1, y2, y3, y4], axis=1)

    with timer('preprocessing'):
        train_df, val_df = df[df.fold_id != FOLD_ID], df[df.fold_id == FOLD_ID]
        LOGGER.info("len train={}  len val={}".format(len(train_df), len(val_df)))
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
        #train_sampler = MaskProbSampler(train_df, demand_non_empty_proba=0.6)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

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
            {'params': model.encoder.parameters(), 'lr': 3e-4}
        ], eps=1e-4)
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

        for epoch in range(1, EPOCHS + 1):
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
