# ===============
# best_ckpt=8, fold=3
# 2019-09-26 08:28:58,887 - INFO - Train loss on step 500 was 0.00956
# 2019-09-26 08:29:36,765 - INFO - Mean train loss: 0.00959
# 2019-09-26 08:30:37,860 - INFO - Mean valid loss: 0.00891
# 2019-09-26 11:03:52,345 - INFO - dice=0.9608533816529758 on 200
# 2019-09-26 11:04:11,042 - INFO - dice=0.9622246940878061 on 400
# 2019-09-26 11:04:29,639 - INFO - dice=0.962590984460253 on 600
# 2019-09-26 11:04:48,320 - INFO - dice=0.9618099999714957 on 800
# 2019-09-26 11:05:06,967 - INFO - dice=0.9603123487217322 on 1000
# 2019-09-26 11:05:25,638 - INFO - dice=0.958792292149539 on 1200
# 2019-09-26 11:05:44,348 - INFO - dice=0.9566235744459466 on 1400
# 2019-09-26 11:06:02,977 - INFO - dice=0.9539530041113559 on 1600
# 2019-09-26 11:06:21,577 - INFO - dice=0.9520858316542717 on 1800
# 2019-09-26 11:06:56,625 - INFO - dice=0.9887559622307402 on 200
# 2019-09-26 11:07:15,321 - INFO - dice=0.989895294313552 on 400
# 2019-09-26 11:07:33,964 - INFO - dice=0.990381136824443 on 600
# 2019-09-26 11:07:52,630 - INFO - dice=0.9915749291045862 on 800
# 2019-09-26 11:08:11,258 - INFO - dice=0.991331090681408 on 1000
# 2019-09-26 11:08:29,905 - INFO - dice=0.9911418232994673 on 1200
# 2019-09-26 11:08:48,617 - INFO - dice=0.9905835935946448 on 1400
# 2019-09-26 11:09:07,272 - INFO - dice=0.9904576371766234 on 1600
# 2019-09-26 11:09:25,942 - INFO - dice=0.9899412314434045 on 1800
# 2019-09-26 11:10:00,746 - INFO - dice=0.8576688220190247 on 200
# 2019-09-26 11:10:19,337 - INFO - dice=0.8592434027874902 on 400
# 2019-09-26 11:10:37,852 - INFO - dice=0.859815176659575 on 600
# 2019-09-26 11:10:56,413 - INFO - dice=0.8601637889913384 on 800
# 2019-09-26 11:11:14,830 - INFO - dice=0.861251819831836 on 1000
# 2019-09-26 11:11:33,231 - INFO - dice=0.8605155309689223 on 1200
# 2019-09-26 11:11:51,732 - INFO - dice=0.8597644290625278 on 1400
# 2019-09-26 11:12:10,277 - INFO - dice=0.8575801148997854 on 1600
# 2019-09-26 11:12:28,801 - INFO - dice=0.8569980313634737 on 1800
# 2019-09-26 11:13:03,759 - INFO - dice=0.9825777331212298 on 200
# 2019-09-26 11:13:22,390 - INFO - dice=0.9825777331212298 on 400
# 2019-09-26 11:13:41,092 - INFO - dice=0.9829756638812774 on 600
# 2019-09-26 11:13:59,756 - INFO - dice=0.9829756638812774 on 800
# 2019-09-26 11:14:18,413 - INFO - dice=0.9829756638812774 on 1000
# 2019-09-26 11:14:37,051 - INFO - dice=0.9829447081634318 on 1200
# 2019-09-26 11:14:55,671 - INFO - dice=0.9829447081634318 on 1400
# 2019-09-26 11:15:14,337 - INFO - dice=0.9836043565708306 on 1600
# 2019-09-26 11:15:32,971 - INFO - dice=0.9834299963161914 on 1800
# best_ckpt=12, fold=3
# 2019-09-26 11:28:33,521 - INFO - Mean train loss: 0.00871
# 2019-09-26 11:29:35,461 - INFO - Mean valid loss: 0.00883
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
from trainer import train_one_epoch_dsv, validate_dsv
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
BATCH_SIZE = 16
EPOCHS = 101 #101
FOLD_ID = 3
EXP_ID = "exp68_unetpp_resnet"
CLASSIFICATION = True
base_ckpt = 0
base_model = None
#base_model = "models/{}_fold{}_latest.pth".format(EXP_ID, FOLD_ID)

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
        model = smp.UnetPP('resnet34', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
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

        for epoch in range(1, EPOCHS + 1):
            seed = seed + epoch
            seed_torch(seed)

            LOGGER.info("Starting {} epoch...".format(epoch))
            tr_loss = train_one_epoch_dsv(model, train_loader, criterion, optimizer, device, classification=CLASSIFICATION)
            train_losses.append(tr_loss)
            LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

            valid_loss = validate_dsv(model, val_loader, criterion, device, classification=CLASSIFICATION)
            valid_losses.append(valid_loss)
            LOGGER.info('Mean valid loss: {}'.format(round(valid_loss, 5)))

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
