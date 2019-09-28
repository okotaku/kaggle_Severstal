# ===============
# best_ckpt=15, fold=0
# 2019-09-27 17:46:30,740 - INFO - Mean train loss: 0.0117
# 2019-09-27 17:47:29,708 - INFO - Mean valid loss: 0.01085
# 2019-09-27 17:48:28,817 - INFO - Mean EMA valid loss: 0.01037
# 2019-09-28 00:42:04,990 - INFO - dice=0.9614316420292547 on 200
# 2019-09-28 00:42:21,054 - INFO - dice=0.9634205043999786 on 400
# 2019-09-28 00:42:36,973 - INFO - dice=0.9640378042145211 on 600
# 2019-09-28 00:42:52,794 - INFO - dice=0.9639386334682853 on 800
# 2019-09-28 00:43:08,681 - INFO - dice=0.9639835302020137 on 1000
# 2019-09-28 00:43:24,511 - INFO - dice=0.9620253013688154 on 1200
# 2019-09-28 00:43:40,416 - INFO - dice=0.9597985296300837 on 1400
# 2019-09-28 00:43:56,461 - INFO - dice=0.9571689659732502 on 1600
# 2019-09-28 00:44:12,289 - INFO - dice=0.955208550029795 on 1800
# 2019-09-28 00:44:41,674 - INFO - dice=0.8648110138454254 on 200
# 2019-09-28 00:44:57,414 - INFO - dice=0.867741221169205 on 400
# 2019-09-28 00:45:13,419 - INFO - dice=0.86913936487825 on 600
# 2019-09-28 00:45:29,179 - INFO - dice=0.8692236702352804 on 800
# 2019-09-28 00:45:44,933 - INFO - dice=0.8691227635696854 on 1000
# 2019-09-28 00:46:00,790 - INFO - dice=0.8695890395635283 on 1200
# 2019-09-28 00:46:16,675 - INFO - dice=0.8691353399983928 on 1400
# 2019-09-28 00:46:32,550 - INFO - dice=0.8674578977788835 on 1600
# 2019-09-28 00:46:48,251 - INFO - dice=0.8665808302892621 on 1800
# 2019-09-28 00:47:17,963 - INFO - dice=0.9842251192599204 on 200
# 2019-09-28 00:47:33,943 - INFO - dice=0.9842251192599204 on 400
# 2019-09-28 00:47:49,944 - INFO - dice=0.9842251192599204 on 600
# 2019-09-28 00:48:05,964 - INFO - dice=0.9846228917340651 on 800
# 2019-09-28 00:48:21,782 - INFO - dice=0.9848820713297014 on 1000
# 2019-09-28 00:48:37,674 - INFO - dice=0.9848820713297014 on 1200
# 2019-09-28 00:48:53,505 - INFO - dice=0.9848820713297014 on 1400
# 2019-09-28 00:49:09,523 - INFO - dice=0.9848820713297014 on 1600
# 2019-09-28 00:49:25,499 - INFO - dice=0.9848820713297014 on 1800
# 2019-09-28 00:49:25,501 - INFO - holdout dice=0.9355571505495862
# best_ckpt=16, fold=0
# 2019-09-27 18:09:01,702 - INFO - Mean train loss: 0.01077
# 2019-09-27 18:10:00,754 - INFO - Mean valid loss: 0.01045
# 2019-09-27 18:10:59,880 - INFO - Mean EMA valid loss: 0.01048

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
N_CLASSES = 3


# ===============
# Settings
# ===============
SEED = np.random.randint(100000)
device = "cuda:0"
IMG_SIZE = (1600, 256)
CLR_CYCLE = 3
BATCH_SIZE = 32
EPOCHS = 155
FOLD_ID = 0
EXP_ID = "exp74_unet_resnet"
CLASSIFICATION = True
EMA = True
EMA_START = 6
base_ckpt = 10
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
        df.drop("EncodedPixels_2", axis=1, inplace=True)
        df = df.rename(columns={"EncodedPixels_3": "EncodedPixels_2"})
        df = df.rename(columns={"EncodedPixels_4": "EncodedPixels_3"})
        y1 = (df.EncodedPixels_1 != "-1").astype("float32").values.reshape(-1, 1)
        y2 = (df.EncodedPixels_2 != "-1").astype("float32").values.reshape(-1, 1)
        y3 = (df.EncodedPixels_3 != "-1").astype("float32").values.reshape(-1, 1)
        #y4 = (df.EncodedPixels_4 != "-1").astype("float32").values.reshape(-1, 1)
        y = np.concatenate([y1, y2, y3], axis=1)

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
        model = smp.Unet('resnet34', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
                         decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
                         classification=CLASSIFICATION, attention_type="cbam", center=True, mode="train")
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
            ema_model = torch.nn.DataParallel(ema_model)
        else:
            ema_model = None
        model = torch.nn.DataParallel(model)

    with timer('train'):
        train_losses = []
        valid_losses = []

        best_model_loss = 999
        best_model_ema_loss = 999
        best_model_ep = 0
        ema_decay = 0
        checkpoint = base_ckpt + 1

        for epoch in range(60, EPOCHS + 1):
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
                    best_model_ema_loss = 999
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
