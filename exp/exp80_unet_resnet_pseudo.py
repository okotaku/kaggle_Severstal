# ===============
# best_ckpt=11, fold=0
# 2019-10-01 22:36:13,985 - INFO - Mean train loss: 0.00774
# 2019-10-01 22:38:37,824 - INFO - Mean valid loss: 0.0084
# 2019-10-01 22:38:37,824 - INFO - Mean valid score: 0.9487500000000001
# best_score_ckpt=16
# 2019-10-02 03:20:24,883 - INFO - Mean train loss: 0.0064
# 2019-10-02 03:22:48,085 - INFO - Mean valid loss: 0.00868
# 2019-10-02 03:22:48,086 - INFO - Mean valid score: 0.95293
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
FOLD_ID = 0
EXP_ID = "exp80_unet_resnet"
CLASSIFICATION = True
base_ckpt = 0
base_model = None
#base_model = "models/{}_fold{}_latest.pth".format(EXP_ID, FOLD_ID)
PSEUDO_PATH = "../input/severstal_pseudo01.csv"

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
        checkpoint = base_ckpt+1

        for epoch in range(1, EPOCHS + 1):
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
