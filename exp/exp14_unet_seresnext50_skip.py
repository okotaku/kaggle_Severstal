# ===============
# Unet+Resnet34
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
from datasets import SeverDataset
from logger import setup_logger, LOGGER
from trainer import train_one_epoch, validate
from scheduler import GradualWarmupScheduler
sys.path.append("../segmentation_models_pytorch/")
import segmentation_models_pytorch as smp


# ===============
# Constants
# ===============
IMG_DIR = "../input/severstal-steel-defect-detection/train_images/"
LOGGER_PATH = "log.txt"
FOLD_PATH = "../input/make-folds-severstal/severstal_folds01.csv"
ID_COLUMNS = "ImageId"
N_CLASSES = 4


# ===============
# Settings
# ===============
SEED = np.random.randint(100000)
device = "cuda:0"
IMG_SIZE = (800, 256)
CLR_CYCLE = 3
BATCH_SIZE = 8
EPOCHS = 11
FOLD_ID = 0
EXP_ID = "exp14_unet_seresnext50"
#model_path = "../input/sev-exp6-unetpp-seresnext50-fold0-3-256-skip/exp6_unet_seresnext_fold0_ckpt1.pth"

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)
LOGGER.info("seed={}".format(SEED))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def main():
    with timer('load data'):
        df = pd.read_csv(FOLD_PATH)

    with timer('preprocessing'):
        train_df, val_df = df[df.fold_id != FOLD_ID], df[df.fold_id == FOLD_ID]

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
            ], p=0.5)
        ])
        val_augmentation = None

        train_dataset = SeverDataset(train_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                    transforms=train_augmentation, crop_rate=1.0)
        val_dataset = SeverDataset(val_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                  transforms=val_augmentation)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        del train_df, val_df, df, train_dataset, val_dataset
        gc.collect()

    with timer('create model'):
        model = smp.Unet('se_resnext50_32x4d', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
                         decoder_semodule=True, h_columns=False, skip=True)
        #model.load_state_dict(torch.load(model_path))
        model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=CLR_CYCLE, eta_min=3e-5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=CLR_CYCLE*2, after_scheduler=scheduler_cosine)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    with timer('train'):
        train_losses = []
        valid_losses = []

        best_model_loss = 999
        best_model_ep = 0
        checkpoint = 0

        for epoch in range(1, EPOCHS + 1):
            SEED = SEED + epoch
            if epoch % (CLR_CYCLE * 2) == 0:
                checkpoint += 1
                best_model_loss = 999

            LOGGER.info("Starting {} epoch...".format(epoch))
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, cutmix_prob=0.0)
            train_losses.append(tr_loss)
            LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

            valid_loss = validate(model, val_loader, criterion, device)
            valid_losses.append(valid_loss)
            LOGGER.info('Mean valid loss: {}'.format(round(valid_loss, 5)))

            scheduler.step()

            if valid_loss < best_model_loss:
                torch.save(model.state_dict(), 'models/{}_fold{}_ckpt{}.pth'.format(EXP_ID, FOLD_ID, checkpoint))
                best_model_loss = valid_loss
                best_model_ep = epoch
                #np.save("val_pred.npy", val_pred)

            #del val_pred
            gc.collect()

    xs = list(range(1, len(train_losses) + 1))
    plt.plot(xs, train_losses, label='Train loss')
    plt.plot(xs, valid_losses, label='Val loss')
    plt.legend()
    plt.xticks(xs)
    plt.xlabel('Epochs')
    plt.savefig("loss.png")


if __name__ == '__main__':
    main()