# ===============
# best_ckpt=10, fold=0
# 2019-09-27 02:31:28,952 - INFO - dice=0.9601060367725855 on 200
# 2019-09-27 02:31:44,804 - INFO - dice=0.9623477185769066 on 400
# 2019-09-27 02:32:00,480 - INFO - dice=0.9625510089471736 on 600
# 2019-09-27 02:32:16,097 - INFO - dice=0.9637142514261622 on 800
# 2019-09-27 02:32:31,889 - INFO - dice=0.9630842514091559 on 1000
# 2019-09-27 02:32:47,574 - INFO - dice=0.9621581444438071 on 1200
# 2019-09-27 02:33:03,491 - INFO - dice=0.9609585676221292 on 1400
# 2019-09-27 02:33:19,240 - INFO - dice=0.9592675801162609 on 1600
# 2019-09-27 02:33:35,059 - INFO - dice=0.9581930090576831 on 1800
# 2019-09-27 02:34:04,773 - INFO - dice=0.9800192591333302 on 200
# 2019-09-27 02:34:20,667 - INFO - dice=0.9800192591333302 on 400
# 2019-09-27 02:34:36,465 - INFO - dice=0.9810003654343227 on 600
# 2019-09-27 02:34:52,204 - INFO - dice=0.982193682856757 on 800
# 2019-09-27 02:35:07,995 - INFO - dice=0.9845803177016258 on 1000
# 2019-09-27 02:35:23,635 - INFO - dice=0.9873647250206393 on 1200
# 2019-09-27 02:35:39,398 - INFO - dice=0.9884884840833796 on 1400
# 2019-09-27 02:35:55,224 - INFO - dice=0.9892840290316692 on 1600
# 2019-09-27 02:36:10,888 - INFO - dice=0.9892840290316692 on 1800
# 2019-09-27 02:36:40,239 - INFO - dice=0.8280698225563331 on 200
# 2019-09-27 02:36:55,914 - INFO - dice=0.8359623804974472 on 400
# 2019-09-27 02:37:11,658 - INFO - dice=0.8387053009047434 on 600
# 2019-09-27 02:37:27,377 - INFO - dice=0.8414281605538899 on 800
# 2019-09-27 02:37:42,933 - INFO - dice=0.8430728919580207 on 1000
# 2019-09-27 02:37:58,581 - INFO - dice=0.8444671125054972 on 1200
# 2019-09-27 02:38:14,113 - INFO - dice=0.8453136656317431 on 1400
# 2019-09-27 02:38:29,855 - INFO - dice=0.84423511579418 on 1600
# 2019-09-27 02:38:45,561 - INFO - dice=0.8434123634566442 on 1800
# 2019-09-27 02:39:15,097 - INFO - dice=0.9806489743541847 on 200
# 2019-09-27 02:39:30,891 - INFO - dice=0.9810467468283296 on 400
# 2019-09-27 02:39:46,756 - INFO - dice=0.9810467468283296 on 600
# 2019-09-27 02:40:02,518 - INFO - dice=0.9810467468283296 on 800
# 2019-09-27 02:40:18,339 - INFO - dice=0.9810467468283296 on 1000
# 2019-09-27 02:40:33,926 - INFO - dice=0.9810467468283296 on 1200
# 2019-09-27 02:40:49,540 - INFO - dice=0.9810467468283296 on 1400
# 2019-09-27 02:41:05,353 - INFO - dice=0.9810467468283296 on 1600
# 2019-09-27 02:41:20,963 - INFO - dice=0.9810467468283296 on 1800
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
from losses import ComboLoss
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
EPOCHS = 101
FOLD_ID = 0
EXP_ID = "exp70_unet_resnet"
CLASSIFICATION = True
base_ckpt = 9
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

        criterion = ComboLoss({'bce': 4,
                        'dice': 1,
                        'focal': 3}, channel_weights=[1, 1, 1, 1])
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

        for epoch in range(54, EPOCHS + 1):
            seed = seed + epoch
            seed_torch(seed)

            LOGGER.info("Starting {} epoch...".format(epoch))
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, cutmix_prob=0.0,
                                      classification=CLASSIFICATION)
            train_losses.append(tr_loss)
            LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

            valid_loss = validate(model, val_loader, criterion, device, classification=CLASSIFICATION)
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
