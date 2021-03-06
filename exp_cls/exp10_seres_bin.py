# ===============
# best_ckpt=5, fold=0
# 2019-10-04 18:22:41,960 - INFO - Mean train loss: 0.04755
# 2019-10-04 18:23:36,176 - INFO - Mean valid loss: 0.06908
# 2019-10-04 18:23:36,177 - INFO - Mean valid score: (array([0.96195203, 0.9862069 ]), array([0.98475868, 0.96549137]), array([0.97322176, 0.9757392 ]), array([1181, 1333]))
# 2019-10-04 18:24:30,378 - INFO - Mean EMA valid loss: 0.06122
# best_ckpt=5, fold=1
# 2019-10-04 17:38:35,582 - INFO - Mean train loss: 0.06278
# 2019-10-04 17:39:24,887 - INFO - Mean valid loss: 0.09282
# 2019-10-04 17:39:24,888 - INFO - Mean valid score: (array([0.953125 , 0.9830639]), array([0.98137172, 0.95727136]), array([0.96704214, 0.9699962 ]), array([1181, 1334]))
# 2019-10-04 17:40:14,158 - INFO - Mean EMA valid loss: 0.06325
# best_ckpt=4, fold=2
# 2019-10-04 10:19:18,386 - INFO - Mean train loss: 0.06578
# 2019-10-04 10:20:08,519 - INFO - Mean valid loss: 0.08781
# 2019-10-04 10:20:08,520 - INFO - Mean valid score: (array([0.95778146, 0.98238897]), array([0.98050847, 0.96176912]), array([0.96901173, 0.9719697 ]), array([1180, 1334]))
# 2019-10-04 10:20:58,785 - INFO - Mean EMA valid loss: 0.06684
# best_ckpt=3, fold=3
# 2019-10-06 09:45:46,396 - INFO - Mean train loss: 0.09002
# 2019-10-06 09:46:36,796 - INFO - Mean valid loss: 0.06749
# 2019-10-06 09:46:36,797 - INFO - Mean valid score: (array([0.9673913 , 0.98253607]), array([0.98050847, 0.97074269]), array([0.97390572, 0.97660377]), array([1180, 1333]))
# 2019-10-06 09:47:29,024 - INFO - Mean EMA valid loss: 0.0578
# best_ckpt=4, fold=4
# 2019-10-06 10:16:55,191 - INFO - Mean train loss: 0.07664
# 2019-10-06 10:17:46,428 - INFO - Mean valid loss: 0.09433
# 2019-10-06 10:17:46,429 - INFO - Mean valid score: (array([0.97529813, 0.97384155]), array([0.97033898, 0.97822823]), array([0.97281223, 0.97602996]), array([1180, 1332]))
# 2019-10-06 10:18:37,802 - INFO - Mean EMA valid loss: 0.06616
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
from sklearn.metrics import precision_recall_fscore_support
from albumentations import *
import torch
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append("../severstal-src/")
from util import seed_torch, search_threshold
from losses import FocalLovaszLoss
from datasets import SeverCLSDataset, MaskProbSampler
from logger import setup_logger, LOGGER
from cls_trainer import train_one_epoch, validate
from scheduler import GradualWarmupScheduler
from cls_models import SEResNext
sys.path.append("../")
from sync_batchnorm import convert_model


# ===============
# Constants
# ===============
IMG_DIR = "../input/train_images/"
LOGGER_PATH = "log.txt"
FOLD_PATH = "../input/severstal_folds01.csv"
ID_COLUMNS = "ImageId"
N_CLASSES = 1


# ===============
# Settings
# ===============
SEED = np.random.randint(100000)
device = "cuda"
IMG_SIZE = (1600, 256)
CLR_CYCLE = 3
BATCH_SIZE = 32
EPOCHS = 59
FOLD_ID = 3
EXP_ID = "cls_exp10_seresnext"
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

        train_dataset = SeverCLSDataset(train_df, IMG_DIR, IMG_SIZE, N_CLASSES, y_train, id_colname=ID_COLUMNS,
                                    transforms=train_augmentation, crop_rate=1.0)
        val_dataset = SeverCLSDataset(val_df, IMG_DIR, IMG_SIZE, N_CLASSES, y_val, id_colname=ID_COLUMNS,
                                  transforms=val_augmentation)
        #train_sampler = MaskProbSampler(train_df, demand_non_empty_proba=0.6)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

        del train_df, val_df, df, train_dataset, val_dataset
        gc.collect()

    with timer('create model'):
        model = SEResNext(num_classes=N_CLASSES)
        #model = convert_model(model)
        if base_model is not None:
            model.load_state_dict(torch.load(base_model))
        model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)
        if base_model is None:
            scheduler_cosine = CosineAnnealingLR(optimizer, T_max=CLR_CYCLE, eta_min=1e-5)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=CLR_CYCLE*2, after_scheduler=scheduler_cosine)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=CLR_CYCLE, eta_min=1e-5)


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
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                      ema_model=ema_model, ema_decay=ema_decay)
            train_losses.append(tr_loss)
            LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

            valid_loss, y_pred, y_true = validate(model, val_loader, criterion, device)
            scores = precision_recall_fscore_support(y_true, y_pred>0.5)
            valid_losses.append(valid_loss)
            LOGGER.info('Mean valid loss: {}'.format(round(valid_loss, 5)))
            LOGGER.info('Mean valid score: {}'.format(scores))

            if EMA and epoch >= EMA_START:
                ema_valid_loss, y_pred_ema, _ = validate(ema_model, val_loader, criterion, device)
                LOGGER.info('Mean EMA valid loss: {}'.format(round(ema_valid_loss, 5)))

                if ema_valid_loss < best_model_ema_loss:
                    torch.save(ema_model.module.state_dict(),
                               'models/{}_fold{}_ckpt{}_ema.pth'.format(EXP_ID, FOLD_ID, checkpoint))
                    best_model_ema_loss = ema_valid_loss
                    np.save("y_pred_ema_ckpt{}.npy".format(checkpoint), y_pred_ema)

            scheduler.step()

            if valid_loss < best_model_loss:
                torch.save(model.module.state_dict(), 'models/{}_fold{}_ckpt{}.pth'.format(EXP_ID, FOLD_ID, checkpoint))
                np.save("y_pred_ckpt{}.npy".format(checkpoint), y_pred)
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
                best_model_ema_loss = 999

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
