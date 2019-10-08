# ===============
#
# ===============
import os
import cv2
import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
from albumentations import *
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import sys
sys.path.append("../severstal-src/")
from util import seed_torch, mask2rle, rle2mask
from losses import FocalLovaszLoss
from datasets import SeverDataset, MaskProbSampler
from logger import setup_logger, LOGGER
from scheduler import GradualWarmupScheduler
sys.path.append("../")
import segmentation_models_pytorch as smp
import segmentation_models_pytorch2 as smp_old
from sync_batchnorm import convert_model


class SeverDatasetTest(Dataset):

    def __init__(self,
                 df,
                 img_dir,
                 img_size,
                 n_classes,
                 crop_rate=1.0,
                 id_colname="ImageId",
                 mask_colname=["EncodedPixels_{}".format(i) for i in range(1, 5)],
                 transforms=None,
                 means=[0.485, 0.456, 0.406],
                 stds=[0.229, 0.224, 0.225],
                 p_black_crop=0.0
                 ):
        self.df = df
        self.img_dir = img_dir
        self.img_size = img_size
        self.transforms = transforms
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.id_colname = id_colname
        self.mask_colname = mask_colname
        self.n_classes = n_classes
        self.crop_rate = crop_rate
        self.gamma = 0.8
        self.p_black_crop = p_black_crop

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_path = os.path.join(self.img_dir, img_id)

        img = cv2.imread(img_path)

        if np.random.rand() <= self.p_black_crop:
            mask_img = img > 20
            sum_channel = np.sum(mask_img, 2)
            w_cr = np.where(sum_channel.sum(0) != 0)
            h_cr = np.where(sum_channel.sum(1) != 0)
            img = img[np.min(h_cr):np.max(h_cr) + 1, np.min(w_cr):np.max(w_cr) + 1, :]
            bbox = [np.min(h_cr), np.max(h_cr) + 1, np.min(w_cr), np.max(w_cr) + 1]

        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, self.gamma) * 255.0, 0, 255)
        img_g = cv2.LUT(img, lookUpTable)

        img = cv2.resize(img, self.img_size)
        img_g = cv2.resize(img_g, self.img_size)

        img = self._preprocess(img)
        img_g = self._preprocess(img_g)

        return torch.Tensor(img), torch.Tensor(img_g), img_id

    def _preprocess(self, img_):
        img_ = img_ / 255
        img_ -= self.means
        img_ /= self.stds
        img_ = img_.transpose((2, 0, 1))

        return img_


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
FOLD_ID = 1
EXP_ID = "exp86_unet_resnet"
CLASSIFICATION = True
remove_mask_pixels = [200, 400, 200, 1000]
best_ckpt = 14
base_model = "../exp/models/{}_fold{}_ckpt{}.pth".format(EXP_ID, FOLD_ID, best_ckpt)

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)
LOGGER.info("seed={}".format(SEED))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def post_process(mask, min_size):
    """Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored"""
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1

    if num >= 1:
        sum_predictions = np.sum(predictions)
        if sum_predictions > min_size * 2:
            predictions = np.flipud(np.rot90(predictions, k=1))
            return mask2rle(predictions, 1600, 256)
        else:
            return None
    else:
        return None


def predict(models, test_loader, device):
    preds_rle = []
    ids = []
    with torch.no_grad():
        for step, (features, features_g, img_id) in enumerate(test_loader):
            features = features.to(device)

            for i, m in enumerate(models):
                if i == 0:
                    logits = torch.sigmoid(m(features)[0])
                else:
                    logits += torch.sigmoid(m(features)[0])

            logits = logits / len(models)
            logits = logits.float().cpu().numpy().astype("float64")

            rles = []
            sub_ids = []
            for i, (preds, id_) in enumerate(zip(logits, img_id)):
                for class_, (remove_mask_pixel, p) in enumerate(zip(remove_mask_pixels, preds)):
                    sub_ids.append(id_ + "_{}".format(class_ + 1))
                    if class_ + 1 == 2:
                        rles.append(None)
                    else:
                        rles.append(post_process(p > 0.5, remove_mask_pixel))

            ids.extend(sub_ids)
            preds_rle.extend(rles)

            del features, logits, rles
            gc.collect()

    return preds_rle, ids


def main(seed):
    with timer('load data'):
        df = pd.read_csv(FOLD_PATH)

    with timer('preprocessing'):
        val_df = df[df.fold_id == FOLD_ID]
        val_augmentation = None

        val_dataset = SeverDatasetTest(val_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                  transforms=val_augmentation)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        del val_df, df, val_dataset
        gc.collect()

    with timer('create model'):
        models = []
        model = smp_old.Unet('resnet34', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
                         decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
                         classification=CLASSIFICATION)
        model = convert_model(model)
        if base_model is not None:
            model.load_state_dict(torch.load(base_model))
        model.to(device)
        models.append(model)

    with timer('predict'):
        rles, sub_ids = predict(models, val_loader, device)
        sub_df = pd.DataFrame({'ImageId_ClassId': sub_ids, 'EncodedPixels': rles})
        LOGGER.info(sub_df.head())

        sub_df.to_csv('{}_{}.csv'.format(EXP_ID, FOLD_ID), index=False)


if __name__ == '__main__':
    main(SEED)
