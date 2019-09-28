import os
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager
from albumentations import *
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys

sys.path.append("../severstal-src/")
from util import seed_torch, search_threshold
from losses import FocalLovaszLoss
from logger import setup_logger, LOGGER
from scheduler import GradualWarmupScheduler
import cls_models

sys.path.append("../")
import segmentation_models_pytorch as smp
import segmentation_models_pytorch2 as smp_old

import os
import cv2
import torch
import random
import pydicom
import numpy as np
from torch.utils.data import Dataset
from util import rle2mask


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
                 stds=[0.229, 0.224, 0.225]
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

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_path = os.path.join(self.img_dir, img_id)

        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)

        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented['image']

        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))

        return torch.Tensor(img), img_id


def predict_dsv(models, test_loader, device):
    preds_rle = []
    ids = []
    with torch.no_grad():
        for step, (features, img_id) in enumerate(test_loader):
            features = features.to(device)

            for i, m in enumerate(models):
                if i == 0:
                    logits = torch.sigmoid(m(features)[0])
                else:
                    logits += torch.sigmoid(m(features)[0])
                # logits += torch.sigmoid(m(features[:, :, :, torch.arange(IMG_SIZE[0] - 1, -1, -1)])[0][:, :, :, torch.arange(IMG_SIZE[0] - 1, -1, -1)])
                # logits += torch.sigmoid(m(features[:, :, torch.arange(IMG_SIZE[1] - 1, -1, -1), :])[0][:, :, torch.arange(IMG_SIZE[1] - 1, -1, -1), :])
                # logits += torch.sigmoid(m(features[:, :, torch.arange(IMG_SIZE[1] - 1, -1, -1), :][:, :, :, torch.arange(IMG_SIZE[0] - 1, -1, -1)])[0][:, :, :, torch.arange(IMG_SIZE[0] - 1, -1, -1)][:, :, torch.arange(IMG_SIZE[1] - 1, -1, -1), :])
            # logits = logits / 4
            logits = logits / (len(models))
            logits = logits.float().cpu().numpy().astype("float16")

            rles = []
            sub_ids = []
            for i, (preds, id_) in enumerate(zip(logits, img_id)):
                for class_, (remove_mask_pixel, threshold, threshold_after_remove, p) in enumerate(
                        zip(remove_mask_pixels, thresholds, threshold_after_removes, preds)):
                    sub_ids.append(id_ + "_{}".format(class_ + 1))
                    if np.sum(p > threshold) < remove_mask_pixel:
                        rles.append(None)
                        continue
                    p = np.flipud(np.rot90(p, k=1))
                    im = cv2.resize(p.astype("float64"), (256, 1600))

                    rles.append(mask2rle((im > threshold_after_remove).astype("int8"), 1600, 256))

            ids.extend(sub_ids)
            preds_rle.extend(rles)

            del features, logits, rles
            gc.collect()

    return preds_rle, ids


def predict_cls(models, test_loader, device):
    preds_rle = []
    ids = []
    next_ids = []
    next_sub_ids = []
    with torch.no_grad():
        for step, (features, img_id) in enumerate(test_loader):
            features = features.to(device)

            for i, m in enumerate(models):
                if i == 0:
                    logits = m(features)
                else:
                    logits += m(features)
            # logits += torch.sigmoid(unet_model(features[:, :, :, torch.arange(IMG_SIZE[0] - 1, -1, -1)])[0][:, :, :, torch.arange(IMG_SIZE[0] - 1, -1, -1)])
            # logits += torch.sigmoid(unet_model(features[:, :, torch.arange(IMG_SIZE[1] - 1, -1, -1), :])[0][:, :, torch.arange(IMG_SIZE[1] - 1, -1, -1), :])
            # logits += torch.sigmoid(unet_model(features[:, :, torch.arange(IMG_SIZE[1] - 1, -1, -1), :][:, :, :, torch.arange(IMG_SIZE[0] - 1, -1, -1)])[0][:, :, :, torch.arange(IMG_SIZE[0] - 1, -1, -1)][:, :, torch.arange(IMG_SIZE[1] - 1, -1, -1), :])
            # logits = logits / 4
            logits = logits / len(models)
            logits = torch.sigmoid(logits).float().cpu().numpy().astype("float16")

            rles = []
            sub_ids = []
            for i, (preds, id_) in enumerate(zip(logits, img_id)):
                for class_, (p, th) in enumerate(zip(preds, cls_thresholds)):
                    if p <= th:
                        sub_ids.append(id_ + "_{}".format(class_ + 1))
                        rles.append(None)
                        continue
                    else:
                        if id_ not in next_ids:
                            next_ids.append(id_)
                        next_sub_ids.append(id_ + "_{}".format(class_ + 1))

            ids.extend(sub_ids)
            preds_rle.extend(rles)

            del features, logits
            gc.collect()

    return preds_rle, ids, next_ids, next_sub_ids


def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel += 1;

    return " ".join(rle)


# ===============
# Constants
# ===============
IMG_DIR = "../input/test_images/"
LOGGER_PATH = "log.txt"
TEST_PATH = "../input/sample_submission.csv"
ID_COLUMNS = "ImageId"
N_CLASSES = 4

# ===============
# Settings
# ===============
SEED = np.random.randint(10000)
device = "cuda:0"
IMG_SIZE = (1600, 256)
CLR_CYCLE = 3
BATCH_SIZE = 32
FOLD_ID = 0
CLASSIFICATION = True
cls_model_path_res = [
    "../exp/models/cls_exp1_resnet_fold0_ckpt8_ema.pth",
    "../exp/models/cls_exp1_resnet_fold1_ckpt8_ema.pth",
    "../exp/models/cls_exp1_resnet_fold2_ckpt7_ema.pth",
    "../exp/models/cls_exp1_resnet_fold3_ckpt5_ema.pth",
    "../exp/models/cls_exp1_resnet_fold4_ckpt5_ema.pth",
]
cls_model_path_se = [
    "../exp/models/cls_exp2_seresnext_fold0_ckpt4_ema.pth",
    "../exp/models/cls_exp2_seresnext_fold1_ckpt3_ema.pth",
    "../exp/models/cls_exp2_seresnext_fold2_ckpt5_ema.pth",
    "../exp/models/cls_exp2_seresnext_fold3_ckpt4_ema.pth",
    "../exp/models/cls_exp2_seresnext_fold4_ckpt3_ema.pth",
]
cls_model_path_inc = [
    "../exp/models/cls_exp7_incep_fold0_ckpt3_ema.pth",
]
model_pathes = [
    "../exp/models/exp57_unet_resnet_fold0_ckpt11_ema.pth",
    "../exp/models/exp57_unet_resnet_fold1_ckpt17_ema.pth",
    "../exp/models/exp57_unet_resnet_fold2_ckpt13_ema.pth",
    "../exp/models/exp57_unet_resnet_fold3_ckpt13_ema.pth",
    "../exp/models/exp57_unet_resnet_fold4_ckpt9_ema.pth",
    "../exp/models/exp69_unet_resnet_fold0_ckpt14.pth",
]
model_pathes2 = [
    "../exp/models/exp35_unet_resnet_fold0_ckpt16.pth",
    "../exp/models/exp35_unet_resnet_fold1_ckpt14.pth",
    "../exp/models/exp35_unet_resnet_fold2_ckpt15.pth",
    "../exp/models/exp35_unet_resnet_fold4_ckpt12.pth",
]
remove_mask_pixels = [400, 800, 600, 1600]
#thresholds = [0.51, 0.58, 0.47, 0.46]
thresholds = [0.8, 0.8, 0.8, 0.8]
threshold_after_removes = [0.35, 0.39, 0.38, 0.45]
# remove_mask_pixels = [200, 1000, 400,1400]
# remove_mask_pixels = [800, 800, 800,1600]
# remove_mask_pixels = [1000, 1000, 1000,1000]
# thresholds = [0.5, 0.5, 0.5, 0.5]
# threshold_after_removes = [0.5, 0.5, 0.5, 0.5]
add_cls = 0.1
cls_thresholds = [0.11, 1.0, 0.59, 0.71]

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


with timer('load data'):
    df = pd.read_csv(TEST_PATH)
    df["ImageId"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[0])
    df["class"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[1])
    df = df.set_index(["ImageId", "class"])
    df.drop("ImageId_ClassId", axis=1, inplace=True)
    df = df.unstack()
    df.columns = ["_".join(c) for c in df.columns]
    df = df.reset_index()
    df = df[["ImageId"] + ["EncodedPixels_{}".format(i) for i in range(1, 5)]]

with timer('preprocessing'):
    test_augmentation = None
    test_dataset = SeverDatasetTest(df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                    transforms=test_augmentation)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    del test_dataset
    gc.collect()

with timer('create cls model'):
    models = []
    for p in cls_model_path_res:
        model = cls_models.ResNet(num_classes=N_CLASSES, pretrained=None)
        model.load_state_dict(torch.load(p))
        model.to(device)
        model.eval()
        models.append(model)
        del model
        torch.cuda.empty_cache()

    for p in cls_model_path_se:
        model = cls_models.SEResNext(num_classes=N_CLASSES, pretrained=None)
        model.load_state_dict(torch.load(p))
        model.to(device)
        model.eval()
        models.append(model)
        del model
        torch.cuda.empty_cache()

    for p in cls_model_path_inc:
        model = cls_models.InceptionResNetV2(num_classes=N_CLASSES, pretrained=None)
        model.load_state_dict(torch.load(p))
        model.to(device)
        model.eval()
        models.append(model)
        del model
        torch.cuda.empty_cache()

with timer('cls predict'):
    rles, sub_ids, next_ids, next_sub_ids = predict_cls(models, test_loader, device)
    sub_df = pd.DataFrame({'ImageId_ClassId': sub_ids, 'EncodedPixels': rles})
    #sub_df.to_csv('submission_cls.csv', index=False)

with timer('preprocessing'):
    test_augmentation = None
    LOGGER.info(len(df))
    df = df[df.ImageId.isin(next_ids)]
    LOGGER.info(len(df))
    test_dataset = SeverDatasetTest(df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                    transforms=test_augmentation)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    del df, test_dataset
    gc.collect()

with timer('create model'):
    models = []
    for model_path in model_pathes:
        model = smp.Unet('resnet34', encoder_weights=None, classes=N_CLASSES, encoder_se_module=True,
                         decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
                         classification=CLASSIFICATION, attention_type="cbam", center=True)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        models.append(model)
        del model
        torch.cuda.empty_cache()

    for model_path in model_pathes2:
        model = smp_old.Unet('resnet34', encoder_weights=None, classes=N_CLASSES, encoder_se_module=True,
                             decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
                             classification=CLASSIFICATION)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        models.append(model)
        del model
        torch.cuda.empty_cache()

with timer('predict'):
    rles, sub_ids = predict_dsv(models, test_loader, device)
    sub_df_ = pd.DataFrame({'ImageId_ClassId': sub_ids, 'EncodedPixels': rles})
    LOGGER.info(len(sub_df_))
    sub_df_ = sub_df_[sub_df_.ImageId_ClassId.isin(next_sub_ids)]

    sub_df_.to_csv('submission_seg_hard.csv', index=False)
