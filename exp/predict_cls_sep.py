import os
import gc
import cv2
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import segmentation_models_pytorch as smp
from apex import amp
from tqdm import tqdm
from contextlib import contextmanager
from albumentations import *
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append("../severstal-src/")
from util import seed_torch, search_threshold, rle2mask
from logger import setup_logger, LOGGER
from trainer import predict
from metric import dice
from cls_trainer import validate
import cls_models
sys.path.append("../")
import segmentation_models_pytorch as smp


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
EPOCHS = 71
FOLD_ID = 0
EXP_ID = "exp57_unet_resnet"
CLASSIFICATION = True
base_ckpt = 11
#base_model = None
base_model = "models/{}_fold{}_ckpt{}_ema.pth".format(EXP_ID, FOLD_ID, base_ckpt)
base_model_cls = "models/{}_fold{}_ckpt{}_ema.pth".format("cls_exp2_seresnext", FOLD_ID, 4)
ths = [0.5, 0.5, 0.5, 0.5]
remove_pixels = [800, 800, 800, 400]

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)
LOGGER.info("seed={}".format(SEED))


class SeverDataset(Dataset):

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
                 class_y=None,
                 cut_h=False
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
        self.class_y = class_y
        self.cut_h = cut_h

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_path = os.path.join(self.img_dir, img_id)

        img = cv2.imread(img_path)
        w, h, _ = img.shape
        mask = np.zeros((w, h, self.n_classes))
        for i, encoded in enumerate(cur_idx_row[self.mask_colname]):
            if encoded in "-1":
                continue
            else:
                mask[:, :, i] = rle2mask(encoded, (w, h))

        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        mask[mask != 0] = 1

        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        if self.class_y is not None:
            class_y_ = self.class_y[idx]
            target = {"mask": torch.Tensor(mask), "class_y": torch.tensor(class_y_)}
        else:
            target = mask

        return torch.Tensor(img), target


class SeverCLSDataset(Dataset):

    def __init__(self,
                 df,
                 img_dir,
                 img_size,
                 n_classes,
                 class_y,
                 crop_rate=1.0,
                 id_colname="ImageId",
                 mask_colname=["EncodedPixels_{}".format(i) for i in range(1, 5)],
                 transforms=None,
                 means=[0.485, 0.456, 0.406],
                 stds=[0.229, 0.224, 0.225],
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
        self.class_y = class_y

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

        class_y_ = self.class_y[idx]
        target = torch.tensor(class_y_)

        return torch.tensor(img), target
    

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def predict(model, valid_loader, criterion, device, classification=False):
    model.eval()
    test_loss = 0.0
    true_ans_list = []
    preds_cat = []
    cls = []
    with torch.no_grad():

        for step, (features, targets) in enumerate(tqdm(valid_loader)):
            features = features.to(device)
            targets, val_cls = targets["mask"].to(device), targets["class_y"]

            if classification:
                logits, cls_ = model(features)
            else:
                logits = model(features)
            loss = criterion(logits, targets)

            targets = targets.float().cpu().numpy().astype("int8")
            logits = torch.sigmoid(logits.view(targets.shape)).float().cpu().numpy().astype("float16")
            if step == 0:
                print(val_cls)
            val_cls = torch.sigmoid(val_cls).numpy()
            if step == 0:
                print(val_cls)

            test_loss += loss.item()

            true_ans_list.append(targets)
            preds_cat.append(logits)
            cls.append(val_cls)

            del features, targets, logits
            gc.collect()

        all_true_ans = np.concatenate(true_ans_list, axis=0)
        all_preds = np.concatenate(preds_cat, axis=0)
        cls = np.concatenate(cls, axis=0)

    return test_loss / (step + 1), all_preds, all_true_ans, cls


def main(seed):
    with timer('load data'):
        df = pd.read_csv(FOLD_PATH)
        y1 = (df.EncodedPixels_1 != "-1").astype("float32").values.reshape(-1, 1)
        y2 = (df.EncodedPixels_2 != "-1").astype("float32").values.reshape(-1, 1)
        y3 = (df.EncodedPixels_3 != "-1").astype("float32").values.reshape(-1, 1)
        y4 = (df.EncodedPixels_4 != "-1").astype("float32").values.reshape(-1, 1)
        y = np.concatenate([y1, y2, y3, y4], axis=1)

    with timer('preprocessing'):
        val_df = df[df.fold_id == FOLD_ID]
        y_val = y[df.fold_id == FOLD_ID]

        val_dataset = SeverCLSDataset(val_df, IMG_DIR, IMG_SIZE, N_CLASSES, y_val, id_colname=ID_COLUMNS,
                                      transforms=None)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

        model = cls_models.SEResNext(num_classes=N_CLASSES)
        model.load_state_dict(torch.load(base_model_cls))
        model.to(device)
        criterion = torch.nn.BCEWithLogitsLoss()

        valid_loss, y_val, y_true = validate(model, val_loader, criterion, device)
        #y_val = np.load("../exp_cls/y_pred_ema_ckpt8.npy")
        LOGGER.info("val loss={}".format(valid_loss))

        val_augmentation = None
        val_dataset = SeverDataset(val_df, IMG_DIR, IMG_SIZE, N_CLASSES, id_colname=ID_COLUMNS,
                                  transforms=val_augmentation, class_y=y_val)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        del val_df, df, val_dataset
        gc.collect()

    with timer('create model'):
        model = smp.Unet('resnet34', encoder_weights="imagenet", classes=N_CLASSES, encoder_se_module=True,
                         decoder_semodule=True, h_columns=False, skip=True, act="swish", freeze_bn=True,
                         classification=CLASSIFICATION, attention_type="cbam", center=True)
        model.load_state_dict(torch.load(base_model))
        model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()

    with timer('predict'):
        valid_loss, y_pred, y_true, cls = predict(model, val_loader, criterion, device, classification=CLASSIFICATION)
        LOGGER.info('Mean valid loss: {}'.format(round(valid_loss, 5)))

        scores = []
        for i, (th, remove_mask_pixel) in enumerate(zip(ths, remove_pixels)):
            sum_val_preds = np.sum(y_pred[:, i, :, :].reshape(len(y_pred), -1) > th, axis=1)
            cls_ = cls[:, i]

            best = 0
            for th_cls in np.linspace(0, 1, 101):
                val_preds_ = copy.deepcopy(y_pred[:, i, :, :])
                val_preds_[sum_val_preds < remove_mask_pixel] = 0
                val_preds_[cls_ <= th_cls] = 0
                scores = []
                for y_val_, y_pred_ in zip(y_true[:, i, :, :], val_preds_):
                    score = dice(y_val_, y_pred_ > 0.5)
                    if np.isnan(score):
                        scores.append(1)
                    else:
                        scores.append(score)
                if np.mean(scores) >= best:
                    best = np.mean(scores)
                    best_th = th_cls
                #else:
                #    break
            LOGGER.info('dice={} on {}'.format(best, best_th))
            scores.append(best)

        LOGGER.info('holdout dice={}'.format(np.mean(scores)))


if __name__ == '__main__':
    main(SEED)
