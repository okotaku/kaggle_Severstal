import os
import cv2
import torch
import random
import pydicom
import numpy as np
from torch.utils.data import Dataset

from util import rle2mask


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

        if self.crop_rate < 1:
            img = random_cropping(img, mask, is_random=True, ratio=self.crop_rate)
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        mask[mask != 0] = 1

        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        if self.cut_h:
            img, mask = cutout_h(img, mask, self.img_size, self.n_classes)

        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        if self.class_y is None:
            return torch.from_numpy(img), torch.from_numpy(mask)
        else:
            class_y_ = self.class_y[idx]
            return torch.from_numpy(img), torch.from_numpy(mask), torch.tensor(class_y_)


def pytorch_image_to_tensor_transform(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image).float().div(255)

    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]

    return tensor

def random_cropping(image, mask, ratio = 0.8, is_random = True):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    crop = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    crop_mask = mask[start_y:start_y+target_h,start_x:start_x+target_w,:]

    #crop = cv2.resize(zeros ,(width,height)) #pad to original size
    return crop, crop_mask

def cutout_h(img, mask, img_size, n_classes, mask_value="zeros", min_h=10, max_h=60):
    if mask_value == "mean":
        mask_value = [
            int(np.mean(img[:, :, 0])),
            int(np.mean(img[:, :, 1])),
            int(np.mean(img[:, :, 2])),
        ]
    elif mask_value == "zeros":
        mask_value = [0, 0, 0]

    mask_size_h = int(np.random.randint(min_h, max_h))

    cutout_left = np.random.randint(
        0 - mask_size_h // 2, img_size[0] - mask_size_h
    )
    cutout_right = cutout_left + mask_size_h

    if cutout_left < 0:
        cutout_left = 0

    img[:, cutout_left:cutout_right, :] = mask_value
    mask[:, cutout_left:cutout_right, :] = [0 for _ in range(n_classes)]

    return img, mask