import os
import random
import numpy as np
import torch
from metric import dice


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def search_threshold(mask, pred):
    best_score = 0
    best_th = 0
    count = 0
    results = []
    ths = np.linspace(0, 1, 101)

    for th in ths:
        scores = []
        for y_val_, y_pred_ in zip(mask, pred):
            score = dice(y_val_, y_pred_ > th)
            if np.isnan(score):
                scores.append(1)
            else:
                scores.append(score)
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_th = th
            count = 0
        results.append(np.mean(scores))

        if np.mean(scores) < best_score:
            count += 1
        #if count == 50:
        #    break
    #results = results + [0 for _ in range(len(ths) - len(results))]

    return best_th, best_score, ths, results


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height = imgshape[1]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return np.flipud(np.rot90(mask.reshape(height, width), k=1))
