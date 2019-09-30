import gc
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
from apex import amp
from logger import LOGGER
from torch.autograd import Variable
from tqdm import tqdm

sys.path.append("../severstal-src/")
from util import seed_torch, search_threshold
from metric import dice_all


def get_cutmixv5_data(inputs, target, beta=1, device=0):
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        addition = np.random.randint(-bbx1, W-bbx2)

        return bbx1, bby1, bbx2, bby2, addition

    lam = np.random.beta(beta, beta)

    indices_orig = torch.randperm(inputs.size(0))
    indices_shuffle = torch.randperm(inputs.size(0))
    shuffled_data = inputs[indices_shuffle]
    shuffled_targets = target[indices_shuffle]
    bbx1, bby1, bbx2, bby2, addition = _rand_bbox(inputs.size(), lam)

    inputs[indices_orig, :, bbx1:bbx2, :] = shuffled_data[:, :, bbx1+addition:bbx2+addition, :]
    target[indices_orig, :, bbx1:bbx2, :] = shuffled_targets[:, :, bbx1+addition:bbx2+addition, :]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    input_var = torch.autograd.Variable(inputs, requires_grad=True).to(device)
    target_var = torch.autograd.Variable(target, requires_grad=True).to(device)
    return input_var, target_var


def get_cutmixv4_data(inputs, target, beta=1, device=0):
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    lam = np.random.beta(beta, beta)

    indices_orig = torch.randperm(inputs.size(0))
    indices_shuffle = torch.randperm(inputs.size(0))
    shuffled_data = inputs[indices_shuffle]
    shuffled_targets = target[indices_shuffle]
    bbx1, bby1, bbx2, bby2 = _rand_bbox(inputs.size(), lam)

    inputs[indices_orig, :, bbx1:bbx2, :] = shuffled_data[:, :, bbx1:bbx2, :]
    target[indices_orig, :, bbx1:bbx2, :] = shuffled_targets[:, :, bbx1:bbx2, :]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    input_var = torch.autograd.Variable(inputs, requires_grad=True).to(device)
    target_var = torch.autograd.Variable(target, requires_grad=True).to(device)
    return input_var, target_var


def get_cutmixv2_data(inputs, target, beta=1, device=0):
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    lam = np.random.beta(beta, beta)

    indices_shuffle = torch.randperm(inputs.size(0))
    bbx1, bby1, bbx2, bby2 = _rand_bbox(inputs.size(), lam)

    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[indices_shuffle, :, bbx1:bbx2, bby1:bby2]
    target[:, :, bbx1:bbx2, bby1:bby2] = target[indices_shuffle, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    input_var = torch.autograd.Variable(inputs, requires_grad=True).to(device)
    target_var = torch.autograd.Variable(target, requires_grad=True).to(device)
    return input_var, target_var


def get_cutmixv1_data(inputs, target, beta=1, device=0):
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    lam = np.random.beta(beta, beta)

    rand_index = torch.randperm(inputs.size()[0]).to(device)
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = _rand_bbox(inputs.size(), lam)

    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
    target[:, :, bbx1:bbx2, bby1:bby2] = target[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    input_var = torch.autograd.Variable(inputs, requires_grad=True).to(device)
    target_var = torch.autograd.Variable(target, requires_grad=True).to(device)
    return input_var, target_var


def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def accumulate(model1, model2, decay=0.99):
    par1 = model1.state_dict()
    par2 = model2.state_dict()

    with torch.no_grad():
        for k in par1.keys():
            par1[k].data.copy_(par1[k].data * decay + par2[k].data * (1 - decay))


def train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=1, steps_upd_logging=500,
                    scheduler=None, cutmix_prob=0.3, beta=1, classification=False, ema_model=None, ema_decay=0.99,
                    clipnorm=None):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features = features.to(device)
        if classification:
            targets, targets_cls = targets["mask"].to(device), targets["class_y"].to(device)
        else:
            targets = targets.to(device)

        if np.random.rand() < cutmix_prob:
            features, targets = get_cutmixv2_data(
                features,
                targets,
                beta=beta,
                device=device
            )

        optimizer.zero_grad()


        if classification:
            logits, cls = model(features)
            loss = criterion(logits, targets)
            loss += criterion(cls.view(targets_cls.shape), targets_cls) / 1024
        else:
            logits = model(features)
            loss = criterion(logits, targets)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if clipnorm is not None:
                nn.utils.clip_grad_norm(model.parameters(), clipnorm)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        if ema_model is not None:
            accumulate(ema_model, model, decay=ema_decay)

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))
        del features, targets, logits
        gc.collect()


    return total_loss / (step + 1)


def train_one_epoch_noapex(model, train_loader, criterion, optimizer, device, accumulation_steps=1, steps_upd_logging=500,
                    scheduler=None, cutmix_prob=0.3, beta=1, classification=False, ema_model=None, ema_decay=0.99,
                    clipnorm=None):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features = features.to(device, dtype=torch.float)
        if classification:
            targets, targets_cls = targets["mask"].to(device, dtype=torch.float), targets["class_y"].to(device, dtype=torch.float)
        else:
            targets = targets.to(device)

        if np.random.rand() < cutmix_prob:
            features, targets = get_cutmixv2_data(
                features,
                targets,
                beta=beta,
                device=device
            )

        optimizer.zero_grad()


        if classification:
            logits, cls = model(features)
            loss = criterion(logits, targets)
            loss += criterion(cls.view(targets_cls.shape), targets_cls) / 1024
        else:
            logits = model(features)
            loss = criterion(logits, targets)

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if clipnorm is not None:
                nn.utils.clip_grad_norm(model.parameters(), clipnorm)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        if ema_model is not None:
            accumulate(ema_model, model, decay=ema_decay)

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))
        del features, targets, logits
        gc.collect()


    return total_loss / (step + 1)


#https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
def train_one_epoch_mixup(model, train_loader, criterion, optimizer, device, steps_upd_logging=500,
                          mixup_alpha=1.0):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(features, targets, device, alpha=mixup_alpha)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))

        optimizer.zero_grad()

        logits = model(inputs)
        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))


    return total_loss / (step + 1)


def train_one_epoch_dsv(model, train_loader, criterion, optimizer, device,
                        accumulation_steps=1, steps_upd_logging=500, classification=False,
                        seg_weight=0.5, cutmix_prob=0.0, beta=1, ema_model=None, ema_decay=0.99):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features = features.to(device)
        if classification:
            targets, targets_cls = targets["mask"].to(device), targets["class_y"].to(device)
        else:
            targets = targets.to(device)

        optimizer.zero_grad()

        if np.random.rand() < cutmix_prob:
            features, targets = get_cutmixv5_data(
                features,
                targets,
                beta=beta,
                device=device
            )

        out_dic = model(features)
        logits = out_dic["mask"]
        loss = 0
        for l in logits:
            loss += criterion(l, targets)
        loss /= len(logits)

        if classification:
            pred_y = out_dic["class"]
            class_loss = criterion(pred_y.view(targets_cls.shape), targets_cls)
            loss += class_loss / 1024

        if ema_model is not None:
            accumulate(ema_model, model, decay=ema_decay)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))
        del features, targets, logits
        gc.collect()

    return total_loss / (step + 1)


def validate(model, valid_loader, criterion, device, classification=False, remove_mask_pixel=[400, 1200, 1400, 1800]):
    model.eval()
    test_loss = 0.0
    true_ans_list = []
    preds_cat = []
    with torch.no_grad():

        for step, (features, targets) in enumerate(valid_loader):
            features, targets = features.to(device, dtype=torch.float), targets.to(device)

            if classification:
                logits, _ = model(features)
                loss = criterion(logits, targets)
            else:
                logits = model(features)
                loss = criterion(logits, targets)

            test_loss += loss.item()
            #true_ans_list.append(targets.float().cpu().numpy().astype("int8"))
            #preds_cat.append(torch.sigmoid(logits).float().cpu().numpy().astype("float16"))

            targets = targets.float().cpu().numpy().astype("int8")
            logits = torch.sigmoid(logits).float().cpu().numpy().astype("float32")
            scores = []
            for i in range(4):
                sum_val_preds = np.sum(logits[:, i, :, :].reshape(len(logits), -1) > 0.5, axis=1)
                val_preds_ = copy.deepcopy(logits[:, i, :, :])
                val_preds_[sum_val_preds < remove_mask_pixel[i]] = 0
                score = dice_all(targets[:, i, :, :], val_preds_)
                scores.append(score)

            del features, targets, logits
            gc.collect()

        #all_true_ans = np.concatenate(true_ans_list, axis=0)
        #all_preds = np.concatenate(preds_cat, axis=0)

    return test_loss / (step + 1), np.mean(scores)


def validate_noapex(model, valid_loader, criterion, device, classification=False):
    model.eval()
    test_loss = 0.0
    true_ans_list = []
    preds_cat = []
    with torch.no_grad():

        for step, (features, targets) in enumerate(valid_loader):
            features, targets = features.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)

            if classification:
                logits, _ = model(features)
                loss = criterion(logits, targets)
            else:
                logits = model(features)
                loss = criterion(logits, targets)

            test_loss += loss.item()
            #true_ans_list.append(targets.float().cpu().numpy().astype("int8"))
            #preds_cat.append(torch.sigmoid(logits).float().cpu().numpy().astype("float16"))

            del features, targets, logits
            gc.collect()

        #all_true_ans = np.concatenate(true_ans_list, axis=0)
        #all_preds = np.concatenate(preds_cat, axis=0)

    return test_loss / (step + 1)#, all_preds, all_true_ans


def validate_crop(model, valid_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():

        for step, (features, targets) in enumerate(valid_loader):
            start_w = 0
            for i in range(5):
                f = features[:, :, :, start_w:start_w+320].to(device)
                t = targets[:, :, :, start_w:start_w+320].to(device)
                logits = model(f)
                loss = criterion(logits, t)
                test_loss += loss.item() / 5
                start_w += 320
                del f, t
                gc.collect()

            del features, targets, logits
            gc.collect()

    return test_loss / (step + 1)


def validate_dsv(model, valid_loader, criterion, device, classification=False):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():

        for step, (features, targets) in enumerate(valid_loader):
            features, targets = features.to(device, dtype=torch.float), targets.to(device)

            logits = model(features)["mask"]#[0]
            #loss = criterion(logits, targets)
            loss = 0
            for l in logits:
                loss += criterion(l, targets)
            loss /= len(logits)

            test_loss += loss.item()

            del features, targets, logits
            gc.collect()


    return test_loss / (step + 1)


def predict_sep(model, valid_loader, criterion, device, classification=False):
    model.eval()
    test_loss = 0.0
    score_all1 = []
    score_all2 = []
    score_all3 = []
    score_all4 = []
    with torch.no_grad():

        for step, (features, targets) in enumerate(tqdm(valid_loader)):
            features, targets = features.to(device), targets.to(device)

            if classification:
                logits, _ = model(features)
            else:
                logits = model(features)
            loss = criterion(logits, targets)

            targets = targets.float().cpu().numpy().astype("int8")
            logits = torch.sigmoid(logits.view(targets.shape)).float().cpu().numpy().astype("float16")

            test_loss += loss.item()
            for i in range(4):
                th, score, ths, scores = search_threshold(targets[:, i, :, :],logits[:, i, :, :])
                if i == 0:
                    score_all1.append(scores)
                if i == 1:
                    score_all2.append(scores)
                if i == 2:
                    score_all3.append(scores)
                if i == 3:
                    score_all4.append(scores)

            del features, targets, logits
            gc.collect()

    return test_loss / (step + 1), np.array(score_all1), np.array(score_all2), np.array(score_all3), np.array(score_all4), ths

def predict(model, valid_loader, criterion, device, classification=False):
    model.eval()
    test_loss = 0.0
    true_ans_list = []
    preds_cat = []
    cls = []
    with torch.no_grad():

        for step, (features, targets) in enumerate(tqdm(valid_loader)):
            features, targets = features.to(device), targets.to(device)

            if classification:
                logits, cls_ = model(features)
            else:
                logits = model(features)
            loss = criterion(logits, targets)

            targets = targets.float().cpu().numpy().astype("int8")
            logits = torch.sigmoid(logits.view(targets.shape)).float().cpu().numpy().astype("float16")
            cls_ = torch.sigmoid(cls_).float().cpu().numpy().astype("float16")

            test_loss += loss.item()

            true_ans_list.append(targets)
            preds_cat.append(logits)
            cls.append(cls_)

            del features, targets, logits
            gc.collect()

        all_true_ans = np.concatenate(true_ans_list, axis=0)
        all_preds = np.concatenate(preds_cat, axis=0)
        cls = np.concatenate(cls, axis=0)

    return test_loss / (step + 1), all_preds, all_true_ans, cls


def predict_dsv(model, valid_loader, criterion, device, classification=False):
    model.eval()
    test_loss = 0.0
    true_ans_list = []
    preds_cat = []
    cls = []
    with torch.no_grad():

        for step, (features, targets) in enumerate(tqdm(valid_loader)):
            features, targets = features.to(device), targets.to(device)

            if classification:
                out_dic = model(features)
            else:
                out_dic = model(features)
            logits = out_dic["mask"][0]
            cls_ = out_dic["class"]
            loss = criterion(logits, targets)

            targets = targets.float().cpu().numpy().astype("int8")
            logits = torch.sigmoid(logits.view(targets.shape)).float().cpu().numpy().astype("float16")
            cls_ = torch.sigmoid(cls_).float().cpu().numpy().astype("float16")

            test_loss += loss.item()

            true_ans_list.append(targets)
            preds_cat.append(logits)
            cls.append(cls_)

            del features, targets, logits
            gc.collect()

        all_true_ans = np.concatenate(true_ans_list, axis=0)
        all_preds = np.concatenate(preds_cat, axis=0)
        cls = np.concatenate(cls, axis=0)

    return test_loss / (step + 1), all_preds, all_true_ans, cls
