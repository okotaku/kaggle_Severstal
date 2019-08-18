import gc
import torch
import numpy as np

from apex import amp
from logger import LOGGER
from torch.autograd import Variable


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


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    accumulation_steps=1, steps_upd_logging=500, scheduler=None):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info(f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}')
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
            LOGGER.info(f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}')


    return total_loss / (step + 1)


def train_one_epoch_dsv(model, train_loader, criterion, optimizer, device,
                    accumulation_steps=1, steps_upd_logging=500, scheduler=None, classification=False, seg_weight=0.5):
    model.train()

    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        features, targets = batch[0].to(device), batch[1].to(device)
        if len(batch) == 3:
            true_y = batch[2].to(device)
        del batch
        gc.collect()

        optimizer.zero_grad()

        out_dic = model(features)
        logits = out_dic["mask"]
        loss = 0
        for l in logits:
            loss += criterion(l, targets)
        loss /= len(logits)

        if classification:
            pred_y = out_dic["class"]
            class_loss = criterion(pred_y, true_y)
            loss = (loss*seg_weight + class_loss*(1-seg_weight)) * 2

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info(f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}')
        del features, targets, logits
        gc.collect()

    return total_loss / (step + 1)


def train_one_epoch_dsv_mixup(model, train_loader, criterion, optimizer, device, img_size,
                        accumulation_steps=1, steps_upd_logging=500, scheduler=None):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(features, targets, device, alpha=mixup_alpha)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))

        optimizer.zero_grad()

        loss = 0
        for l in logits:
            loss += mixup_criterion(criterion, l.view(len(l), 1, img_size, img_size),
                                    targets_a.view(len(l), 1, img_size, img_size),
                                    targets_b.view(len(l), 1, img_size, img_size), lam)
        loss /= len(logits)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info(f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}')
        del features, targets, logits
        gc.collect()

    return total_loss / (step + 1)


def validate(model, valid_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    true_ans_list = []
    preds_cat = []
    with torch.no_grad():

        for step, (features, targets) in enumerate(valid_loader):
            features, targets = features.to(device), targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)

            test_loss += loss.item()
            true_ans_list.append(targets.float().cpu().numpy().astype("int8"))
            preds_cat.append(torch.sigmoid(logits).float().cpu().numpy().astype("float16"))

            del features, targets, logits
            gc.collect()

        all_true_ans = np.concatenate(true_ans_list, axis=0)
        all_preds = np.concatenate(preds_cat, axis=0)

    return test_loss / (step + 1), all_preds, all_true_ans

def validate_dsv(model, valid_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    true_ans_list = []
    preds_cat = []
    with torch.no_grad():

        for step, batch in enumerate(valid_loader):
            features, targets = batch[0].to(device), batch[1].to(device)
            del batch
            gc.collect()

            logits = model(features)["mask"]
            loss = 0
            for l in logits:
                loss += criterion(l, targets)
            loss /= len(logits)

            test_loss += loss.item()
            true_ans_list.append(targets.float().cpu().numpy().astype("int8"))
            preds_cat.append(torch.sigmoid(logits[0]).float().cpu().numpy().astype("float16"))

            del features, targets, logits
            gc.collect()

        all_true_ans = np.concatenate(true_ans_list, axis=0)
        all_preds = np.concatenate(preds_cat, axis=0)

    return test_loss / (step + 1), all_preds, all_true_ans
