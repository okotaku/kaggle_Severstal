import gc
import torch
import numpy as np

from apex import amp
from logger import LOGGER
from torch.autograd import Variable


def get_cutmix_data(inputs, target, beta=1, device=0):
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
                    accumulation_steps=1, steps_upd_logging=500, scheduler=None, cutmix_prob=0.3, beta=1):
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
                        accumulation_steps=1, steps_upd_logging=500, scheduler=None, classification=False,
                        seg_weight=0.5, cutmix_prob=0.3, beta=1):
    model.train()

    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        features, targets = batch[0].to(device), batch[1].to(device)
        if len(batch) == 3:
            true_y = batch[2].to(device)
        del batch
        gc.collect()

        optimizer.zero_grad()

        if np.random.rand() < cutmix_prob:
            input_var, target_var = get_cutmix_data(
                features,
                targets,
                beta=beta,
                device=device
            )
            out_dic = model(input_var)
            logits = out_dic["mask"]
            loss = 0
            for l in logits:
                loss += criterion(l, target_var)
            loss /= len(logits)
        else:
            out_dic = model(features)
            logits = out_dic["mask"]
            loss = 0
            for l in logits:
                loss += criterion(l, targets)
            loss /= len(logits)

        if classification:
            pred_y = out_dic["class"]
            class_loss = criterion(pred_y, true_y)
            loss = (loss * seg_weight + class_loss * (1 - seg_weight)) * 2

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))
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
            #true_ans_list.append(targets.float().cpu().numpy().astype("int8"))
            #preds_cat.append(torch.sigmoid(logits).float().cpu().numpy().astype("float16"))

            del features, targets, logits
            gc.collect()

        #all_true_ans = np.concatenate(true_ans_list, axis=0)
        #all_preds = np.concatenate(preds_cat, axis=0)

    return test_loss / (step + 1)#, all_preds, all_true_ans

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
