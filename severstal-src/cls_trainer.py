import torch
import numpy as np

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

    rand_index = torch.randperm(inputs.size()[0]).to(device)
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = _rand_bbox(inputs.size(), lam)

    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    input_var = torch.autograd.Variable(inputs, requires_grad=True).to(device)
    target_a_var = torch.autograd.Variable(target_a).to(device).long()
    target_b_var = torch.autograd.Variable(target_b).to(device).long()
    return input_var, target_a_var, target_b_var, lam


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


def train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=1,
                    steps_upd_logging=500, scheduler=None, ema_model=None, ema_decay=0.0):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        if ema_model is not None:
            accumulate(ema_model, model, decay=ema_decay)

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

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


def train_one_epoch_cutmix(model, train_loader, criterion, optimizer, device, accumulation_steps=1,
                    steps_upd_logging=500, scheduler=None,
                    cutmix_prob=0.3, beta=1
                    ):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        if np.random.rand() < cutmix_prob:
            input_var, target_a_var, target_b_var, lam = get_cutmix_data(
                features.to(device),
                targets.double(),
                beta=beta,
                device=device
            )
            output = model(input_var)
            loss = criterion(output, target_a_var) * lam + criterion(output, target_b_var) * (1. - lam)
        else:
            logits = model(features)
            loss = criterion(logits, targets)

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))
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
            true_ans_list.append(targets)
            preds_cat.append(logits)

        all_true_ans = torch.cat(true_ans_list).float().cpu().numpy()
        all_preds = torch.cat(preds_cat).float().cpu().numpy()

    return test_loss / (step + 1), all_preds, all_true_ans


def validate_tta(model, valid_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    true_ans_list = []
    preds_cat = []
    with torch.no_grad():

        for step, (features, flip_features, targets) in enumerate(valid_loader):
            features, targets = features.to(device), targets.to(device)
            flip_features = flip_features.to(device)

            logits = model(features) /2
            logits += model(flip_features) / 2
            loss = criterion(logits, targets)

            test_loss += loss.item()
            true_ans_list.append(targets)
            preds_cat.append(logits)

        all_true_ans = torch.cat(true_ans_list).float().cpu().numpy()
        all_preds = torch.cat(preds_cat).float().cpu().numpy()

    return test_loss / (step + 1), all_preds, all_true_ans
