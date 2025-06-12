from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import _sigmoid, _transpose_and_gather_feat


def _dice_loss(input_mask, cls_gt):
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:, i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt == (i + 1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


class BootstrappedCE(nn.Module):

    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / (self.end_warm - self.start_warm))

        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return _dice_loss(input, target)


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super().__init__()
        self.neg_loss = _neg_loss

    def forward(self, input, target):
        return self.neg_loss(input, target)


class RegL1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class Loss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.multitask = config['multitask']

        self.bce_criterion = BootstrappedCE(config['start_warm'], config['end_warm'])
        self.dice_criterion = DiceLoss()
        if self.multitask:
            self.hm_criterion = FocalLoss()
            # self.hm_criterion = nn.MSELoss()
            self.bs_criterion = nn.L1Loss()
            self.of_criterion = RegL1Loss()

    def forward(self, output, data, it, is_train=True):
        losses = defaultdict(int)

        losses['total_loss'] = 0
        if is_train:
            loss, p = self.bce_criterion(output['seg_logits'], data['mask'][:, 0], it)
            losses['p'] += p
            losses['ce_loss'] += loss
            losses['dice_loss'] += self.dice_criterion(output['seg_prob'], data['mask'][:, 0])
            if self.multitask:
                output['hm'] = _sigmoid(output['hm'])
                losses['hm_loss'] += self.hm_criterion(output['hm'], data['hm'])
                losses['dense_bs_loss'] += self.bs_criterion(
                    output['dense_bs'] * data['dense_bs_mask'], data['dense_bs'] * data['dense_bs_mask']
                )
                losses['of_loss'] += self.of_criterion(output['of'], data['of_mask'], data['ind'], data['of'])
        else:
            with torch.no_grad():
                loss, p = self.bce_criterion(output['seg_logits'], data['mask'][:, 0], 0)
                losses['p'] += p
                losses['ce_loss'] += loss.item()
                losses['dice_loss'] = self.dice_criterion(output['seg_prob'], data['mask'][:, 0]).item()
                if self.multitask:
                    losses['hm_loss'] += self.hm_criterion(output['hm'], data['hm']).item()
                    losses['dense_bs_loss'] += self.bs_criterion(
                        output['dense_bs'] * data['dense_bs_mask'], data['dense_bs'] * data['dense_bs_mask']
                    ).item()
                    losses['of_loss'] += self.of_criterion(output['of'], data['of_mask'], data['ind'], data['of']).item()

        losses['total_loss'] += losses['ce_loss'] + losses['dice_loss']
        if self.multitask:
            losses['total_loss'] += 1 * losses['hm_loss']
            losses['total_loss'] += 1 * losses['dense_bs_loss']
            losses['total_loss'] += 1 * losses['of_loss']

        return losses
    
class square_loss(nn.Module):
    def __init__(self, config):
        self.multitask = config['multitask']

        self.bce_criterion = BootstrappedCE(config['start_warm'], config['end_warm'])
        self.dice_criterion = DiceLoss()
        if self.multitask:
            self.hm_criterion = FocalLoss()
            # self.hm_criterion = nn.MSELoss()
            self.bs_criterion = nn.L1Loss()
            self.of_criterion = RegL1Loss()

    '''
    OutputData
    '''
    def forward(self, output, data, it, is_train=True):
        losses = defaultdict(int)

        losses['total_loss'] = 0
        if is_train:
            loss, p = self.bce_criterion(output['seg_logits'], data['mask'][:, 0], it)
            losses['p'] += p
            losses['ce_loss'] += loss
            losses['dice_loss'] += self.dice_criterion(output['seg_prob'], data['mask'][:, 0])
            if self.multitask:
                output['hm'] = _sigmoid(output['hm'])
                losses['hm_loss'] += self.hm_criterion(output['hm'], data['hm'])
                losses['dense_bs_loss'] += self.bs_criterion(
                    output['dense_bs'] * data['dense_bs_mask'], data['dense_bs'] * data['dense_bs_mask']
                )
                losses['of_loss'] += self.of_criterion(output['of'], data['of_mask'], data['ind'], data['of'])
        else:
            with torch.no_grad():
                loss, p = self.bce_criterion(output['seg_logits'], data['mask'][:, 0], 0)
                losses['p'] += p
                losses['ce_loss'] += loss.item()
                losses['dice_loss'] = self.dice_criterion(output['seg_prob'], data['mask'][:, 0]).item()
                if self.multitask:
                    losses['hm_loss'] += self.hm_criterion(output['hm'], data['hm']).item()
                    losses['dense_bs_loss'] += self.bs_criterion(
                        output['dense_bs'] * data['dense_bs_mask'], data['dense_bs'] * data['dense_bs_mask']
                    ).item()
                    losses['of_loss'] += self.of_criterion(output['of'], data['of_mask'], data['ind'], data['of']).item()

        losses['total_loss'] += losses['ce_loss'] + losses['dice_loss']
        if self.multitask:
            losses['total_loss'] += 1 * losses['hm_loss']
            losses['total_loss'] += 1 * losses['dense_bs_loss']
            losses['total_loss'] += 1 * losses['of_loss']

        return losses
    
def lossManager(config):
    if config['dataset'] == 'square':
        return square_loss(config)
    else:
        return Loss(config)