import torch
import torch.nn as nn
import torch.nn.functional as F

from config import _C as config
from DualTaskLoss import DualTaskLoss
from DiceLoss import BinaryDiceLoss


class SegLoss(nn.Module):
    def __init__(self, ignore_label=255):
        super(SegLoss, self).__init__()

        weight = torch.FloatTensor([0.15, 0.85])
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w),
                                  mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.loss(score, target)

        return loss


class BoundaryBCELoss(nn.Module):
    def __init__(self, ignore_label=255):
        super(BoundaryBCELoss, self).__init__()
        self.ignore_index = ignore_label

    def forward(self, edge, target, boundary):
        edge = edge.squeeze(dim=1)
        mask = target != self.ignore_index
        pos_mask = (boundary == 1) & mask
        neg_mask = (boundary == 0) & mask
        num = torch.clamp(mask.sum(), min=1)
        pos_weight = neg_mask.sum() / num
        neg_weight = pos_mask.sum() / num

        weight = torch.zeros_like(boundary.float())
        weight[pos_mask] = pos_weight
        weight[neg_mask] = neg_weight
        loss = F.binary_cross_entropy(edge, boundary.float(), weight, reduction='sum') / num
        return loss


class DualLoss(nn.Module):
    def __init__(self, cuda=True):
        super(DualLoss, self).__init__()
        self.loss = DualTaskLoss(cuda=cuda)

    def forward(self, score, target):
        loss = self.loss(score, target)

        return loss


class FullLoss(nn.Module):
    def __init__(self):
        super(FullLoss, self).__init__()
        self.loss1 = SegLoss()
        self.loss2 = BoundaryBCELoss()
        self.loss3 = DualLoss(cuda=True)

    def forward(self, seg, edge, target, boundary):
        loss1 = self.loss1(seg, target)
        loss2 = self.loss2(edge, target, boundary)
        loss3 = self.loss3(seg, target)
        loss = loss1 + 20 * loss2 + loss3

        return loss

