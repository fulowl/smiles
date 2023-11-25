import torch
import torch.nn.functional as F
from .losses import Loss


class BLoss(Loss):
    def __init__(self):
        super(BLoss, self).__init__()

    def compute(self, anchor, sample, pos_mask, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        anchor = F.normalize(anchor, dim=-1, p=2)
        sample = F.normalize(sample, dim=-1, p=2)

        similarity = anchor @ sample.t()
        loss = (similarity * pos_mask).sum(dim=-1)
        return loss.mean()
