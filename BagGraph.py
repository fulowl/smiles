import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from util_funcs import cos_sim

class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        return h * self.weight


class GraphGenerator(nn.Module):
    # Generate graph
    def __init__(self, dim, num_head=2, threshold=0.1, dev=torch.device('cuda')):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left, right):
        if torch.sum(left) == 0 or torch.sum(right) == 0:
            return torch.zeros((left.shape[0], right.shape[0])).to(self.dev)
        s = torch.zeros((left.shape[0], right.shape[0])).to(self.dev)
        zero_lines = torch.nonzero(torch.sum(left, 1) == 0)
        if len(zero_lines) > 0:
            left[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            weighted_left = self.metric_layer[i](left)
            weighted_right = self.metric_layer[i](right)
            s += cos_sim(weighted_left, weighted_right)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s