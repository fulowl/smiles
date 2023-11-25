from cl.augmentors.augmentor import Graph, Augmentor
from cl.augmentors.functional import drop_feature, replace_feature, replace_feature_np
import torch


class InstanceMasking(Augmentor):
    def __init__(self, pf: float):
        super(InstanceMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        if self.pf == 0.0:
            return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        # x = replace_feature(x, self.pf)
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
