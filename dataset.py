import torch
import scipy.io as sio
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_dataset(path, dataset_nm):
    # load data from file
    data = sio.loadmat(path + '/' + dataset_nm + '.mat')
    ins_fea = data['x']['data'][0, 0]
    if dataset_nm.startswith('musk'):
        bags_nm = data['x']['ident'][0, 0]['milbag'][0, 0]
    else:
        bags_nm = data['x']['ident'][0, 0]['milbag'][0, 0][:, 0]
    bags_label = data['x']['nlab'][0, 0][:, 0] - 1

    # L2 norm for musk1 and musk2
    if dataset_nm.startswith('newsgroups') is False:
        mean_fea = np.mean(ins_fea, axis=0, keepdims=True)+1e-6
        std_fea = np.std(ins_fea, axis=0, keepdims=True)+1e-6
        ins_fea = np.divide(ins_fea-mean_fea, std_fea)

    # store data in bag level
    ins_idx_of_input = {}
    for id, bag_nm in enumerate(bags_nm):
        if bag_nm in ins_idx_of_input:
            ins_idx_of_input[bag_nm].append(id)
        else:
            ins_idx_of_input[bag_nm] = [id]
    bags_fea = []
    for bag_nm, ins_idxs in ins_idx_of_input.items():
        bag_fea = ([], [])
        for ins_idx in ins_idxs:
            bag_fea[0].append(ins_fea[ins_idx])
            bag_fea[1].append(bags_label[ins_idx])
        bag_fea0 = torch.tensor(bag_fea[0]).float()
        bags_fea.append([bag_fea0, bag_fea[1]])
    
    return bags_fea


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty(
        (x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x

