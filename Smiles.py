from statistics import mean
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cl.losses as L
import cl.augmentors as A
import torch.nn.functional as F
import copy
import torch
from tqdm import tqdm
from torch.optim import Adam
from cl.eval import SVMEvaluator, get_split
from cl.models import BContrast
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GCNConv

from torch_geometric.data import DataLoader, Data

from BagGraph import GraphGenerator
from dataset import load_dataset
from torch_geometric.utils import dense_to_sparse

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np


class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

def make_gin_conv(input_dim: int, out_dim: int) -> GINConv:
    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim))
    return GINConv(mlp)

def make_gcn_conv(input_dim: int, out_dim: int) -> GCNConv:
    return GCNConv(input_dim, out_dim)

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        
        self.layers.append(make_gcn_conv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(make_gcn_conv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, feat_dim, num_head=2, threshold=0.1, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.encoder1 = encoder
        self.encoder2 = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        # Bag Graph Generation
        self.bag2graph = GraphGenerator(feat_dim, num_head, threshold)

    def get_encoder2(self):
        if self.encoder2 is None:
            self.encoder2 = copy.deepcopy(self.encoder1)

            for p in self.encoder2.parameters():
                p.requires_grad = False
        return self.encoder2

    def update_encoder2(self, momentum: float):
        for p, new_p in zip(self.get_encoder2().parameters(), self.encoder1.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        aug1, aug2 = self.augmentor
        
        # Bag Augmentations
        bag1, _, _ = aug1(x, edge_index, edge_weight)
        bag2, _, _ = aug2(x, edge_index, edge_weight)

        # Bag to Graph
        sim1 = self.bag2graph(bag1, bag1)
        sim2 = self.bag2graph(bag2, bag2)

        sim1 += sim1.clone().t()  # sysmetric
        sim1 = F.normalize(sim1, dim=0, p=1)
        sim2 += sim2.clone().t()  # sysmetric
        sim2 = F.normalize(sim2, dim=0, p=1)

        edge_index1, edge_weight1 = dense_to_sparse(sim1)
        edge_index2, edge_weight2 = dense_to_sparse(sim2)
        
        h1, h1_ = self.encoder1(bag1, edge_index1, edge_weight=edge_weight1)
        h2, h2_ = self.encoder1(bag2, edge_index2, edge_weight=edge_weight2)

        # OWA Averaging Aggregation
        g1 = global_mean_pool(h1, batch)
        # g1 = global_add_pool(h1, batch)
        h1_pred = self.predictor(h1_)
        g2 = global_mean_pool(h2, batch)
        # g2 = global_add_pool(h2, batch)
        h2_pred = self.predictor(h2_)

        with torch.no_grad():
            _, h1_ = self.get_encoder2()(bag1, edge_index1, edge_weight1)
            _, h2_ = self.get_encoder2()(bag2, edge_index2, edge_weight2)
            g1_ = global_mean_pool(h1_, batch)
            # g1_ = global_add_pool(h1_, batch)
            g2_ = global_mean_pool(h2_, batch)
            # g2_ = global_add_pool(h2_, batch)

        return g1, g2, h1_pred, h2_pred, g1_, g2_


def train(encoder_model, contrast_model, dataloader, optimizer):
    
    encoder_model.train()
    total_loss = 0

    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_insts = data.batch.size(0)
            data.x = torch.ones((num_insts, 1), dtype=torch.float32).to(data.batch.device)

        optimizer.zero_grad()
        
        g1, _, h1_pred, h2_pred, g1_, g2_ = encoder_model(data.x, data.edge_index, batch=data.batch)

        loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred,
                              g1_=g1_.detach(), g2_=g2_.detach(), batch=data.batch)
        
        loss.backward()
        optimizer.step()
        
        encoder_model.update_encoder2(0.99)

        total_loss += loss.item()

    return total_loss

@ignore_warnings(category=ConvergenceWarning)
def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    i=0

    for data in dataloader:
        i += 1
        data = data.to('cuda')
        if data.x is None:
            num_insts = data.batch.size(0)
            data.x = torch.ones((num_insts, 1), dtype=torch.float32, device=data.batch.device)
        
        g1, g2, _, _, _, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
        
        z = torch.cat([g1, g2], dim=1)
        
        x.append(z)
        
        y.append(data.y)
    
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    print('Mean ACC:', np.mean(accuracies), 'STD:', np.std(accuracies))

    results = []

    results.append(np.mean(accuracies))

    return results


def main():
    device = torch.device('cuda')
    # load multi-instance data [musk1, musk2, fox, tiger, elephant]
    # more data are listed in './data' folder
    data_name = 'elephant'
    
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    bags = load_dataset(path, data_name)
    
    # convert the bag data format to the edgeless graph format 
    data2graph = []
    for bagi in bags:
        xi = bagi[0]
        yi = torch.tensor(bagi[1][0])
        nodenum = xi.size(0)
        num_features = xi.size(1)
        # edgeless graph
        init_adj = torch.zeros([nodenum, nodenum])
        edge_index, edge_attr = dense_to_sparse(init_adj)
        data2graph.append(Data(x=xi, edge_index=edge_index, y=yi))
    # print('length of bags is ', len(data2graph))
    # torch.save(data2graph, 'data2graph.pth')

    dataloader = DataLoader(data2graph, batch_size=10)

    input_dim = max(num_features, 1)

    # Augmentation Strategies
    aug1 = A.Compose([A.InstanceDrop(pf=0.0), A.InstanceMasking(pf=0.2), A.InstanceRand(pf=0.0), A.InstanceReplace(pf=0.0)])
    aug2 = A.Compose([A.InstanceDrop(pf=0.0), A.InstanceMasking(pf=0.2), A.InstanceRand(pf=0.0), A.InstanceReplace(pf=0.0)])

    gconv = GConv(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=128, feat_dim=input_dim).to(device)
    contrast_model = BContrast(loss=L.BLoss(), mode='B2I').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    res = []
    for run in range(5):
        print('Run', run, ':')
        with tqdm(total=100, desc='(T)') as pbar:
            for epoch in range(1, 101):
                loss = train(encoder_model, contrast_model, dataloader, optimizer)
                
                pbar.set_postfix({'loss': loss})
                pbar.update()
            
                if epoch % 10 == 0:
                    test_result = test(encoder_model, dataloader)
                    res.append(test_result)


if __name__ == '__main__':
    main()
