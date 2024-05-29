import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import argparse

from torch_geometric.data import DataLoader, InMemoryDataset
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')


def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./data/graph_dataset/ast/')
    # parser.add_argument('-o', '--output', help='The dir path of output', type=str, default='./data/result/GCN/ast/')
    # parser.add_argument('-m', '--model', help='choose one model', type=str, default='gcn')
    parser.add_argument('-s', '--size', help='Batch_size', type=int , default=64)
    parser.add_argument('-e', '--epoch', help='num_epochs', type=int, default=200)
    args = parser.parse_args()
    return args


class MyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):  
        super(MyDataset, self).__init__(root, transform, pre_transform)  
        self.data, self.slices = torch.load(self.processed_paths[0])  
  
    @property  
    def raw_file_names(self):  
        # 返回原始数据文件的名称，如果有的话  
        return []  
  
    @property  
    def processed_file_names(self):  
        # 返回处理后的数据文件（即.pt文件）的名称
        return ['data.pt']  
  
    def download(self):  
        # 如果需要下载数据，则在此处实现  
        pass  
  
    def process(self):  
        # 如果需要处理原始数据，则在此处实现  
        # 在大多数情况下，由于我们已经有了.pt文件，所以这里不需要做任何处理  
        pass


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


def main():
    args = parse_options()
    input = args.input
    batch_size = args.size
    num_epochs = args.epoch
    # output = args.output
    
    device = torch.device('cuda')
    
    # path = osp.join(osp.expanduser('~'), 'datasets') 
    # dataset = TUDataset(path, name='PTC_MR')  # 344个图，每个图格式如下 Data(edge_index=[2, 2], x=[2, 18], edge_attr=[2, 4], y=[1])
    
    dataset = MyDataset(input)  # 3352个图， 每个图格式如下 Data(x=[80, 257], edge_index=[2, 79], y=[1], edge_weight=[79])
    # print(dataset)
    
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    input_dim = max(dataset.num_features, 1)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=num_epochs, desc='(T)') as pbar:
        for epoch in range(1, num_epochs+1):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()