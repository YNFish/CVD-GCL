from torch_geometric.data import DataLoader, InMemoryDataset
from torch_geometric.nn import GCNConv, GraphConv, GATConv
import torch
import argparse
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import scipy.sparse as sp
from torch import nn, optim
from torch.utils.data import random_split
from torch_geometric.nn import SAGPooling  
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import csv
from sklearn.metrics import confusion_matrix, confusion_matrix, classification_report, precision_recall_fscore_support
# import matplotlib.pyplot as plt

torch.manual_seed(12345)


def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./data/graph_dataset/ast/')
    parser.add_argument('-o', '--output', help='The dir path of output', type=str, default='./data/result/GCN/ast/')
    parser.add_argument('-m', '--model', help='choose one model', type=str, default='gcn')
    parser.add_argument('-s', '--size', help='Batch_size', type=int , default=64)
    parser.add_argument('-e', '--epoch', help='num_epochs', type=int, default=400)
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
  
def split_dateset(dataset):
    num_samples = len(dataset)
    # 定义训练集、验证集和测试集的大小  0.8 0.1 0.1
    train_size = int(0.8 * num_samples)  
    val_size = int(0.1 * num_samples)  
    test_size = num_samples - train_size - val_size  
    
    # 使用random_split函数划分数据集  
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset,val_dataset,test_dataset


# 定义GCN类
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)   # [batch_size, hidden_channels]
        
        # 3. 分类器
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    

# 定义GGNN
class GraphGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(GraphGNN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    
    
# 定义GAT模型
class GAT(torch.nn.Module):  
    def __init__(self, num_features, num_classes, hidden_channels):  
        super(GAT, self).__init__()  
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=0.6)  
        self.conv2 = GATConv(8 * hidden_channels, num_classes, heads=1, concat=False, dropout=0.6)  
  
    def forward(self, x, edge_index, batch):  
        x = F.dropout(x, p=0.6, training=self.training)  
        x = F.elu(self.conv1(x, edge_index))  
        x = F.dropout(x, p=0.6, training=self.training)  
        x = self.conv2(x, edge_index)  
  
        # 由于我们使用的是单头输出，所以直接全局平均池化即可  
        return global_mean_pool(x, batch)  
    


def train(model, train_loader, optimizer, criterion, device):
    model.train().to(device)
    for data in train_loader:
        data.x = data.x.to(device)         # 将节点特征转移到GPU上  
        data.edge_index = data.edge_index.to(device)  # 将边索引转移到GPU上  
        data.y = data.y.to(device)         # 将标签转移到GPU上（如果有的话）  
        data.batch = data.batch.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        

def validate(model, val_loader, criterion, device):  
    model.eval()  # 设置模型为评估模式  
    val_loss = 0  
    correct = 0  
    total = 0  
    y_true = []  
    y_pred = []  
    with torch.no_grad():  # 不计算梯度  
        for data in val_loader:  
            data.x = data.x.to(device)  
            data.edge_index = data.edge_index.to(device)  
            data.y = data.y.to(device)  
            data.batch = data.batch.to(device)
            out = model(data.x, data.edge_index, data.batch)  
            loss = criterion(out, data.y)  
            val_loss += loss.item() * data.num_graphs  
            _, predicted = torch.max(out, 1)  
            total += data.y.size(0)  
            correct += (predicted == data.y).sum().item()  
            y_true.extend(data.y.cpu().numpy())  
            y_pred.extend(predicted.cpu().numpy())  
            
    # 计算混淆矩阵  
    cm = confusion_matrix(y_true, y_pred) 
    
    # 计算每个类别的精确率、召回率和F1值  
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None)  
      
    avg_val_loss = val_loss / len(val_loader.dataset)  
    val_acc = 100 * correct / total  
    return avg_val_loss, val_acc, precision, recall, f1_score, cm, y_true, y_pred

        
def test(loader, model, device):
    model.eval().to(device)  # 将模型设置为评估模式并转移到GPU上  
    correct = 0
    for data in loader:                            # 批遍历测试集数据集。
        data.x = data.x.to(device)         # 将节点特征转移到GPU上  
        data.edge_index = data.edge_index.to(device)  # 将边索引转移到GPU上  
        data.y = data.y.to(device)         # 将标签转移到GPU上（如果有的话）
        data.batch = data.batch.to(device)
        out = model(data.x, data.edge_index, data.batch) # 一次前向传播
        pred = out.argmax(dim=1)                         # 使用概率最高的类别
        correct += int((pred == data.y).sum())           # 检查真实标签
    return correct / len(loader.dataset)
    

def write_to_csv(output, repr, repr_list_epochs):
    # CSV文件的名称  
    csv_repr_output = output+repr+'.csv'  
    # 生成epoch序列，这里假设epoch从0开始，依次递增  
    epochs = list(range(len(repr_list_epochs)))  
    # 合并epoch和对应的数字列表  
    combined_data = [[epoch] + data for epoch, data in zip(epochs, repr_list_epochs)]  
    # print(combined_data)
    # 将数据写入CSV文件  
    with open(csv_repr_output, mode='w', newline='') as csv_file:  
        writer = csv.writer(csv_file)         
        # 写入列名  
        writer.writerow(['Epoch'] + ['Class {}'.format(i+1) for i in range(len(repr_list_epochs[0]))])  
        # 写入数据  
        for row in combined_data:  
            writer.writerow(row)  
    
    print(f"Data has been saved to {csv_repr_output}")


def main():
    args = parse_options()
    input = args.input
    batch_size = args.size
    num_epochs = args.epoch
    output = args.output
    choose_model = args.model
    
    # 加载数据集  
    dataset = MyDataset(input)
    print(input)
    
    # 定义参数模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    # 确定输入特征和输出类别的数量  
    num_features = dataset.num_node_features  
    num_classes = dataset.num_classes  
    
    train_dataset,val_dataset,test_dataset = split_dateset(dataset)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if choose_model == 'gcn':
        model = GCN(num_features,num_classes, hidden_channels=64)
    elif choose_model == 'ggnn':
        model = GraphGNN(num_features, num_classes, hidden_channels=64)
    elif choose_model == 'gat':
        model = GAT(num_features, num_classes, hidden_channels=64)
    else:
        print(f"{choose_model} is not our model we provided! you can try -gcn-, -ggnn-, -gat-")
    # print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 初始化CSV文件的写入  
    fieldnames = ['Epoch', 'Validation Loss', 'Validation Acc', 'Train Acc', 'Test Acc']  
    output_csv = output+'training_results.csv'
    with open(output_csv, 'w', newline='') as csvfile:  
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  
        writer.writeheader()  
    
    precision_list = []
    recall_list = []
    f1_score_list = []
    
    for epoch in range(1, num_epochs+1):
        train(model, train_loader, optimizer,criterion, device)
        train_acc = test(train_loader, model, device)
        val_loss, val_acc, precision, recall, f1_score, cm, y_true, y_pred = validate(model, val_loader, criterion, device)  
        test_acc = test(test_loader, model, device)
        
        print(f'Epoch: {epoch:03d}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        # 将结果写入CSV文件  
        with open(output_csv, 'a', newline='') as csvfile:  
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  
            writer.writerow({'Epoch': epoch, 'Validation Loss': val_loss, 'Validation Acc': val_acc, 'Train Acc': train_acc, 'Test Acc': test_acc})  
                
        # print('Precision per class:', precision)
        precision_list.append(precision.tolist())
        # print(precision_list)
        # print('Recall per class:', recall)  
        recall_list.append(recall.tolist())
        # print('F1 Score per class:', f1_score) 
        f1_score_list.append(f1_score.tolist())
        
        
    write_to_csv(output, 'Precision', precision_list)
    write_to_csv(output, 'Recall', recall_list)
    write_to_csv(output, 'F1_score', f1_score_list)
        
        
        

if __name__ == '__main__':
    main()