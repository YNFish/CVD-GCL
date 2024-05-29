from sklearn.preprocessing import LabelEncoder
import pandas as pd
import gensim
import re
import os
from torch_geometric.data import Data
import argparse
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
import warnings
warnings.filterwarnings('ignore')


def get_parse():
    parser = argparse.ArgumentParser(description='csv to vec ')
    parser.add_argument('-i', '--inputfile', help='The file of input', type=str, default='./Origin_dataset/csv/FFmpeg/')
    parser.add_argument('-o', '--outfile', help='The file of output', type=str, default='./Graph_dataset/detection/FFmpeg')
    parser.add_argument('-r', '--repr', help='The repr of Graph (ast, cfg, pdg, cpg)', type=str, default='cpg')  # 不同的图 
    parser.add_argument('-m', '--model', help='The model of embedding', type=str, default='./GloVe/Glove_Vector')  # GloVe 训练的向量表
    
    args = parser.parse_args()
    return args

# ------------------------------------------------- 输入参数处理 begin --------------------------------------------
args = get_parse()
repr = args.repr
# print(repr)
input_file = args.inputfile
output_file = args.outfile

if input_file[-1] == '/':
        input_file = input_file
else:
    input_file += '/'

if output_file[-1] == '/':
    output_file = output_file
else:
    output_file += '/'

embedding_model = args.model


folder_name_input = input_file
print(folder_name_input)


folder_name_output = output_file+repr+'/'
print(folder_name_output)

# ------------------------------------------------- 输入参数处理 end --------------------------------------------

# 导入embedding 模型
model = gensim.models.KeyedVectors.load(embedding_model)

repr_to_value = {'AST': 8, 'DDG': 4, 'CFG': 2, 'CDG': 1}
detection_lab = {'Vul': 1,
                 'NoVul': 0}


def process_string(x):
    x = x.replace(r'\012','\n').replace('&lt;', '<')\
        .replace('&gt;','>').replace('&amp','&').replace('&quot','"')
    x = x.replace('(', ' ( ').replace(')', ' ) ').replace('{', ' { ').replace('}', ' } ')\
        .replace('!',' ').replace(';',' ; ').replace(',',' , ')\
        .replace(':',' : ').replace('"',' " ')
    x = x.replace("\\n"," ")
    x = x.replace("\n"," ")
    x = re.sub(r'/\*.*?\*/', '', x)
    x = re.sub(r'\s{2,}', ' ', x)
    return x


def process_graphs(folder_path, repr): # 输入上层目录  ./Origin_dataset/csv/FFmpeg/
    graph_data_list = []  
    
    nodecsv = f'{repr}-node.csv'
    edgecsv = f'{repr}-edge.csv'
    folders = os.listdir(folder_path)  # ['Vul', 'NoVul']
    for subfolder in folders:
        label = detection_lab[subfolder]
        print(label)
        subfolder = subfolder+'/'+repr+'s_csv'
        subfolders = os.listdir(folder_path+subfolder)  # 找到 项目 下面， repr 类型的所有文件
        for subsubfolder in subfolders:
            print(os.path.join(folder_path, subfolder, subsubfolder, nodecsv))
        
            if os.path.exists(os.path.join(folder_path, subfolder, subsubfolder, nodecsv)) and os.path.exists(os.path.join(folder_path, subfolder, subsubfolder, edgecsv)):
                node_df = pd.read_csv(os.path.join(folder_path, subfolder, subsubfolder, nodecsv))
                edge_df = pd.read_csv(os.path.join(folder_path, subfolder, subsubfolder, edgecsv))
                # print(node_df, edge_df)

                # 读取节点的特征
                original_node_ids = node_df['node'].values
                # print(original_node_ids)

                node_sentences = node_df['true_code'].apply(process_string)
                node_vecs = np.array([np.mean([model[w] for w in words if w in model]
                                            or [np.zeros(model.vector_size)], axis=0)
                                    for words in node_sentences])
                operator_vec = np.array([model[w] if w in model else np.zeros(model.vector_size)
                            for w in node_df['operator']])
                
                subscript_vec = node_df['subscript'].values.reshape(-1,1)
                # print(node_vecs.shape, operator_vec.shape, subscript_vec.shape)
                combined_vecs = np.concatenate((node_vecs, operator_vec), axis=1)
                combined_vecs = np.concatenate((combined_vecs, subscript_vec), axis=1)
                
                node_features = torch.tensor(combined_vecs, dtype=torch.float)
                
                # 获取节点编号的最小值和最大值  
                # min_node_id = original_node_ids.min()  
                # max_node_id = original_node_ids.max()  
                
                # 为节点重新编号并创建一个映射
                node_id_map = {original_id: new_id for new_id, original_id in enumerate(original_node_ids)}   # 师弟要注意，踩坑了
                # print(node_id_map)
                
                # 将repr字段的值转换为相应的数值  
                edge_df['repr_value'] = edge_df['repr'].map(repr_to_value)

                # 使用映射更新边信息中的节点编号  
                edge_df['source'] = edge_df['source'].map(node_id_map)  
                edge_df['distination'] = edge_df['distination'].map(node_id_map)  
                
                # 按source和destination分组，并计算repr_value的总和  
                edge_df_merged = edge_df.groupby(['source', 'distination'], as_index=False)['repr_value'].sum() 
                # print(edge_df_merged)

                # 提取合并后的边的起点、终点和权重  
                edge_index = torch.tensor([edge_df_merged['source'].values, edge_df_merged['distination'].values], dtype=torch.long).contiguous()
                edge_weight = torch.tensor(edge_df_merged['repr_value'].values, dtype=float) 
                    
                data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, y=label)
                
                print(data)
                graph_data_list.append(data)
        
        # torch.save(graph_data_list, 'cpg_data.pt')
    return graph_data_list
    

class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform, pre_filter)  # transform是数据增强，对每个数据都执行
        # self.repr = repr_param
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):  
    # 检查self.raw_dir目录是否存在raw_file_names()属性方法返回的每个文件，如果文件不存在，则调用download()方法执行原始文件下载
        return []
    
    @property
    def processed_file_names(self):  
    # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = process_graphs(folder_name_input, repr=repr)  # 要修改
        print(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    
def main():
    dataset = YooChooseBinaryDataset(root = folder_name_output)
    # process_graphs(folder_name_input, repr=repr)
    print(folder_name_input, " make graph successed")
    
    
    
if __name__ == '__main__':
    main()