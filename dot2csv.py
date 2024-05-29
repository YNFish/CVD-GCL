import networkx as nx
import re
import pandas as pd
import os
import argparse
import glob
import dealDot

# error_list_over = []


def get_parse():
    parser = argparse.ArgumentParser(description='dot2csv')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='./testdata/sard/pdgs/Vul')
    parser.add_argument('-o', '--output', help='The dir path of output', type=str, default='./testdata/sard/csv/Vul')
    parser.add_argument('-r', '--repr', help='The type of graph(ast, cfg, pdg, cpg)', type=str, default='pdg')
    args = parser.parse_args()
    return args


def dot_to_gcl(dot_file, outputdir, repr):
    """
    @dot_file: 输入的dot文件
    @outputdir: 输出路径
    """
    # ./testdata/sard/pdgs/Vul/CVE_raw_000062516_CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_01_bad.dot
    # dataset/Alldot/FFmpeg2_0/1-cpg.dot
    # print(dot_file)
    filename_with_extension = os.path.basename(dot_file)
    
    filename = os.path.splitext(filename_with_extension)[0]
    print(filename)
    
    if os.path.exists(outputdir+filename):
        print(f"*************************** {outputdir+filename} is existed!")
        # return
    # file_dir = "dataset/node_edge_dataset/"
    node_attr_csv = outputdir+filename+"/"+ f"{repr}-node.csv"
    edge_attr_csv = outputdir+filename+"/"+ f"{repr}-edge.csv"
    
    output_dir = os.path.dirname(node_attr_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} is not exist, Created success!")
    
    # print(edge_attr_csv)
    # 解析.dot文件并构图
    try:
        G = nx.drawing.nx_pydot.read_dot(dot_file)
    except:
        print("-------------------------------- error is happened---------------------------")
        return
    
    # 创建空的DataFrame
    dfs = []
    for node, attrs in G.nodes(data=True):
        replaced_attrs = attrs['label'].replace('&lt;', '<').replace('&gt;','>').replace('&amp','&').replace('&quot','"') # 将所有的<>转运字符全部替换
        # print(replaced_attrs)
        # 获取 operator
        left_paren_index = replaced_attrs.find('(') # 第一个左括号
        comma_index = replaced_attrs.find(',') # 第一个逗号
        right_paren_index = replaced_attrs.rfind(')') # 最后一个右括号
        operator = replaced_attrs[left_paren_index+1:comma_index]
        true_code = replaced_attrs[comma_index+1:right_paren_index]
        subscript = re.findall(r'<SUB>(\d+)</SUB>',replaced_attrs)
        if subscript ==[]:
            # os.remove(output_dir)
            # error_list_over.append(dot_file)
            return
        else:
            subscript = subscript[0]
        # df = df.append({'node':node, 'operator': operator, 'true_code': true_code, 'subscript': subscript}, ignore_index=True)
        df = pd.DataFrame({'node': [node], 'operator': [operator], 'subscript': [subscript], 'true_code': [''.join(true_code)]})
        dfs.append(df)
    try:
        result_df = pd.concat(dfs, ignore_index=True)
    except ValueError as e:
        print(f'发生错误：{e}')
        result_df = None
    if result_df is not None:
        result_df.to_csv(node_attr_csv, index=False)
    else:
        return
    
    # 保存节点属性
    # edge_attributes = {}
    dfedge = []
    for edge in G.edges:
        # edge --> ('449', '450', 0) (源结点，目标节点，不知道)
        attributes = G.get_edge_data(*edge)
        if attributes == {}:   # 主要为解决ast等图中的边没有label这一个问题
            edge_attr = ""
        else:
            edge_attr = attributes['label'].replace('&lt;', '<').replace('&gt;','>').replace('&amp','&').replace('&quot','"') # 将所有的<>转运字符全部替换
        # print(edge_attr)
        edge_repr = edge_attr[1:4]
        left_colon_index = edge_attr.find(':')
        right_quot_index = edge_attr.rfind('"')
        # print(left_colon_index, right_quot_index)
        if left_colon_index+2 == right_quot_index:
            edge_code = ""
        else:
            edge_code = edge_attr[left_colon_index+2: right_quot_index]
            # print(edge_code)
        source_node = edge[0]
        distination_node = edge[1]
        dfe = pd.DataFrame({'source': [source_node], 'distination': [distination_node], 'repr':[edge_repr], 'code':[edge_code]})
        dfedge.append(dfe)
        # edge_attributes[edge] = edge_attr
    result_df_edge = pd.concat(dfedge, ignore_index=True)
    result_df_edge.to_csv(edge_attr_csv, index=False)
        

def replace_in_dot_file(dot_file_path):
    # 打开 DOT 文件
    with open(dot_file_path, 'r') as file:
        dot_content = file.read()

    # 替换字符串
    modified_dot_content = dot_content.replace(' <(', ' "(').replace('>> ]', '>" ]')

    # 将修改后的内容写回到文件中
    with open(dot_file_path, 'w') as file:
        file.write(modified_dot_content)


def main():
    args = get_parse()
    dot_folder = args.input
    save_folder = args.output
    repr = args.repr
    
    if dot_folder[-1] == '/':
        dot_folder = dot_folder
    else:
        dot_folder += '/'

    if save_folder[-1] == '/':
        save_folder = save_folder
    else:
        save_folder += '/'
    
    dot_files = glob.glob(dot_folder+"*.dot")
    # print(dot_files)

    index = 0
    for dot_file in dot_files:
        # print("+++++++++++++++++++++++++++++++++++++++++   ",dot_file)
        replace_in_dot_file(dot_file)
        # print("dealDot successed   ",dot_file)
        dot_to_gcl(dot_file, save_folder, repr)
        index +=1
        print("-----------------------------------------  ")
    print(index,"--------------------end------------------")


 
if __name__ == "__main__":
    main()