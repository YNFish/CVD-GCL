import os

def dealDot(dot_file, output_file):
    try:
        with open(dot_file, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError as e:
        print(f'{dot_file} is not exised',e)
        return
    
    # print(lines)
   # 逐行处理并替换字符
    corrected_lines = []
    corrected_lines = [lines[0]]  # 将第一行添加到修正后的行列表中
    for line in lines[1:-1]:
        if "->" not in line:
            corrected_line = line.replace("<", "\"", 1).rsplit(">", 1)
            if len(corrected_line) == 2:
                corrected_line = corrected_line[0] + "\"" + corrected_line[1]
            corrected_lines.append(corrected_line)
        else:
            corrected_lines.append(line)


    # 直接将最后一行替换为“}”
    last_line = "}"
    corrected_lines.append(last_line)
    
    output_dir = os.path.dirname(output_file)
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} is not exist, Created sucess!")
    
    # 将修正后的内容写入输出文件
    with open(output_file, 'w') as file:
        file.write("".join(str(line) for line in corrected_lines))
    
    
# 获取子文件夹
def get_subfolders(folder_path):
    subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            subfolders.append(subfolder_path)
    return subfolders

    
if __name__=='__main__':
    print("-------------------begin-------------------")
    # 修改下面两个
    dot_dir = "dataset/picAll/picAllpdg"
    file_type = "-pdg.dot"
    output_dir = "dataset/Alldot"
    subfolders = get_subfolders(dot_dir)
    print(subfolders)
    for subfolder in subfolders:
        file_dir = subfolder.split('/')[-1]
        dot_file = subfolder+'/1'+file_type
        output_file = output_dir+"/"+file_dir+"/1"+file_type
        print(dot_file, output_file)
        dealDot(dot_file, output_file)
        