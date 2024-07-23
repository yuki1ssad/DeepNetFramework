import os

def count_lines_in_file(file_path):
    """计算单个文件的行数"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)

def count_files_in_directory(directory, extensions):
    """遍历目录，统计指定扩展名的文件行数"""
    total_lines = 0
    file_counts = {ext: 0 for ext in extensions}

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                file_lines = count_lines_in_file(file_path)
                print(f'{file_path}: {file_lines}')
                file_counts[file.split('.')[-1]] += file_lines
                total_lines += file_lines

    return file_counts, total_lines

if __name__ == "__main__":
    directory = "/home/wangxuefei/workspace/DeepNetFramework"
    extensions = ['h', 'cpp', 'cu']  # 指定要统计的文件扩展名
    file_counts, total_lines = count_files_in_directory(directory, extensions)

    print("文件行数统计:")
    for ext, count in file_counts.items():
        print(f"{ext.upper()} 文件的行数: {count}")
    print(f"总行数: {total_lines}")
    
'''
运行的时候需要先执行 rm -rf build/*
结果 ~2000
'''