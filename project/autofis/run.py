import os
import chardet
# 获取当前目录
dir_path = os.getcwd()

# 获取当前目录下的所有文件名
files = os.listdir(dir_path)

# 根据最后修改时间对文件进行排序
files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)))
files = [f for f in files if f != 'run.py']
# 选择最新修改的文件
newest_file = files[-1]

# 输出最新修改的文件名
print(newest_file)

# 打开最新修改的文件，读取文件内容
with open(newest_file, 'r', encoding='UTF-8') as f:
    file_content = f.read()
    # encoding = chardet.detect(file_content)['encoding']
    # # 将文件内容解码为字符串
    # file_content = file_content.decode(encoding)

# 在文件内容中进行修改
modified_content = file_content.replace('Text is not SVG - cannot display', ' ')
# 请在上一行代码中将 'old_string' 替换为您要修改的字符串，并将 'new_string' 替换为您要替换成的字符串

# 将修改后的内容写回文件
with open(newest_file, 'w', encoding='UTF-8') as f:
    f.write(modified_content)

print("修改完成")