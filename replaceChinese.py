import sys
import re

"""
该脚本用于替换给定 .tex 文件中的中文逗号“，”为英文逗号加空格", "，
以及中文句号“。”为英文句号加空格". ".
用法:
    python replaceChineseComma.py <input_file>
    <input_file>: 要处理的 .tex 文件路径.
"""

def replace_chinese_punctuation(tex_content):
    # 替换中文逗号为英文逗号加空格
    text = re.sub(r'，', ', ', tex_content)
    # 替换中文句号为英文句号加空格
    text = re.sub(r'。', '. ', text)
    # 替换中文冒号为英文冒号加空格
    text = re.sub(r'：', ': ', text)
    # 替换中文括号为英文括号加空格
    text = re.sub(r'（', '(', text)
    text = re.sub(r'）', ')', text)
    # 替换中文分号为英文分号加空格
    text = re.sub(r'；', '; ', text)
    # 替换中文问号为英文问号加空格
    text = re.sub(r'？', '? ', text)
    # 替换中文叹号为英文叹号加空格
    text = re.sub(r'！', '! ', text)
    # 替换中文省略号为英文省略号加空格
    text = re.sub(r'……', '... ', text)
    # 替换中文破折号为英文破折号加空格
    text = re.sub(r'——', '-- ', text)

    return text

def main(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        tex_content = file.read()
    
    new_content = replace_chinese_punctuation(tex_content)
    
    with open(input_file, 'w', encoding='utf-8') as file:
        file.write(new_content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python replaceChineseComma.py <input_file>")
    else:
        main(sys.argv[1])