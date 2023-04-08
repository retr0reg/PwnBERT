import re

def main():
    # dirs = ["vuln/","nvuln/","eval/vuln/","eval/nvuln/"]
    dirs = ["nvuln/"]
    for dirr in dirs:
    # 读取文件内容到字符串中
        with open(dirr+"outputs.txt", 'r') as f:
            content = f.read()

        # 使用正则表达式匹配每个程序代码
        # ^\s*#include.*?\breturn\s+0;\s*}
        # pattern = re.compile(r'^\s*#include\s+<stdlib\.h>.*?\breturn\s+0;\s*}', re.DOTALL | re.MULTILINE)
        pattern = re.compile(r'(\s*#include.*?return\s+0;\s*})', re.DOTALL)
        matches = re.finditer(pattern, content)

        # 将每个程序代码保存到不同的文件中
        for i, match in enumerate(matches):
            code = match.group(0)
            with open(f'{dirr}program_{i}.c', 'w') as f:
                f.write(code)
                
if __name__ == "__main__":
    main()
