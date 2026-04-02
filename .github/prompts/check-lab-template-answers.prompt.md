---
name: "检查实验模板是否泄露答案"
description: "检查实验 notebook 里学生应填写的代码区是否仍为空模板，并区分是否仅保留了历史输出"
argument-hint: "输入实验目录、附件目录或 notebook 路径；不填则优先检查当前打开的 notebook 及其同级实验目录"
agent: "agent"
model: "GPT-5 (copilot)"
---

请检查用户给定的实验目录或 notebook，判断学生需要完成的部分是否还保持为空模板，还是已经被误填入正确答案。

工作要求：

1. 先确定检查范围。
如果用户提供了目录或文件路径，就以该路径为准。
如果用户没有提供，就优先检查当前打开的 notebook；如果它位于某个实验目录中，再顺带检查同级相关 notebook。

2. 重点检查所有 `.ipynb` 文件中的练习代码区。
优先搜索这些标记或模式：
- `### START CODE HERE ###`
- `### END CODE HERE ###`
- `YOUR CODE HERE`
- `TODO`
- `None`
- `需要完成的函数`
- `练习`

3. 判定标准必须分开处理，不要混淆。
- 如果 `START/END` 之间只有空白、注释或占位内容，判定为“留空正常”。
- 如果 `START/END` 之间出现真实实现代码，判定为“已填入答案”。
- 如果代码区仍为空，但 notebook 保存了测试输出、训练日志、图片或 `All tests passed`，判定为“答案未泄露，但输出未清空”。

4. 尽量自动化扫描，不要只做人工抽样。
优先使用下面这段 Python 脚本在工作区里扫描目标目录下的所有 notebook，并输出每个 notebook 的练习块状态与已保存输出的代码单元。

```python
from pathlib import Path
import json

root = Path(TARGET_PATH)
targets = [root] if root.is_file() else sorted(root.rglob('*.ipynb'))

for path in targets:
    if path.is_dir():
        continue
    nb = json.loads(path.read_text(encoding='utf-8'))
    blocks = []
    output_cells = []

    for idx, cell in enumerate(nb.get('cells', []), start=1):
        if cell.get('cell_type') == 'code' and cell.get('outputs'):
            output_cells.append(idx)

        if cell.get('cell_type') != 'code':
            continue

        src = cell.get('source', [])
        lines = src if isinstance(src, list) else src.splitlines(True)
        starts = [i for i, line in enumerate(lines) if 'START CODE HERE' in line]
        ends = [i for i, line in enumerate(lines) if 'END CODE HERE' in line]

        for start in starts:
            end = next((e for e in ends if e > start), None)
            if end is None:
                blocks.append((idx, 'missing-end', []))
                continue

            inner = lines[start + 1:end]
            meaningful = []
            for raw in inner:
                text = raw.strip()
                if not text:
                    continue
                if text.startswith('#'):
                    continue
                meaningful.append(text)

            blocks.append((idx, 'filled' if meaningful else 'blank', meaningful[:3]))

    print(f'FILE: {path}')
    print(f'  practice_blocks={len(blocks)}')
    print(f'  saved_output_cells={output_cells}')
    for idx, status, sample in blocks:
        print(f'  cell {idx}: {status}')
        if sample:
            print(f'    sample={sample}')
```

执行脚本时：
- 把 `TARGET_PATH` 替换成用户给出的目录或 notebook 绝对路径。
- 如果是当前工作区中的路径，优先使用 Python 代码执行工具直接运行，不要为了这件事创建临时脚本文件。

5. 脚本扫描后，再对可疑位置做二次核验。
- 对所有 `filled` 的块，读取原 notebook 对应位置，确认是不是标准答案、辅助代码或误判。
- 对所有存在 `saved_output_cells` 的 notebook，抽查输出内容，确认只是运行结果，而不是把答案直接写回代码区。

6. 同时检查 git 状态。
- 看目标仓库是否有未提交改动。
- 如果没有改动，要明确说“当前仓库没有本地未提交改动”。
- 如果有改动，只报告事实，不要擅自回退。

7. 最终回答必须简洁，但要覆盖这四点：
- 检查了哪些文件
- 哪些练习块仍为空模板
- 是否有任何代码区被填入答案
- 是否存在未清空的输出或运行痕迹

输出格式建议：

`结论：` 一句话给出总判断。

`细节：`
1. 按 notebook 列出练习块数量与结论。
2. 如果有泄露答案的位置，给出具体文件和单元位置。
3. 如果只有输出未清空，也要单独指出。

`补充：`
- git 状态结论。
- 如果合适，提出“是否需要我顺手清空 notebook 输出”。

约束：
- 不要修改任何文件，除非用户明确要求你清空输出或恢复模板。
- 不要只凭 notebook 里已有输出就判断“答案已经泄露”。
- 对于 `START/END` 之外的教学示例代码，不要误判为学生答案。
