---
name: paper-reading
description: 论文精读技能。根据用户给出的论文名称，从 arXiv 搜索并下载论文 PDF，然后生成一篇通俗易懂的中文 Markdown 论文解读。适用于用户提到"论文精读"、"读论文"、"解读论文"、"paper reading"等场景。
---

# 论文精读

根据用户给出的论文名称或 arXiv ID，下载论文到 paper 目录，并生成一篇面向初学者的中文 Markdown 论文解读。

## 前置依赖

本 Skill 依赖 **arxiv MCP 服务**（`arxiv-mcp-server`）。请确保项目根目录的 `.mcp.json` 已被你的 AI 平台加载，或手动配置：

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "uvx",
      "args": ["arxiv-mcp-server", "--storage-path", "."]
    }
  }
}
```

> MCP 服务名配置为 `arxiv` 时，工具调用名为 `mcp__arxiv__<tool>`（如 `mcp__arxiv__search_papers`）。若你的平台使用其他服务名（如 `arxiv-mcp-server`），则前缀相应变为 `mcp__arxiv-mcp-server__<tool>`。下文统一使用 `arxiv` 作为服务名。

## 工作流程

### Step 1: 确定论文信息

从用户输入中提取：
- **论文名称**或 **arXiv ID**（如 `2401.12345`）
- 如果用户给出的是模糊的名称，先用 WebSearch 确认论文全名和 arXiv ID

### Step 2: 搜索论文

**首选方式 — arXiv MCP**：

1. **按名称搜索**：使用 `mcp__arxiv__search_papers` 工具
   - 用论文标题作为 query，如 `ti:"Attention Is All You Need"`
   - 设置 `max_results: 5`，按 `relevance` 排序
   - 返回匹配论文列表，确认目标论文的 arXiv ID

2. **已知 arXiv ID**：直接使用 `mcp__arxiv__get_abstract` 获取元信息
   - 返回：标题、作者、摘要、分类、发表日期、PDF URL

**降级方式 — 下载脚本（MCP 不可用时）**：

```bash
# 从项目根目录运行（需要先激活虚拟环境）
python skills/paper-reading/scripts/download_arxiv.py "<论文名称或arXiv ID>" "<保存目录>"
```

脚本会自动搜索 arXiv、下载 PDF 并返回论文元信息 JSON。若脚本也失败，使用 WebSearch 搜索论文 arXiv 页面。

### Step 3: 下载论文

**首选方式 — arXiv MCP**：

1. 使用 `mcp__arxiv__download_paper`，传入 `paper_id`（如 `"1706.03762"`）
   - 该工具自动下载并提取全文（优先 HTML，回退 PDF）
   - 返回论文全文内容（Markdown 格式），可直接阅读

2. 在用户的 paper 目录下创建**论文专属文件夹**，并用 `curl` 下载 PDF：
   ```bash
   mkdir -p "<paper目录>/<论文简称>"
   curl -L -o "<paper目录>/<论文简称>/<论文简称>.pdf" "https://arxiv.org/pdf/<paper_id>.pdf"
   ```
   论文简称 = 英文标题中每个主要单词首字母大写、用下划线连接，如 `LoRA_Low-Rank_Adaptation_of_Large_Language_Models`

**降级方式**：使用 `skills/paper-reading/scripts/download_arxiv.py` 脚本完成搜索+下载。

### Step 4: 阅读论文全文

**首选方式 — arXiv MCP**：

使用 `mcp__arxiv__read_paper` 读取已下载论文的全文（Markdown 格式）。

**降级方式**：使用 PDF 解析工具从 PDF 文件提取论文内容。

**阅读时重点关注**：
- Abstract（摘要）
- Introduction（引言）— 问题定义和研究动机
- Method / Approach（方法）— 核心技术方案
- Experiments / Results（实验结果）— 效果和对比
- Conclusion（结论）
- 关键图表的描述

### Step 4.5: 提取论文原图

**原则：论文中的图优先从 PDF 提取原图使用；若原图提取失败（如矢量图裁剪不干净、PDF 加密无法提取等），则降级为自绘 ASCII 图表替代。自绘图也可用于论文中没有的概念性说明。**

在论文专属文件夹下创建 `resource/` 子目录存放提取的图片。

**提取方式 — 使用 `extract_pdf_figures.py` 脚本**：

```bash
# 从项目根目录运行（需要先激活虚拟环境）
# 自动检测：渲染含 Figure 的页面为高清截图
python skills/paper-reading/scripts/extract_pdf_figures.py "<论文专属文件夹>/<论文简称>.pdf" "<论文专属文件夹>/resource" --dpi 300

# 手动裁剪指定区域（坐标为 PDF 坐标系，单位 pt，原点左上角）
python skills/paper-reading/scripts/extract_pdf_figures.py "<论文专属文件夹>/<论文简称>.pdf" "<论文专属文件夹>/resource" --crop <页码>:<x0>,<y0>,<x1>,<y1> --name <figure名称> --dpi 300
```

**提取流程**：

1. 先运行自动检测模式，获取各 Figure 标题所在页码和 y 坐标
2. 查看全页截图（`pageN_full_300dpi.png`），确定每个 Figure 的精确裁剪坐标
3. 用手动裁剪模式逐一提取干净的 Figure 图片
4. 裁剪时注意：排除周围正文文字，仅保留图形本体和标题（caption）

**命名规范**：`figure1_<简要描述>.png`，如 `figure1_lora_reparametrization.png`

### Step 5: 生成论文解读

参照 [template.md](template.md) 的模板格式，用**中文**撰写论文解读。

**写作原则**：

1. **面向小白**：假设读者没有相关背景知识，用生活化的类比和例子解释专业概念
2. **讲清故事线**：按"问题 → 动机 → 方法 → 效果"的逻辑组织
3. **善用类比**：对核心方法用通俗比喻辅助说明（如"这就像..."）
4. **不堆术语**：术语首次出现时给出简单解释，后续用中文替代或加括号注明
5. **突出亮点**：明确指出这篇论文最大的创新点和贡献

**图表要求（重要）**：

解读中必须包含图表来辅助说明，不能全是文字叙述。图表分为两类：

**A. 论文原图（优先使用）**：

论文中已有的 Figure 必须使用 Step 4.5 提取的原图，用 Markdown 图片语法引用：
```markdown
![Figure 1: 描述](./resource/figure1_xxx.png)
```
不要用 ASCII 重画论文中已有的图。

**B. 自绘辅助图（补充使用 / 原图提取失败时的降级方案）**：

以下场景可自行绘制 ASCII 图表：
1. **原图提取降级**：论文中有对应 Figure，但提取失败（矢量图裁剪不干净、嵌入图分辨率过低、PDF 加密等），用 ASCII 图表近似还原
2. **方法对比图**：论文中没有的横向对比图，将本文方法与现有方案并排展示
3. **流程推导图**：配合数值推导示例的计算流向图
4. **梯度流向图**：标注哪些参数有梯度、哪些被冻结
5. **部署对比图**：训练时 vs 推理时的结构变化

ASCII 图表使用 Markdown 代码块（```）包裹，用 `─│┌┐└┘├┤┬┴┼▼▲►◄●★` 等字符绘制，确保等宽字体下对齐。

**数学公式要求**：

使用 LaTeX 语法书写数学公式，以便在 Obsidian 等工具中正确渲染：
- 行内公式：`$...$`，如 `$h = Wx + BAx$`
- 独立公式：`$$...$$`，如 `$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$$`
- 矩阵使用 `\begin{bmatrix}...\end{bmatrix}`
- 不要使用纯文本近似写法（如 `W ∈ R^{d×d}`），统一用 LaTeX 语法

**流程推导要求（重要）**：

对于涉及模型训练的论文，必须包含一个**具体数值的简化示例**，手把手走通核心流程，帮助读者建立直觉。推导需覆盖：

1. **场景设定**：给出一个极简的数值设定（如 4×4 矩阵），标明哪些参数可训练、哪些冻结
2. **前向传播**：用具体数字展示输入如何经过各模块得到输出，标注中间变量的维度变化
3. **损失函数**：写出损失函数的数学形式，并用直白的中文翻译其含义，对比本文方法与传统方法在优化目标上的区别
4. **反向传播**：画出梯度流向图，标明梯度流向哪些参数、不流向哪些参数，用数值展示一步梯度计算
5. **参数更新与下一轮迭代**：用具体数值展示完整的"梯度→更新→下一轮前向"循环，包括：
   - 写出更新公式（$\theta_{\text{new}} = \theta - \eta \cdot \nabla_\theta \mathcal{L}$），解释学习率的作用
   - 用 Step 4 算出的梯度和一个具体的学习率（如 $\eta = 0.1$），逐元素算出更新后的参数值
   - 用更新后的参数对同一输入再做一次前向传播，算出新的输出和新的损失
   - 用对比表格或列表展示：每个输出分量是否朝目标方向移动了、损失是否下降了
   - 如果存在"冷启动"现象（如某参数初始为零导致另一参数第一轮无梯度），明确指出从第几轮开始所有参数都参与更新
   - 用 ASCII 图画出多轮迭代的全景图，展示训练如何逐步收敛
6. **部署/推理**：如果方法在推理阶段有特殊处理（如权重合并、模块移除），用图示说明

注意：推导的目的是建立直觉，不是严格数学证明。数值可以简化，但维度变化和计算流程必须正确。

**术语解释要求**：

首次出现的专业缩写（如 MLP、FFN、LSTM、GRU 等）必须给出全称和通俗解释。例如 MLP 首次出现时应标注"MLP（Multi-Layer Perceptron，多层感知机）——最基础的神经网络结构，多层线性变换加激活函数"。后续可直接使用缩写。

### Step 6: 保存解读文件

将生成的 Markdown 保存到论文专属文件夹内，文件名格式：

```
<论文简称>_解读.md
```

例如：`Attention_Is_All_You_Need/Attention_Is_All_You_Need_解读.md`

### Step 7: 自动提交并推送

解读文件保存后，自动将本次新增的论文文件夹提交到 git 仓库并推送到远程。

```bash
cd <paper目录>
git add "<论文简称>/"
git commit -m "docs(<论文简称>): 添加论文解读

- 下载论文 PDF
- 提取论文原图至 resource/
- 生成中文解读（含 LaTeX 公式、数值推导、原图引用）"
git push
```

**注意事项**：
- 提交范围仅限当前论文的专属文件夹，不要把其他未完成的论文一起提交
- 如果远程仓库尚未配置（`git remote` 为空），跳过 push 并提示用户先配置远程仓库
- 如果 push 失败（网络问题等），提示用户手动 push，不要重试

## 输出规范

- 语言：中文
- 格式：Markdown（LaTeX 公式使用 `$...$` / `$$...$$`）
- 长度：2000-5000 字（根据论文复杂度调整）
- 推荐阅读工具：Obsidian（免费，原生支持 LaTeX 渲染和图片引用）

### 目录结构规范

每篇论文在 paper 目录下拥有独立的文件夹，所有相关文件集中管理：

```
paper/
├── Attention_Is_All_You_Need/               # 论文专属文件夹
│   ├── Attention_Is_All_You_Need.pdf        # 论文原文 PDF
│   ├── Attention_Is_All_You_Need_解读.md    # 论文解读
│   └── resource/                            # 解读引用的资源
│       ├── figure1_transformer_arch.png     # 从 PDF 提取的原图
│       ├── figure2_attention_detail.png
│       └── extraction_summary.json          # 图片提取摘要
│
├── LoRA_Low-Rank_Adaptation/                # 另一篇论文
│   ├── LoRA_Low-Rank_Adaptation.pdf
│   ├── LoRA_Low-Rank_Adaptation_解读.md
│   └── resource/
│       ├── figure1_reparametrization.png
│       └── figure2_accuracy_vs_params.png
│
└── ...                                      # 更多论文
```

**关键约定**：
- 论文文件夹名 = 论文简称（与 PDF 文件名一致，不含扩展名）
- 解读 Markdown 中引用资源使用**相对路径** `./resource/xxx.png`
- `resource/` 目录仅存放解读所需的资源（提取的原图、自制图表等）
- 全页截图等临时文件（`pageN_full_300dpi.png`）在裁剪完成后可清理，不必保留

## 可用 MCP 工具速查

以下工具由 `arxiv-mcp-server` 提供（工具前缀取决于你的 MCP 配置中的服务名，下文以 `arxiv` 为例）：

| 工具 | 用途 | 关键参数 |
|------|------|----------|
| `mcp__arxiv__search_papers` | 搜索论文 | query, max_results, categories, sort_by |
| `mcp__arxiv__get_abstract` | 获取摘要和元信息 | paper_id |
| `mcp__arxiv__download_paper` | 下载并提取全文 | paper_id |
| `mcp__arxiv__read_paper` | 读取已下载论文 | paper_id |
| `mcp__arxiv__citation_graph` | 获取引用/被引关系 | paper_id |
| `mcp__arxiv__semantic_search` | 本地语义搜索 | query (需先下载论文) |

## 注意事项

- **优先使用 MCP 工具**，MCP 不可用时自动降级为脚本方式
- 降级脚本 `download_arxiv.py` 仅使用 Python 标准库，无需额外安装依赖
- `extract_pdf_figures.py` 需要 PyMuPDF，通过 `requirements.txt` 安装
- arXiv API 有频率限制（3秒间隔），遇到限流稍等重试，不要循环调用
- 如果论文不在 arXiv 上，告知用户并建议手动提供 PDF
