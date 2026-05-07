---
name: paper-reading
description: 论文精读技能。根据用户给出的论文名称，从 arXiv 搜索并下载论文 PDF，然后生成一篇通俗易懂的中文 Markdown 论文解读。适用于用户提到"论文精读"、"读论文"、"解读论文"、"paper reading"等场景。
---

# 论文精读

根据用户给出的论文名称或 arXiv ID，下载论文到 `papers/` 目录，并生成一篇面向初学者的中文 Markdown 论文解读。

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

### Step 0: 去重检查（重要）

在开始任何工作之前，**必须先检查论文是否已经精读过**。

**检查方法**：扫描 `papers/` 目录下所有子文件夹，如果存在同名文件夹且内部已有 `*_解读.md` 文件，说明该论文已精读完成。

```bash
# 列出所有已精读论文
ls papers/*/  # 查看文件夹列表
ls papers/*/_解读.md 2>/dev/null  # 查看已有解读文件
```

**判定逻辑**：
- 文件夹名与论文标题匹配（英文标题转下划线格式）且包含 `_解读.md` → **已精读，跳过**
- 文件夹存在但没有 `_解读.md`（只有 PDF）→ **未完成，继续精读**
- 文件夹不存在 → **未精读，正常开始**

**用户提交多篇论文时**：逐一检查，跳过已精读的，只处理未精读的。对于跳过的论文，告知用户"该论文已精读，跳过"。如果全部已精读，告知用户并列出已有解读文件路径。

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
   - **重要**：`get_abstract` 返回的 `published` 字段包含精确发表日期（如 `"2017-06-12T17:57:34Z"`），**必须记录这个精确日期**。后续生成解读时，`发表时间` 字段使用 `YYYY-MM-DD` 格式（如 `2017-06-12`），不要只写年份。
   
### Step 3: 下载论文

**首选方式 — arXiv MCP**：

1. 使用 `mcp__arxiv__download_paper`，传入 `paper_id`（如 `"1706.03762"`）
   - 该工具自动下载并提取全文（优先 HTML，回退 PDF）
   - 返回论文全文内容（Markdown 格式），可直接阅读
   - **重要**：MCP 下载可能会在项目根目录生成 `{arxiv_id}.md` 缓存文件。该文件必须立即移动到对应论文的 `resource/` 目录下，并重命名为 `temp_paper_{arxiv_id}.md`。禁止让任何论文相关文件散落在根目录。

2. 在项目的 `papers/` 目录下创建**论文专属文件夹**，并用 `curl` 下载 PDF：
   ```bash
   mkdir -p "papers/<论文简称>"
   curl -L -o "papers/<论文简称>/<论文简称>.pdf" "https://arxiv.org/pdf/<paper_id>.pdf"
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

### Step 5: 提取论文原图

**原则：论文中的图优先从 PDF 提取原图使用；若原图提取失败（如矢量图裁剪不干净、PDF 加密无法提取等），则降级为自绘 SVG 图表替代。自绘图也可用于论文中没有的概念性说明。**

在论文专属文件夹（`papers/<论文简称>/`）下创建 `resource/` 子目录存放提取的图片。

**提取方式 — 使用 `extract_pdf_figures.py` 脚本**：

```bash
# 从项目根目录运行（需要先激活虚拟环境）
# 自动检测：渲染含 Figure 的页面为高清截图
python skills/paper-reading/scripts/extract_pdf_figures.py "papers/<论文简称>/<论文简称>.pdf" "papers/<论文简称>/resource" --dpi 300

# 手动裁剪指定区域（坐标为 PDF 坐标系，单位 pt，原点左上角）
python skills/paper-reading/scripts/extract_pdf_figures.py "papers/<论文简称>/<论文简称>.pdf" "papers/<论文简称>/resource" --crop <页码>:<x0>,<y0>,<x1>,<y1> --name <figure名称> --dpi 300
```

**提取流程**：

1. 先运行自动检测模式，获取各 Figure 标题所在页码和 y 坐标
2. 查看全页截图（`temp_pageN_full_300dpi.png`），确定每个 Figure 的精确裁剪坐标
3. 用手动裁剪模式逐一提取干净的 Figure 图片
4. 裁剪时注意：排除周围正文文字，仅保留图形本体和标题（caption）

**文件命名规范（resource/ 目录下）**：

| 文件类型 | 命名格式 | 是否保留 | 说明 |
|---------|---------|---------|------|
| 最终 Figure 图片 | `figure{N}_{描述}.png` | **保留** | 手动裁剪或确认可用的论文原图 |
| 自绘辅助图表 | `{描述}.svg` | **保留** | 方法对比图、流程图、梯度流向图等 |
| 提取摘要 | `extraction_summary.json` | **保留** | 记录提取过程，便于追溯 |
| 嵌入位图 | `page{N}_img{序号}_{宽}x{高}.{ext}` | **保留** | 从 PDF 直接提取的嵌入图片，可能包含可用原图 |
| 全页截图（临时） | `temp_page{N}_full_{dpi}dpi.png` | **清理** | 仅用于手动裁剪参考，确认裁剪完成后即可删除 |
| 论文全文缓存（临时） | `temp_paper_{arxiv_id}.md` | **清理** | MCP 下载的 HTML 转换全文，质量不高，有 PDF 作为正式原文，禁止留在根目录 |

**清理规则**：`temp_*` 前缀的文件是明确的临时文件，在手动裁剪完成、最终 Figure 图片确认无误后，可直接代码删除，无需二次确认。

```bash
# 自动清理临时全页截图（保留最终 figure、svg、json 和嵌入图片）
rm papers/<论文简称>/resource/temp_*
```

### Step 6: 生成论文解读

参照 [template.md](template.md) 的模板格式，用**中文**撰写论文解读。

**写作原则**：

1. **面向小白**：假设读者没有相关背景知识，用生活化的类比和例子解释专业概念
2. **讲清故事线**：按"问题 → 动机 → 方法 → 效果"的逻辑组织
3. **善用类比**：对核心方法用通俗比喻辅助说明（如"这就像..."）
4. **不堆术语**：术语首次出现时给出简单解释，后续用中文替代或加括号注明
5. **突出亮点**：明确指出这篇论文最大的创新点和贡献
6. **忠实于原文（最高原则）**：解读内容必须严格基于论文实际内容，不得臆造、夸大或遗漏论文的关键方法。算法名称、组件职责、流程步骤必须与论文一致。如果论文做了A和B两件事，解读中不能只讲A却声称"统一处理A和B"。前后描述必须一致，不能前面说"算法X同时做A和B"，后面示例只演示了A。
7. **流程一致，数据可简**：【用一个具体例子走通全流程】部分的数值可以大幅简化（如用小矩阵代替大矩阵），但算法流程、步骤顺序、操作粒度（逐个vs批量、先剪枝后量化vs交替执行等）必须与论文完全一致。不得为了"好讲"而改变论文的实际流程。

**图表要求（重要）**：

解读中必须包含图表来辅助说明，不能全是文字叙述。图表分为两类：

**A. 论文原图（优先使用）**：

论文中已有的 Figure 必须使用 Step 4.5 提取的原图，用 Markdown 图片语法引用：
```markdown
![Figure 1: 描述](./resource/figure1_xxx.png)
```
不要用 ASCII 重画论文中已有的图。

**B. 自绘辅助图（补充使用 / 原图提取失败时的降级方案）**：

以下场景需自行绘制 **SVG 图表**，保存为独立 `.svg` 文件到 `resource/` 目录，然后在 Markdown 中用图片语法引用：
1. **原图提取降级**：论文中有对应 Figure，但提取失败（矢量图裁剪不干净、嵌入图分辨率过低、PDF 加密等），用 SVG 图表近似还原
2. **方法对比图**：论文中没有的横向对比图，将本文方法与现有方案并排展示
3. **流程推导图**：配合数值推导示例的计算流向图
4. **梯度流向图**：标注哪些参数有梯度、哪些被冻结
5. **部署对比图**：训练时 vs 推理时的结构变化

**SVG 绘图规范**：

- 保存为独立文件：`resource/<描述性名称>.svg`，如 `resource/gradient_flow.svg`
- 在 Markdown 中引用：`![梯度流向图](./resource/gradient_flow.svg)`
- 使用 `xmlns="http://www.w3.org/2000/svg"` 声明，设置合适的 `viewBox`
- 字体统一使用 `font-family="sans-serif"`，中文文本至少 12px，标题 14-16px
- 图表简单明了，适当使用颜色区分元素（如：冻结用灰色、可训练用绿色、缺点用红色）
- 矩形使用圆角（`rx`/`ry`），箭头清晰，整体风格整洁

**数学公式要求**：

使用 LaTeX 语法书写数学公式，以便在 Obsidian 等工具中正确渲染：
- 行内公式：`$...$`，如 `$h = Wx + BAx$`
- 独立公式：`$$...$$`，如 `$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$$`
- 矩阵使用 `\begin{bmatrix}...\end{bmatrix}`
- 不要使用纯文本近似写法（如 `W ∈ R^{d×d}`），统一用 LaTeX 语法

**流程推导要求（重要）**：

对于涉及模型训练或算法流程的论文，必须包含一个**具体数值的简化示例**，手把手走通核心流程，帮助读者建立直觉。

**通用要求**：
1. **场景设定**：给出一个极简的数值设定（如小矩阵、少样本），标明所有输入参数
2. **流程步骤与论文一致**：算法的执行顺序、操作粒度（逐个处理vs批量处理）、阶段划分（如先剪枝后量化vs交替执行）必须与论文完全一致，不得为了简化而改变流程
3. **数值可以简化**：矩阵维度、样本数量可以大幅缩小，但计算逻辑和步骤不能省略或篡改
4. **对比验证**：用具体数字展示算法执行前后的变化，验证算法确实在起作用

**训练型论文**（涉及梯度、反向传播）额外需要：
5. **损失函数**：写出数学形式，用直白中文翻译含义
6. **反向传播**：画出梯度流向图，用数值展示一步梯度计算
7. **参数更新**：用具体学习率算出更新后的参数，再做一次前向传播验证损失下降

**非训练型论文**（如后训练压缩、推理优化、架构设计等）替代要求：
5. **核心计算步骤**：用具体数值展示算法的关键计算过程（如Hessian计算、权重选择、更新公式等）
6. **多步迭代**：如果算法是迭代式的，展示至少2-3步的完整迭代过程
7. **结果验证**：用对比表格展示算法执行前后的效果差异

**术语解释要求**：

首次出现的专业缩写（如 MLP、FFN、LSTM、GRU 等）必须给出全称和通俗解释。例如 MLP 首次出现时应标注"MLP（Multi-Layer Perceptron，多层感知机）——最基础的神经网络结构，多层线性变换加激活函数"。后续可直接使用缩写。


### Step 7: 一致性自检（必做）

在保存解读文件之前，**必须**执行以下自检，确保解读忠实于论文且前后一致：

**1. 算法职责检查**
- 每个算法/组件的名称和职责是否与论文原文一致？
- 是否存在"把A的功劳归给B"的情况？（如：把框架的功劳归给单个算法）
- 论文中明确区分的多个组件，解读中是否保持了区分？

**2. 流程一致性检查**
- 【用一个具体例子走通全流程】中的步骤顺序是否与论文一致？
- 操作粒度（逐个vs批量、先A后Bvs交替）是否与论文一致？
- 如果论文包含多种算法/模式，示例是否覆盖了主要场景，或明确说明了"本节仅演示X"？

**3. 前后描述一致性检查**
- 前文声称"算法X同时做A和B"，后文示例是否同时展示了A和B？
- 核心概念在前后的定义和解释是否一致？
- 数字、指标、实验结果是否与论文表格/图表一致？

**4. 内容真实性检查**
- 是否有任何臆造的算法步骤、公式或实验结果？
- 是否有夸大或缩小论文贡献的表述？
- 论文的局限性是否被如实呈现？

**如果自检发现不一致，必须修正后再保存。**


### Step 8: 生成初始标签并更新配置

论文解读生成后，**必须根据论文内容给出初始标签**。标签存储在 `blog/blog-config.json` 中，是博客标签管理和筛选的数据来源。

**标签生成规则**：

1. **分析内容**：通读刚生成的解读全文，识别论文的核心主题、技术方法和应用领域
2. **生成 2-4 个标签**：每个标签应精准描述论文的一个维度
   - **方法维度**：论文提出的核心技术（如 LoRA、Transformer、Prompt Tuning）
   - **领域维度**：所属研究方向（如 图像生成、NLP、综述）
   - **范式维度**：技术范式特征（如 参数高效微调、少样本学习、注意力机制）
3. **复用已有标签**：优先使用 `blog/blog-config.json` 中 `tagColors` 里已有的标签，保持标签体系一致性
4. **新建标签**：只有当现有标签无法准确描述时才新建。新建标签时从下方调色盘取色，避免与已有标签颜色过于接近

**标签颜色调色盘**（按顺序取用，避免与已有颜色重复）：

```
#3b82f6(蓝)  #10b981(绿)  #f59e0b(橙)  #ec4899(粉)  #8b5cf6(紫)
#06b6d4(青)  #f43f5e(红)  #f97316(橘)  #a855f7(紫红) #0ea5e9(天蓝)
#6366f1(靛蓝) #64748b(灰)  #14b8a6(teal) #84cc16(lime) #d946ef(洋红)
```

**更新 `blog/blog-config.json`**：

读取 `blog/blog-config.json`，按以下规则更新：

```json
{
  "paperTags": {
    "新论文slug": ["标签1", "标签2"],
    "...": "..."
  },
  "tagColors": {
    "标签1": "#颜色",
    "...": "..."
  }
}
```

- `paperTags`：为新论文 slug 添加生成的标签数组
- `tagColors`：若使用了新标签，为其分配颜色；已有标签的颜色保持不变

**示例**：若精读的是一篇关于 RLHF 的论文：
- 分析后确定标签：`["RLHF", "LLM", "对齐"]`
- 其中 `LLM` 已存在于 `tagColors`，复用 `#6366f1`
- `RLHF` 和 `对齐` 是新建标签，分别取 `#14b8a6` 和 `#84cc16`

> **权限说明**：此步骤自动执行，无需询问用户。标签后续可通过博客管理后台（`/admin`）手动调整。

### Step 9: 保存解读文件

将生成的 Markdown 保存到论文专属文件夹内，文件名格式：

```
papers/<论文简称>/<论文简称>_解读.md
```

例如：`papers/Attention_Is_All_You_Need/Attention_Is_All_You_Need_解读.md`

### Step 7: 自动提交并推送

解读文件保存后，自动将本次新增的论文文件夹提交到 git 仓库并推送到远程。

```bash
cd <项目根目录>
git add "papers/<论文简称>/"
git commit -m "docs(<论文简称>): 添加论文解读

- 下载论文 PDF
- 提取论文原图至 resource/
- 生成中文解读（含 LaTeX 公式、数值推导、原图引用）
- 生成初始标签并更新 blog-config.json"
git push
```

**注意事项**：
- 提交范围仅限当前论文的 `papers/<论文简称>/` 文件夹，不要把其他未完成的论文一起提交
- 如果远程仓库尚未配置（`git remote` 为空），跳过 push 并提示用户先配置远程仓库
- 如果 push 失败（网络问题等），提示用户手动 push，不要重试

## 输出规范

- 语言：中文
- 格式：Markdown（LaTeX 公式使用 `$...$` / `$$...$$`）
- 长度：2000-5000 字（根据论文复杂度调整）
- 推荐阅读工具：Obsidian（免费，原生支持 LaTeX 渲染和图片引用）

### 目录结构规范

每篇论文在 `papers/` 目录下拥有独立的文件夹，所有相关文件集中管理：

```
papers/
├── Attention_Is_All_You_Need/               # 论文专属文件夹
│   ├── Attention_Is_All_You_Need.pdf        # 论文原文 PDF
│   ├── Attention_Is_All_You_Need_解读.md    # 论文解读
│   └── resource/                            # 解读引用的资源
│       ├── figure1_transformer_arch.png     # 最终保留：从 PDF 提取的原图
│       ├── figure2_attention_detail.png
│       ├── gradient_flow.svg                # 最终保留：自绘辅助图表
│       ├── extraction_summary.json          # 最终保留：提取摘要记录
│       └── page5_img1_640x480.png           # 最终保留：从 PDF 提取的嵌入位图
│
├── LoRA_Low-Rank_Adaptation/                # 另一篇论文
│   ├── LoRA_Low-Rank_Adaptation.pdf
│   ├── LoRA_Low-Rank_Adaptation_解读.md
│   └── resource/
│       ├── figure1_reparametrization.png    # 手动裁剪的最终 Figure
│       ├── figure2_accuracy_vs_params.png
│       └── method_comparison.svg            # 自绘对比图
│
└── ...                                      # 更多论文
```

**关键约定**：
- 论文文件夹名 = 论文简称（与 PDF 文件名一致，不含扩展名）
- 解读 Markdown 中引用资源使用**相对路径** `./resource/xxx.png`
- `resource/` 目录存放解读所需的最终资源（提取原图、自制 SVG、`extraction_summary.json`）
- **所有论文相关文件必须集中在 `papers/<论文简称>/` 下**，禁止任何文件散落在项目根目录
- **临时文件统一使用 `temp_` 前缀**（如 `temp_page5_full_300dpi.png`、`temp_paper_2309.12307.md`），仅用于中间处理，确认完成后自动代码删除
- **保留文件命名规则**：最终 Figure 用 `figure{N}_{描述}.png`，自绘图用 `{描述}.svg`，不得使用 `temp_` 前缀
- Git 推送后，GitHub Actions 会自动将论文解读同步发布到博客

### Step 10: 清理临时文件（自动执行）

在手动裁剪完成、最终 Figure 图片已保存后，**自动删除**所有 `temp_` 前缀的临时文件。此步骤无需询问用户，因为 `temp_` 前缀已明确标识这些文件为一次性参考用途。

**需清理的临时文件清单**：
- `temp_page{N}_full_{dpi}dpi.png` —— 全页截图
- `temp_paper_{arxiv_id}.md` —— MCP 下载的 HTML 转换全文缓存

```bash
# 自动清理该论文目录下 resource/ 中的所有临时文件
rm papers/<论文简称>/resource/temp_*
```

**清理后 resource/ 中应仅保留**：最终 Figure 图片（`figure{N}_*.png`）、自绘 SVG、嵌入位图、`extraction_summary.json`。若某论文没有可提取的嵌入图片且所有 figure 均为自绘 SVG，则仅保留 SVG 和 `extraction_summary.json`。


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
