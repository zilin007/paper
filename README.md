# Paper Reading - 论文精读知识库

从 arXiv 搜索论文、下载 PDF、提取原图，并生成通俗易懂的中文 Markdown 解读（含 LaTeX 公式、数值推导、原图引用）。精读内容自动发布为博客。

项目由两层组成，按需使用：

| 层级 | 说明 | 依赖 |
|------|------|------|
| **AI 平台 + Skill + MCP** | 一句话触发完整流程，AI 自动阅读论文并生成解读 | 任意支持 MCP 的 AI 平台 |
| **VitePress 博客** | 论文解读自动发布为静态博客，支持公式渲染、全文搜索、评论 | GitHub Pages（免费） |
| **纯本地模式** | 自配模型 + harness 编排完整流程 | 待实现 |

## 目录结构

```
paper/
├── .mcp.json                            # MCP 服务配置（跨平台通用）
├── skills/
│   └── paper-reading/                   # 论文精读 Skill（完整自包含）
│       ├── SKILL.md                     # Skill 定义（跨平台通用）
│       ├── template.md                  # 解读 Markdown 模板
│       └── scripts/                     # 降级辅助脚本（MCP 不可用时）
│           ├── download_arxiv.py        # arXiv 搜索 & PDF 下载（纯标准库）
│           └── extract_pdf_figures.py   # PDF Figure 提取（需 PyMuPDF）
│
├── papers/                              # 论文解读目录（每篇独立文件夹）
│   └── <论文简称>/
│       ├── <论文简称>.pdf
│       ├── <论文简称>_解读.md
│       └── resource/
│
├── blog/                                # VitePress 博客站点
│   ├── .vitepress/config.mts            # 站点配置
│   ├── .vitepress/theme/                # 自定义主题（含 Giscus 评论）
│   ├── scripts/
│   │   └── sync-papers.mjs              # 论文 → 博客同步脚本
│   ├── index.md                         # 博客首页
│   ├── papers.md                        # 论文列表页
│   ├── DEPLOY.md                        # 博客部署文档
│   └── posts/                           # 【自动生成】论文解读页面 + 图片
│       └── <论文简称>/
│           ├── index.md                 # 带 frontmatter 的解读
│           └── resource/                # 从 papers/ 复制的图片
│
├── .github/workflows/deploy.yml         # GitHub Actions 自动部署
├── requirements.txt                     # Python 依赖
├── .gitignore
└── README.md
```

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone <repo-url> paper && cd paper

# Python 依赖（论文精读脚本需要）
uv venv .venv --python 3.13
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
uv pip install -r requirements.txt

# 博客依赖（本地预览需要）
cd blog && npm install && cd ..
```

### 2. 安装 arXiv MCP 服务（AI 平台模式需要）

```bash
uv tool install arxiv-mcp-server

# 旧论文的 PDF 回退支持（可选）
uv tool install 'arxiv-mcp-server[pdf]'
```

### 3. 使用

在你的 AI 平台中对话即可，例如：

```
请帮我精读这篇论文：LoRA: Low-Rank Adaptation of Large Language Models
```

Skill 会自动完成：搜索论文 → 下载 PDF → 提取原图 → 生成中文解读 → Git 提交。

推送到 GitHub 后，博客自动更新。

### 4. 本地预览博客

```bash
cd blog && npm run dev
```

浏览器打开 http://localhost:5173/paper/ 即可预览。

## 博客部署

详见 [blog/DEPLOY.md](blog/DEPLOY.md)。

首次部署只需：
1. GitHub 仓库 Settings → Pages → Source 选 **GitHub Actions**
2. 推送代码到 main 分支
3. 等待 1-2 分钟，博客即可访问

## 跨平台部署

### MCP 配置

项目根目录的 `.mcp.json` 遵循 MCP 标准格式，大多数平台可直接识别：

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "uvx",
      "args": ["arxiv-mcp-server", "--storage-path", "./.arxiv-cache"]
    }
  }
}
```

各平台配置位置：

| 平台 | MCP 配置方式 | 说明 |
|------|-------------|------|
| **Claude Code** | 项目根目录 `.mcp.json` 自动识别 | 无需额外操作 |
| **Cursor** | 复制到 `.cursor/mcp.json` | `cp .mcp.json .cursor/mcp.json` |
| **VS Code (Copilot)** | 复制到 `.vscode/mcp.json` | `cp .mcp.json .vscode/mcp.json` |
| **Codex** | 项目根目录 `.mcp.json` 自动识别 | 无需额外操作 |
| **QoderWork** | 在设置中手动添加 arxiv MCP 服务 | 服务名设为 `arxiv`，参数 `ARXIV_STORAGE_PATH` 指向本目录 |
| **其他 MCP 客户端** | 将 `mcpServers` 内容复制到对应配置文件 | 参考各平台文档 |

### Skill 配置

`skills/paper-reading/` 目录包含完整的 Skill 定义，可安装到任意支持 Skill/Instruction 的平台。

| 平台 | 安装方式 |
|------|---------|
| **QoderWork** | 复制到 Skill 目录：`xcopy /E /I "skills\paper-reading" "%USERPROFILE%\.qoderwork\skills\paper-reading"` (Win) 或 `cp -r skills/paper-reading/ ~/.qoderwork/skills/paper-reading/` (Mac/Linux) |
| **Claude Code** | 将 `SKILL.md` 内容追加到项目的 `CLAUDE.md`，或放入 `.claude/commands/` 作为自定义命令 |
| **Cursor** | 将 `SKILL.md` 内容追加到 `.cursor/rules/paper-reading.md` |
| **Codex** | 将 `SKILL.md` 内容追加到项目的 `AGENTS.md` |
| **QwenCLI / 通义灵码** | 将 `SKILL.md` 内容作为系统提示或项目规则注入 |

> **Skill 的核心思想**：`SKILL.md` 本质是一份结构化的 Agent 指令，描述了"搜索→下载→阅读→提图→写解读→提交"的完整工作流。任何支持自定义指令/规则的 AI 平台都可以使用，只需将内容放到平台约定的位置即可。

## 依赖说明

| 依赖 | 用途 | 备注 |
|------|------|------|
| `arxiv-mcp-server` | MCP 服务，提供论文搜索/下载/阅读工具 | AI 平台模式需要，`uv tool install` 安装 |
| `PyMuPDF` | 从 PDF 提取嵌入图片、渲染页面、裁剪 Figure | CLI 模式的图片提取需要，通过 `requirements.txt` 安装 |
| `uv` | 包管理器，创建虚拟环境、安装 `arxiv-mcp-server` | 前置工具，`pip install uv` 或独立安装 |
| `vitepress` | 静态博客生成器 | 博客构建需要，通过 `blog/package.json` 安装 |
| Node.js 18+ | 运行 VitePress 和同步脚本 | 博客本地预览/构建需要 |

## 解读质量规范

每篇解读遵循 `skills/paper-reading/template.md` 模板，包含以下要素：

- **论文原图**：从 PDF 提取的 Figure，非重绘
- **自绘辅助图**：方法对比、梯度流向等概念性 ASCII 图表
- **LaTeX 公式**：`$...$` 行内、`$$...$$` 独立公式
- **数值推导示例**：用具体数字走通核心流程
- **通俗类比**：面向初学者，用生活化比喻解释专业概念

推荐使用 [Obsidian](https://obsidian.md)（免费）打开本仓库的 `papers/` 目录作为 Vault，原生支持 LaTeX 渲染和图片预览。

## Roadmap

- [x] arXiv MCP 服务集成
- [x] Skill 工作流定义
- [x] 跨平台 MCP / Skill 配置
- [x] VitePress 博客 + GitHub Pages 自动部署
- [x] Giscus 评论 + 本地搜索
- [ ] 纯本地模式（不依赖任何 AI 平台，自配模型 + harness 编排）
- [ ] 第三方平台同步（知乎/公众号格式导出）

## 许可

MIT
