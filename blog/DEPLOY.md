# 博客部署指南

本文档说明如何将论文精读博客部署到 GitHub Pages，以及日常使用中如何添加新论文、自定义主题。

## 架构概览

```
papers/                     <- 论文源文件（你写的内容）
  └── <论文简称>/
      ├── <论文简称>.pdf
      ├── <论文简称>_解读.md
      └── resource/

scripts/sync-papers.mjs     <- 同步脚本（构建时自动运行）
                              将 papers/ 转换为 VitePress 可用的格式

blog/                       <- VitePress 站点
  ├── .vitepress/config.mts <- 站点配置
  ├── .vitepress/theme/     <- 自定义主题（含 Giscus 评论）
  └── posts/                <- 【自动生成】论文解读页面 + 图片资源
      └── <论文简称>/
          ├── index.md      <- 带 frontmatter 的解读文件
          └── resource/     <- 论文图片（从 papers/ 复制）

.github/workflows/deploy.yml <- GitHub Actions 自动构建+部署
```

核心流程：`git push` → GitHub Actions → 运行同步脚本 → VitePress 构建 → 部署到 GitHub Pages。

## 首次部署

### 1. 启用 GitHub Pages

1. 打开仓库 Settings → Pages
2. Source 选择 **GitHub Actions**（不是 "Deploy from a branch"）
3. 保存

### 2. 推送代码

```bash
git add .
git commit -m "feat: 添加 VitePress 博客和自动部署"
git push origin main
```

推送后，GitHub Actions 会自动构建并部署。首次部署可能需要 2-3 分钟。

部署完成后，博客地址为：`https://<你的用户名>.github.io/paper/`

### 3. 配置 Giscus 评论（可选）

1. 在仓库 Settings → Features 中启用 **Discussions**
2. 安装 Giscus App：https://github.com/apps/giscus
3. 打开 https://giscus.app/zh-CN ，填入你的仓库信息，获取 `repo-id` 和 `category-id`
4. 编辑 `blog/.vitepress/theme/GiscusComment.vue`，填入对应值：

```js
const GISCUS_CONFIG = {
  repo: 'zilin007/paper',
  repoId: '你的 repo-id',
  category: 'Announcements',
  categoryId: '你的 category-id',
}
```

5. 提交并推送，评论功能即生效。

## 日常使用：添加新论文

### 方式一：AI 辅助（推荐）

在你的 AI 平台中输入：

```
请帮我精读这篇论文：<论文名称>
```

Skill 会自动完成：搜索 → 下载 → 阅读 → 提图 → 写解读 → Git 提交推送 → 博客自动更新。

### 方式二：手动添加

1. 在 `papers/` 下创建论文文件夹：

```bash
mkdir -p papers/My_Paper_Name
```

2. 放入 PDF 和解读文件：

```
papers/My_Paper_Name/
├── My_Paper_Name.pdf
├── My_Paper_Name_解读.md
└── resource/
    └── figure1_xxx.png
```

3. 解读文件的开头格式（同步脚本依赖这些信息生成博客元数据）：

```markdown
# 论文标题中文翻译

> **原文**: English Paper Title
> **作者**: Author1, Author2, ...
> **发表时间**: 2024 年
> **arXiv**: https://arxiv.org/abs/xxxx.xxxxx

## 一句话总结

这篇论文做了什么事情的一句话概括。
```

4. 图片使用相对路径引用：`![描述](./resource/figure1_xxx.png)`

5. 提交推送：

```bash
git add papers/My_Paper_Name/
git commit -m "docs(My_Paper_Name): 添加论文解读"
git push
```

博客会在 1-2 分钟内自动更新。

## 本地预览

```bash
# 安装博客依赖（仅首次）
cd blog && npm install && cd ..

# 运行同步 + 启动开发服务器
cd blog && npm run dev
```

浏览器打开 http://localhost:5173/paper/ 即可预览。

## 自定义主题

### 修改站点信息

编辑 `blog/.vitepress/config.mts`：

```ts
export default defineConfig({
  title: '你的博客标题',
  description: '你的博客描述',
  base: '/paper/',  // GitHub Pages 路径，自定义域名改为 '/'
  // ...
})
```

### 修改样式

编辑 `blog/.vitepress/theme/style.css`，覆盖 VitePress 的 CSS 变量来调整颜色、字体等。

### 添加导航链接

在 `config.mts` 的 `themeConfig.nav` 中添加：

```ts
nav: [
  { text: '首页', link: '/' },
  { text: '论文列表', link: '/papers' },
  { text: '关于', link: '/about' },  // 新增
],
```

然后创建 `blog/about.md` 写入个人简介。

### 自定义域名

1. 在 `blog/public/` 下创建 `CNAME` 文件，写入域名（如 `paper.example.com`）
2. 将 `config.mts` 的 `base` 改为 `'/'`
3. 在你的 DNS 服务商添加 CNAME 记录指向 `<用户名>.github.io`

## 同步脚本说明

`scripts/sync-papers.mjs` 的工作流程：

1. 扫描 `papers/` 目录下所有子文件夹
2. 在每个子文件夹中查找 `*_解读.md` 文件
3. 从 Markdown 内容中提取元信息（标题、作者、日期、摘要）
4. 生成 VitePress frontmatter 并注入到文件头部
5. 将处理后的文件写入 `blog/posts/<名称>/index.md`
6. 将 `resource/` 目录复制到 `blog/posts/<名称>/resource/`（与 index.md 同目录，相对路径无需修改）

由于图片资源与 `index.md` 同目录，使用相对路径 `./resource/xxx.png`，VitePress 会自动处理 base path，无需额外配置。

## 故障排查

**Q: 推送后博客没有更新？**

检查 GitHub Actions 是否成功运行：仓库 → Actions → 查看最近的 workflow run。

**Q: 图片在博客上显示不出来？**

确认解读文件中的图片使用 `./resource/xxx.png` 相对路径格式，同步脚本会自动处理。

**Q: 数学公式渲染异常？**

VitePress 使用 MathJax3 渲染公式。确保公式使用 `$...$`（行内）或 `$$...$$`（独立）语法，且 `$` 符号前后没有空格。

**Q: 本地预览正常但部署后 404？**

检查 `config.mts` 的 `base` 配置是否与实际部署路径一致。GitHub Pages 默认路径为 `/<仓库名>/`。
