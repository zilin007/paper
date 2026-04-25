#!/usr/bin/env node
/**
 * sync-papers.mjs — 将 papers/ 目录下的论文解读同步到 blog/ 供 VitePress 构建。
 *
 * 主要工作：
 *   1. 扫描 papers/ 下所有 *_解读.md
 *   2. 提取元信息（标题、作者、日期、摘要）生成 VitePress frontmatter
 *   3. 读取 blog-config.json 注入标签、排序、置顶信息
 *   4. 将处理后的 Markdown 写入 blog/posts/<name>/index.md
 *   5. 将 resource/ 静态资源复制到 blog/posts/<name>/resource/
 *   6. 由于 index.md 和 resource/ 同目录，原始 ./resource/ 相对路径无需修改
 *
 * 用法：在 blog/ 目录下运行 node scripts/sync-papers.mjs
 */

import { readdir, readFile, writeFile, mkdir, cp, stat, rm } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const BLOG_DIR = join(__dirname, '..')         // blog/
const ROOT = join(BLOG_DIR, '..')              // 项目根目录
const PAPERS_DIR = join(ROOT, 'papers')
const POSTS_DIR = join(BLOG_DIR, 'posts')
const CONFIG_PATH = join(BLOG_DIR, 'blog-config.json')

// ─────────────────── 配置加载 ───────────────────

async function loadConfig() {
  try {
    const raw = await readFile(CONFIG_PATH, 'utf-8')
    return JSON.parse(raw)
  } catch {
    return { paperOrder: [], pinnedPapers: [], paperTags: {}, tagColors: {} }
  }
}

// ─────────────────── 元信息提取 ───────────────────

function extractMeta(content) {
  const titleMatch = content.match(/^# (.+)$/m)
  const originalMatch = content.match(/\*\*原文\*\*:\s*(.+)$/m)
  const authorsMatch = content.match(/\*\*作者\*\*:\s*(.+)$/m)
  const dateMatch = content.match(/\*\*发表时间\*\*:\s*(.+)$/m)
  const arxivMatch = content.match(/\*\*arXiv\*\*:\s*\[?([^\]\s)]+)/)
  const summaryMatch = content.match(/## 一句话总结\s*\n+(.+)/m)

  return {
    title: titleMatch?.[1]?.trim() || 'Untitled',
    originalTitle: originalMatch?.[1]?.trim() || '',
    authors: authorsMatch?.[1]?.trim() || '',
    date: extractDate(dateMatch?.[1]) || new Date().toISOString().slice(0, 10),
    arxiv: arxivMatch?.[1]?.trim() || '',
    description: summaryMatch?.[1]?.trim().replace(/\*\*/g, '') || '',
  }
}

function extractDate(dateStr) {
  if (!dateStr) return null
  const fullMatch = dateStr.match(/(\d{4})-(\d{2})-(\d{2})/)
  if (fullMatch) return fullMatch[0]
  const yearMatch = dateStr.match(/(\d{4})/)
  return yearMatch ? `${yearMatch[1]}-01-01` : null
}

// YAML 值安全转义
function yamlStr(str) {
  if (!str) return '""'
  if (/[:#\[\]{},"'|>&*!%@`]/.test(str) || str.includes('\n')) {
    return `"${str.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
  }
  return `"${str}"`
}

// YAML 数组序列化
function yamlTags(tags) {
  if (!tags || !tags.length) return '[]'
  return `[${tags.map((t) => `"${t}"`).join(', ')}]`
}

// ─────────────────── 主流程 ───────────────────

async function main() {
  // 加载博客配置
  const config = await loadConfig()

  // 清理旧的生成文件
  await rm(POSTS_DIR, { recursive: true, force: true })
  await mkdir(POSTS_DIR, { recursive: true })

  // 检查 papers 目录是否存在
  try {
    await stat(PAPERS_DIR)
  } catch {
    console.log('papers/ 目录不存在，跳过同步。')
    return
  }

  const entries = await readdir(PAPERS_DIR, { withFileTypes: true })
  const paperDirs = entries.filter((e) => e.isDirectory() && !e.name.startsWith('.'))

  let count = 0

  for (const dir of paperDirs) {
    const paperPath = join(PAPERS_DIR, dir.name)
    const files = await readdir(paperPath)

    // 查找 *_解读.md 文件
    const readingFile = files.find((f) => f.endsWith('_解读.md'))
    if (!readingFile) {
      console.log(`  跳过 ${dir.name}（无解读文件）`)
      continue
    }

    const mdPath = join(paperPath, readingFile)
    const content = await readFile(mdPath, 'utf-8')

    // 提取元信息
    const meta = extractMeta(content)

    // 从配置获取排序、标签、置顶信息
    const slug = dir.name
    const orderIndex = config.paperOrder.indexOf(slug)
    const order = orderIndex >= 0 ? orderIndex : 999
    const pinned = (config.pinnedPapers || []).includes(slug)
    const tags = (config.paperTags || {})[slug] || []

    // 生成 frontmatter
    const frontmatter = [
      '---',
      `title: ${yamlStr(meta.title)}`,
      `originalTitle: ${yamlStr(meta.originalTitle)}`,
      `authors: ${yamlStr(meta.authors)}`,
      `date: ${meta.date}`,
      `description: ${yamlStr(meta.description)}`,
      `arxiv: ${yamlStr(meta.arxiv)}`,
      `tags: ${yamlTags(tags)}`,
      `order: ${order}`,
      `pinned: ${pinned}`,
      `comment: true`,
      '---',
      '',
    ].join('\n')

    // 创建论文专属文件夹: blog/posts/<paper_name>/
    const postDir = join(POSTS_DIR, dir.name)
    await mkdir(postDir, { recursive: true })

    // 写入 index.md（./resource/ 相对路径天然正确，无需替换）
    const outPath = join(postDir, 'index.md')
    await writeFile(outPath, frontmatter + content, 'utf-8')

    // 复制 resource/ 静态资源到同目录
    const resourceDir = join(paperPath, 'resource')
    try {
      await stat(resourceDir)
      const destResource = join(postDir, 'resource')
      await cp(resourceDir, destResource, { recursive: true })
    } catch {
      // resource 目录不存在，跳过
    }

    count++
    console.log(`  ✓ ${dir.name} → posts/${dir.name}/index.md`)
  }

  console.log(`\n同步完成：共 ${count} 篇论文解读 → blog/posts/`)
}

main().catch((err) => {
  console.error('同步失败:', err)
  process.exit(1)
})
