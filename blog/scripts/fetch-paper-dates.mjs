#!/usr/bin/env node
/**
 * 批量从 arXiv API 获取论文精确发表日期
 * 用法: node scripts/fetch-paper-dates.mjs
 */

import { readFile, writeFile, readdir } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const PAPERS_DIR = join(__dirname, '..', '..', 'papers')

// 从 Markdown 内容提取 arXiv ID
function extractArxivId(content) {
  const match = content.match(/arxiv\.org\/abs\/(\d+\.\d+)/)
  return match?.[1] || null
}

// 从 arXiv API 批量获取日期
async function fetchDates(arxivIds) {
  const idList = arxivIds.join(',')
  const url = `http://export.arxiv.org/api/query?id_list=${idList}&max_results=${arxivIds.length}`

  const res = await fetch(url)
  if (!res.ok) throw new Error(`arXiv API 错误: ${res.status}`)

  const xml = await res.text()
  const results = {}

  // 解析 XML 提取每个论文的 published 日期
  const entryRegex = /<entry>[\s\S]*?<id>http:\/\/arxiv\.org\/abs\/(\d+\.\d+)[\s\S]*?<published>(\d{4}-\d{2}-\d{2})/g
  let m
  while ((m = entryRegex.exec(xml)) !== null) {
    results[m[1]] = m[2]
  }

  return results
}

async function main() {
  // 扫描所有论文
  const dirs = await readdir(PAPERS_DIR, { withFileTypes: true })
  const paperDirs = dirs.filter((d) => d.isDirectory() && !d.name.startsWith('.'))

  const papers = []
  for (const dir of paperDirs) {
    const files = await readdir(join(PAPERS_DIR, dir.name))
    const mdFile = files.find((f) => f.endsWith('_解读.md'))
    if (!mdFile) continue

    const content = await readFile(join(PAPERS_DIR, dir.name, mdFile), 'utf-8')
    const arxivId = extractArxivId(content)
    if (arxivId) {
      papers.push({ slug: dir.name, arxivId })
    } else {
      console.log(`  ⚠ ${dir.name} 未找到 arXiv ID`)
    }
  }

  console.log(`找到 ${papers.length} 篇论文，开始查询 arXiv API...`)

  // arXiv API 一次最多查询 100 个，这里应该够
  const arxivIds = papers.map((p) => p.arxivId)
  const dates = await fetchDates(arxivIds)

  // 生成结果
  const results = []
  for (const p of papers) {
    const date = dates[p.arxivId]
    if (date) {
      results.push({ slug: p.slug, arxivId: p.arxivId, date })
      console.log(`  ✓ ${p.slug}: ${date}`)
    } else {
      console.log(`  ✗ ${p.slug}: 未获取到日期`)
    }
  }

  console.log(`\n共 ${results.length} 条精确日期。`)
  console.log('\n如需更新解读文件中的日期，请在各 *_解读.md 中写入：')
  console.log('> **发表时间**: YYYY-MM-DD')
  console.log('\n各论文日期如下（可直接复制）：\n')
  for (const r of results) {
    console.log(`${r.slug}: ${r.date}`)
  }
}

main().catch((err) => {
  console.error('获取失败:', err)
  process.exit(1)
})
