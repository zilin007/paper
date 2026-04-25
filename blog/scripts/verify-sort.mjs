#!/usr/bin/env node
/**
 * 验证基于 arXiv ID 的排序结果
 */

import { readdir, readFile } from 'fs/promises'
import { join } from 'path'

const POSTS_DIR = 'F:\\professional\\AI笔记\\paper\\blog\\posts'

function getSortKey(arxivId) {
  if (!arxivId) return '0000000000'
  const match = arxivId.match(/(\d{2})(\d{2})\.(\d+)/)
  if (match) {
    return `20${match[1]}${match[2]}${match[3].padStart(5, '0')}`
  }
  return '0000000000'
}

async function main() {
  const dirs = await readdir(POSTS_DIR, { withFileTypes: true })
  const papers = []

  for (const dir of dirs) {
    if (!dir.isDirectory()) continue
    const content = await readFile(join(POSTS_DIR, dir.name, 'index.md'), 'utf-8')
    const arxivMatch = content.match(/arxiv:\s*(.+)$/m)
    const dateMatch = content.match(/date:\s*(.+)$/m)
    const arxiv = arxivMatch?.[1]?.trim().replace(/"/g, '') || ''
    const date = dateMatch?.[1]?.trim() || ''
    papers.push({ slug: dir.name, arxiv, date, sortKey: getSortKey(arxiv) })
  }

  // 按排序键降序
  papers.sort((a, b) => b.sortKey.localeCompare(a.sortKey))

  console.log('排序验证结果（最新在前）：\n')
  for (const p of papers) {
    console.log(`${p.sortKey} | ${p.date} | ${p.arxiv.padEnd(20)} | ${p.slug}`)
  }
}

main().catch(console.error)
