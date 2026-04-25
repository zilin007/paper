/**
 * VitePress 数据加载器 — 扫描 posts/ 目录的所有论文解读并提取元信息。
 * 从 blog-config.json 读取标签颜色。
 * 由 PostList.vue 和 index.md 消费。
 */
import { createContentLoader } from 'vitepress'
import { readFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const CONFIG_PATH = join(__dirname, 'blog-config.json')

function loadConfig() {
  try {
    return JSON.parse(readFileSync(CONFIG_PATH, 'utf-8'))
  } catch {
    return { paperTags: {}, tagColors: {} }
  }
}

export default createContentLoader('posts/**/*.md', {
  transform(rawData) {
    const config = loadConfig()
    const tagColors = config.tagColors || {}

    // 提取所有唯一标签
    const allTagsSet = new Set()

    const posts = rawData.map((page) => {
      const tags = page.frontmatter.tags || []
      tags.forEach((t) => allTagsSet.add(t))

      return {
        title: page.frontmatter.title || 'Untitled',
        originalTitle: page.frontmatter.originalTitle || '',
        url: page.url,
        date: page.frontmatter.date || '',
        description: page.frontmatter.description || '',
        authors: page.frontmatter.authors || '',
        arxiv: page.frontmatter.arxiv || '',
        tags,
      }
    })

    // 按日期降序排序（最新在前）
    posts.sort((a, b) => +new Date(b.date) - +new Date(a.date))

    return {
      posts,
      tagColors,
      allTags: [...allTagsSet],
    }
  },
})
