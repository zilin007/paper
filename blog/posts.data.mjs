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

    // 从 arXiv ID 提取排序键：YYMM.NNNNN → 20YYMMNNNNN
    // arXiv ID 本身即为时间戳（2007年4月起），按 ID 排序即按提交时间排序
    function getSortKey(post) {
      const arxivId = post.arxiv || ''
      const match = arxivId.match(/(\d{2})(\d{2})\.(\d+)/)
      if (match) {
        return `20${match[1]}${match[2]}${match[3].padStart(5, '0')}`
      }
      // 非 arXiv 论文回退到 date 字段
      return post.date ? post.date.replace(/-/g, '') : '00000000'
    }

    posts.sort((a, b) => getSortKey(b).localeCompare(getSortKey(a)))

    return {
      posts,
      tagColors,
      allTags: [...allTagsSet],
    }
  },
})
