/**
 * VitePress 数据加载器 — 扫描 posts/ 目录的所有论文解读并提取元信息。
 * 由 PostList.vue 和 index.md 消费。
 */
import { createContentLoader } from 'vitepress'

export default createContentLoader('posts/**/*.md', {
  transform(rawData) {
    return rawData
      .sort((a, b) => +new Date(b.frontmatter.date) - +new Date(a.frontmatter.date))
      .map((page) => ({
        title: page.frontmatter.title || 'Untitled',
        originalTitle: page.frontmatter.originalTitle || '',
        url: page.url,
        date: page.frontmatter.date || '',
        description: page.frontmatter.description || '',
        authors: page.frontmatter.authors || '',
        arxiv: page.frontmatter.arxiv || '',
        tags: page.frontmatter.tags || [],
      }))
  },
})
