import { defineConfig } from 'vitepress'
import { adminApiPlugin } from './plugins/adminApi.mjs'

export default defineConfig({
  title: 'Paper Reading',
  description: '论文精读博客 — 用通俗的语言解读前沿 AI 论文',
  lang: 'zh-CN',

  // 排除非页面的 Markdown 文件
  srcExclude: ['DEPLOY.md'],

  // GitHub Pages: 仓库名作为 base path
  // 如果部署到 https://<user>.github.io/paper/，base 设为 '/paper/'
  // 如果部署到自定义域名根目录，改为 '/'
  base: '/paper/',

  // 内置 MathJax 支持，渲染 $...$ 和 $$...$$ 公式
  markdown: {
    math: true,
  },

  // <head> 注入
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/paper/logo.svg' }],
    ['meta', { name: 'author', content: 'zilin007' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'Paper Reading' }],
    ['meta', { property: 'og:description', content: '论文精读博客 — 用通俗的语言解读前沿 AI 论文' }],
  ],

  // 注册 Vite 插件（admin API 仅在 dev 模式生效）
  vite: {
    plugins: [adminApiPlugin()],
  },

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: '论文列表', link: '/papers' },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/zilin007/paper' },
    ],

    // 内置本地搜索（基于 minisearch，支持中文）
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索论文', buttonAriaLabel: '搜索论文' },
          modal: {
            noResultsText: '未找到相关结果',
            resetButtonTitle: '清除查询',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    outline: {
      level: [2, 3],
      label: '本文目录',
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: `© 2025-present <a href="https://github.com/zilin007">zilin007</a>`,
    },
  },
})
