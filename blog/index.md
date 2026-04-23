---
layout: home
hero:
  name: Paper Reading
  text: 论文精读博客
  tagline: 用通俗的语言解读前沿 AI 论文，含 LaTeX 公式、数值推导与原图引用
  actions:
    - theme: brand
      text: 开始阅读
      link: /papers
    - theme: alt
      text: ⭐ Star on GitHub
      link: https://github.com/zilin007/paper
    - theme: alt
      text: 🍴 Fork
      link: https://github.com/zilin007/paper/fork
---

<script setup>
import PostList from './.vitepress/theme/PostList.vue'
</script>

<div style="text-align:center;margin:1.5rem 0 0.5rem;padding:1rem 1.5rem;border-radius:12px;background:var(--vp-c-bg-soft);">
  <p style="margin:0;font-size:15px;color:var(--vp-c-text-2);">
    觉得有帮助？给个 ⭐ Star 支持一下 · 想搭建自己的精读博客？点 🍴 Fork 一键复制
  </p>
</div>

## 最新论文解读

<PostList />
