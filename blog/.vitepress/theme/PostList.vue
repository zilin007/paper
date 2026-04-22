<script setup>
import { computed } from 'vue'
import { data as posts } from '../../posts.data.mjs'
import { withBase } from 'vitepress'

const sortedPosts = computed(() =>
  [...posts].sort((a, b) => +new Date(b.date) - +new Date(a.date)),
)

function formatDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  return d.toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' })
}
</script>

<template>
  <div class="paper-list">
    <div v-for="post in sortedPosts" :key="post.url" class="paper-card">
      <h3>
        <a :href="withBase(post.url)">{{ post.title }}</a>
      </h3>
      <div class="paper-meta">
        <span v-if="post.date">{{ formatDate(post.date) }}</span>
        <span v-if="post.authors">{{ post.authors }}</span>
      </div>
      <p v-if="post.description" class="paper-desc">{{ post.description }}</p>
      <div v-if="post.tags?.length" class="paper-tags">
        <span v-for="tag in post.tags" :key="tag" class="paper-tag">{{ tag }}</span>
      </div>
    </div>
    <p v-if="!sortedPosts.length" style="color: var(--vp-c-text-2);">
      暂无论文解读，快去精读一篇吧！
    </p>
  </div>
</template>
