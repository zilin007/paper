<script setup>
import { computed, ref } from 'vue'
import { data } from '../../posts.data.mjs'
import { withBase } from 'vitepress'

const { posts, tagColors, allTags } = data

// 当前选中的标签筛选（空表示全部）
const activeTag = ref('')

const filteredPosts = computed(() => {
  if (!activeTag.value) return posts
  return posts.filter((p) => p.tags?.includes(activeTag.value))
})

function formatDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  return d.toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' })
}

function getTagStyle(tag) {
  const color = tagColors[tag] || '#64748b'
  return {
    backgroundColor: color + '18',
    color: color,
    borderColor: color + '40',
  }
}

function toggleTag(tag) {
  activeTag.value = activeTag.value === tag ? '' : tag
}
</script>

<template>
  <!-- 标签筛选栏 -->
  <div v-if="allTags.length" class="tag-filter">
    <button
      class="tag-filter-btn"
      :class="{ active: !activeTag }"
      @click="activeTag = ''"
    >
      全部
    </button>
    <button
      v-for="tag in allTags"
      :key="tag"
      class="tag-filter-btn"
      :class="{ active: activeTag === tag }"
      :style="activeTag === tag ? getTagStyle(tag) : {}"
      @click="toggleTag(tag)"
    >
      {{ tag }}
    </button>
  </div>

  <!-- 论文列表 -->
  <div class="paper-list">
    <div v-for="post in filteredPosts" :key="post.url" class="paper-card" :class="{ pinned: post.pinned }">
      <div v-if="post.pinned" class="pin-badge">置顶</div>
      <h3>
        <a :href="withBase(post.url)">{{ post.title }}</a>
      </h3>
      <div class="paper-meta">
        <span v-if="post.date">{{ formatDate(post.date) }}</span>
        <span v-if="post.authors">{{ post.authors }}</span>
      </div>
      <p v-if="post.description" class="paper-desc">{{ post.description }}</p>
      <div v-if="post.tags?.length" class="paper-tags">
        <span
          v-for="tag in post.tags"
          :key="tag"
          class="paper-tag"
          :style="getTagStyle(tag)"
          @click.prevent="toggleTag(tag)"
        >
          {{ tag }}
        </span>
      </div>
    </div>
    <p v-if="!filteredPosts.length" style="color: var(--vp-c-text-2); text-align: center; padding: 2rem 0;">
      {{ activeTag ? `没有标签为「${activeTag}」的论文` : '暂无论文解读，快去精读一篇吧！' }}
    </p>
  </div>
</template>
