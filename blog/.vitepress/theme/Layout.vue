<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'
import GiscusComment from './GiscusComment.vue'

const { Layout } = DefaultTheme
const { frontmatter } = useData()
</script>

<template>
  <Layout>
    <!-- 在文章底部插入 Giscus 评论组件 -->
    <template #doc-after>
      <GiscusComment v-if="frontmatter.comment !== false" />
    </template>
  </Layout>
</template>
