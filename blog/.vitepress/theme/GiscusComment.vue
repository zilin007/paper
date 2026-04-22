<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import { useRoute, useData } from 'vitepress'

const route = useRoute()
const { isDark } = useData()
const container = ref(null)

/**
 * Giscus 配置说明：
 * 1. 前往 https://giscus.app/zh-CN 完成配置
 * 2. 在你的 GitHub 仓库启用 Discussions 功能
 * 3. 安装 Giscus App: https://github.com/apps/giscus
 * 4. 将下方 TODO 标记的值替换为你的实际配置
 */
const GISCUS_CONFIG = {
  repo: 'zilin007/paper',          // TODO: 确认仓库名
  repoId: '',                       // TODO: 从 giscus.app 获取
  category: 'Announcements',
  categoryId: '',                   // TODO: 从 giscus.app 获取
}

function loadGiscus() {
  if (!container.value) return
  // 未配置时不加载
  if (!GISCUS_CONFIG.repoId || !GISCUS_CONFIG.categoryId) return

  container.value.innerHTML = ''

  const script = document.createElement('script')
  script.src = 'https://giscus.app/client.js'
  script.setAttribute('data-repo', GISCUS_CONFIG.repo)
  script.setAttribute('data-repo-id', GISCUS_CONFIG.repoId)
  script.setAttribute('data-category', GISCUS_CONFIG.category)
  script.setAttribute('data-category-id', GISCUS_CONFIG.categoryId)
  script.setAttribute('data-mapping', 'pathname')
  script.setAttribute('data-strict', '0')
  script.setAttribute('data-reactions-enabled', '1')
  script.setAttribute('data-emit-metadata', '0')
  script.setAttribute('data-input-position', 'top')
  script.setAttribute('data-theme', isDark.value ? 'dark' : 'light')
  script.setAttribute('data-lang', 'zh-CN')
  script.setAttribute('crossorigin', 'anonymous')
  script.async = true

  container.value.appendChild(script)
}

onMounted(() => {
  loadGiscus()
})

// 路由切换时重新加载评论
watch(() => route.path, () => {
  nextTick(() => loadGiscus())
})

// 主题切换时同步 Giscus 主题
watch(isDark, () => {
  const iframe = container.value?.querySelector('iframe.giscus-frame')
  if (iframe) {
    iframe.contentWindow?.postMessage(
      { giscus: { setConfig: { theme: isDark.value ? 'dark' : 'light' } } },
      'https://giscus.app',
    )
  }
})
</script>

<template>
  <div ref="container" class="giscus-container" />
</template>

<style scoped>
.giscus-container {
  margin-top: 2rem;
}
</style>
