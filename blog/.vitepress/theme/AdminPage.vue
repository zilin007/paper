<script setup>
import { ref, onMounted, computed } from 'vue'

// ─── 状态 ───
const loading = ref(true)
const saving = ref(false)
const error = ref('')
const successMsg = ref('')
const devMode = ref(true)
const activeTab = ref('sort')   // 'sort' | 'tags'

// 配置数据
const paperOrder = ref([])       // slug 数组（排序顺序）
const pinnedPapers = ref([])     // slug 数组
const paperTags = ref({})        // { slug: [tag1, tag2] }
const tagColors = ref({})        // { tag: '#color' }

// 论文元信息
const papers = ref([])           // [{ slug, title, date }]

// 标签编辑
const newTagName = ref('')
const newTagColor = ref('#3b82f6')
const editingTag = ref(null)     // { name, color }
const editTagName = ref('')
const editTagColor = ref('#3b82f6')

// 论文标签分配
const tagAssignPaper = ref('')   // 当前选中论文 slug
const tagAssignValue = ref('')   // 下拉选中的标签

// ─── 加载数据 ───
async function loadData() {
  loading.value = true
  error.value = ''
  try {
    const res = await fetch('/api/admin/config')
    if (!res.ok) throw new Error(`API 不可用 (${res.status})`)
    const data = await res.json()

    const config = data.config || {}
    papers.value = data.papers || []

    paperOrder.value = config.paperOrder || []
    pinnedPapers.value = config.pinnedPapers || []
    paperTags.value = config.paperTags || {}
    tagColors.value = config.tagColors || {}

    // 确保所有论文都在 paperOrder 中
    for (const p of papers.value) {
      if (!paperOrder.value.includes(p.slug)) {
        paperOrder.value.push(p.slug)
      }
    }
  } catch (e) {
    devMode.value = false
    error.value = e.message
  } finally {
    loading.value = false
  }
}

// ─── 保存 ───
async function saveConfig() {
  saving.value = true
  error.value = ''
  successMsg.value = ''
  try {
    const config = {
      paperOrder: paperOrder.value,
      pinnedPapers: pinnedPapers.value,
      paperTags: paperTags.value,
      tagColors: tagColors.value,
    }
    const res = await fetch('/api/admin/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    if (!res.ok) throw new Error('保存失败')
    const result = await res.json()
    if (result.success) {
      successMsg.value = '保存成功，同步已完成'
      setTimeout(() => (successMsg.value = ''), 3000)
    }
  } catch (e) {
    error.value = '保存失败: ' + e.message
  } finally {
    saving.value = false
  }
}

// ─── 排序操作 ───
const sortedPaperList = computed(() => {
  return paperOrder.value.map((slug) => {
    const info = papers.value.find((p) => p.slug === slug)
    return {
      slug,
      title: info?.title || slug,
      date: info?.date || '',
      pinned: pinnedPapers.value.includes(slug),
      tags: paperTags.value[slug] || [],
    }
  })
})

function moveUp(index) {
  if (index <= 0) return
  const arr = [...paperOrder.value]
  ;[arr[index - 1], arr[index]] = [arr[index], arr[index - 1]]
  paperOrder.value = arr
}

function moveDown(index) {
  if (index >= paperOrder.value.length - 1) return
  const arr = [...paperOrder.value]
  ;[arr[index], arr[index + 1]] = [arr[index + 1], arr[index]]
  paperOrder.value = arr
}

function moveToTop(index) {
  if (index <= 0) return
  const arr = [...paperOrder.value]
  const [item] = arr.splice(index, 1)
  arr.unshift(item)
  paperOrder.value = arr
}

function moveToBottom(index) {
  if (index >= paperOrder.value.length - 1) return
  const arr = [...paperOrder.value]
  const [item] = arr.splice(index, 1)
  arr.push(item)
  paperOrder.value = arr
}

function togglePin(slug) {
  const idx = pinnedPapers.value.indexOf(slug)
  if (idx >= 0) {
    pinnedPapers.value.splice(idx, 1)
  } else {
    pinnedPapers.value.push(slug)
  }
}

// ─── 标签操作 ───
const allTags = computed(() => Object.keys(tagColors.value))

function addTag() {
  const name = newTagName.value.trim()
  if (!name || tagColors.value[name]) return
  tagColors.value[name] = newTagColor.value
  newTagName.value = ''
  newTagColor.value = '#3b82f6'
}

function startEditTag(tag) {
  editingTag.value = tag
  editTagName.value = tag
  editTagColor.value = tagColors.value[tag] || '#3b82f6'
}

function confirmEditTag() {
  if (!editingTag.value) return
  const oldName = editingTag.value
  const newName = editTagName.value.trim()
  if (!newName) return

  // 更新颜色
  if (oldName !== newName) {
    // 重命名：更新所有引用
    delete tagColors.value[oldName]
    tagColors.value[newName] = editTagColor.value

    for (const slug of Object.keys(paperTags.value)) {
      const tags = paperTags.value[slug]
      const idx = tags.indexOf(oldName)
      if (idx >= 0) tags[idx] = newName
    }
  } else {
    tagColors.value[oldName] = editTagColor.value
  }

  editingTag.value = null
}

function cancelEditTag() {
  editingTag.value = null
}

function deleteTag(tag) {
  delete tagColors.value[tag]
  for (const slug of Object.keys(paperTags.value)) {
    paperTags.value[slug] = paperTags.value[slug].filter((t) => t !== tag)
  }
}

// 论文标签分配
function addTagToPaper(slug, tag) {
  if (!tag) return
  if (!paperTags.value[slug]) paperTags.value[slug] = []
  if (!paperTags.value[slug].includes(tag)) {
    paperTags.value[slug].push(tag)
  }
  tagAssignValue.value = ''
}

function removeTagFromPaper(slug, tag) {
  if (!paperTags.value[slug]) return
  paperTags.value[slug] = paperTags.value[slug].filter((t) => t !== tag)
}

function getTagStyle(tag) {
  const color = tagColors.value[tag] || '#64748b'
  return {
    backgroundColor: color + '18',
    color: color,
    borderColor: color + '40',
  }
}

onMounted(loadData)
</script>

<template>
  <!-- 非开发模式 -->
  <div v-if="!devMode" class="admin-notice">
    <h2>管理功能仅在本地开发环境下可用</h2>
    <p>请在本地运行 <code>npm run dev</code> 后访问此页面。</p>
  </div>

  <!-- 加载中 -->
  <div v-else-if="loading" class="admin-loading">加载中...</div>

  <!-- 管理界面 -->
  <div v-else class="admin-container">
    <!-- 顶部操作栏 -->
    <div class="admin-toolbar">
      <div class="admin-tabs">
        <button
          :class="['admin-tab', { active: activeTab === 'sort' }]"
          @click="activeTab = 'sort'"
        >
          排序管理
        </button>
        <button
          :class="['admin-tab', { active: activeTab === 'tags' }]"
          @click="activeTab = 'tags'"
        >
          标签管理
        </button>
      </div>
      <button class="admin-save-btn" :disabled="saving" @click="saveConfig">
        {{ saving ? '保存中...' : '保存配置' }}
      </button>
    </div>

    <!-- 消息提示 -->
    <div v-if="error" class="admin-msg admin-msg-error">{{ error }}</div>
    <div v-if="successMsg" class="admin-msg admin-msg-success">{{ successMsg }}</div>

    <!-- ===== 排序管理 ===== -->
    <div v-show="activeTab === 'sort'" class="admin-section">
      <p class="admin-hint">拖拽或使用按钮调整论文显示顺序。置顶的论文始终排在最前面。</p>

      <div class="sort-list">
        <div
          v-for="(item, index) in sortedPaperList"
          :key="item.slug"
          :class="['sort-item', { 'sort-item-pinned': item.pinned }]"
        >
          <div class="sort-item-order">{{ index + 1 }}</div>
          <div class="sort-item-info">
            <div class="sort-item-title">
              <span v-if="item.pinned" class="sort-pin-badge">置顶</span>
              {{ item.title }}
            </div>
            <div class="sort-item-meta">
              <span v-if="item.date" class="sort-item-date">{{ item.date }}</span>
              <span v-for="tag in item.tags" :key="tag" class="sort-item-tag" :style="getTagStyle(tag)">{{ tag }}</span>
            </div>
          </div>
          <div class="sort-item-actions">
            <button title="置顶" @click="moveToTop(index)" :disabled="index === 0">⏫</button>
            <button title="上移" @click="moveUp(index)" :disabled="index === 0">⬆️</button>
            <button title="下移" @click="moveDown(index)" :disabled="index === sortedPaperList.length - 1">⬇️</button>
            <button title="置底" @click="moveToBottom(index)" :disabled="index === sortedPaperList.length - 1">⏬</button>
            <button
              :title="item.pinned ? '取消置顶' : '设为置顶'"
              :class="['pin-btn', { active: item.pinned }]"
              @click="togglePin(item.slug)"
            >
              📌
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== 标签管理 ===== -->
    <div v-show="activeTab === 'tags'" class="admin-section">
      <!-- 标签列表 -->
      <h3 class="admin-subtitle">已有标签</h3>
      <div class="tag-list-admin">
        <div v-for="tag in allTags" :key="tag" class="tag-item-admin">
          <template v-if="editingTag === tag">
            <input v-model="editTagName" class="tag-edit-input" @keyup.enter="confirmEditTag" />
            <input type="color" v-model="editTagColor" class="tag-color-input" />
            <button class="tag-action-btn confirm" @click="confirmEditTag">确认</button>
            <button class="tag-action-btn cancel" @click="cancelEditTag">取消</button>
          </template>
          <template v-else>
            <span class="tag-preview" :style="getTagStyle(tag)">{{ tag }}</span>
            <span class="tag-color-swatch" :style="{ backgroundColor: tagColors[tag] }"></span>
            <button class="tag-action-btn" @click="startEditTag(tag)">编辑</button>
            <button class="tag-action-btn delete" @click="deleteTag(tag)">删除</button>
          </template>
        </div>
      </div>

      <!-- 新增标签 -->
      <div class="tag-add-row">
        <input
          v-model="newTagName"
          class="tag-add-input"
          placeholder="标签名称"
          @keyup.enter="addTag"
        />
        <input type="color" v-model="newTagColor" class="tag-color-input" />
        <button class="tag-action-btn confirm" @click="addTag" :disabled="!newTagName.trim()">
          添加标签
        </button>
      </div>

      <!-- 论文标签分配 -->
      <h3 class="admin-subtitle" style="margin-top: 2rem;">论文标签分配</h3>
      <div class="tag-assign-list">
        <div v-for="item in sortedPaperList" :key="item.slug" class="tag-assign-item">
          <div class="tag-assign-title">{{ item.title }}</div>
          <div class="tag-assign-tags">
            <span
              v-for="tag in (paperTags[item.slug] || [])"
              :key="tag"
              class="tag-assign-chip"
              :style="getTagStyle(tag)"
            >
              {{ tag }}
              <button class="tag-remove-btn" @click="removeTagFromPaper(item.slug, tag)">&times;</button>
            </span>
            <select
              class="tag-assign-select"
              @change="addTagToPaper(item.slug, $event.target.value); $event.target.value = ''"
            >
              <option value="">+ 添加标签</option>
              <option
                v-for="tag in allTags.filter(t => !(paperTags[item.slug] || []).includes(t))"
                :key="tag"
                :value="tag"
              >
                {{ tag }}
              </option>
            </select>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
