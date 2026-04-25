/**
 * Vite 插件：博客管理后台 API（仅开发模式生效）
 *
 * 提供 RESTful 接口：
 *   GET  /api/admin/config   — 获取当前配置 + 论文元信息
 *   POST /api/admin/config   — 保存配置并触发同步
 *
 * 仅在 configureServer（dev 模式）中注册，生产构建不会包含。
 */

import { readFile, writeFile, readdir, stat } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { exec } from 'child_process'
import { promisify } from 'util'

const execAsync = promisify(exec)
const __dirname = dirname(fileURLToPath(import.meta.url))
const BLOG_DIR = join(__dirname, '..', '..')
const CONFIG_PATH = join(BLOG_DIR, 'blog-config.json')
const PAPERS_DIR = join(BLOG_DIR, '..', 'papers')

// 读取 blog-config.json
async function readConfig() {
  try {
    const raw = await readFile(CONFIG_PATH, 'utf-8')
    return JSON.parse(raw)
  } catch {
    return { paperOrder: [], pinnedPapers: [], paperTags: {}, tagColors: {} }
  }
}

// 写入 blog-config.json
async function writeConfig(config) {
  await writeFile(CONFIG_PATH, JSON.stringify(config, null, 2) + '\n', 'utf-8')
}

// 扫描 papers/ 目录获取论文元信息
async function scanPapers() {
  const papers = []
  try {
    await stat(PAPERS_DIR)
  } catch {
    return papers
  }

  const entries = await readdir(PAPERS_DIR, { withFileTypes: true })
  const dirs = entries.filter((e) => e.isDirectory() && !e.name.startsWith('.'))

  for (const dir of dirs) {
    const paperPath = join(PAPERS_DIR, dir.name)
    const files = await readdir(paperPath)
    const readingFile = files.find((f) => f.endsWith('_解读.md'))
    if (!readingFile) continue

    const content = await readFile(join(paperPath, readingFile), 'utf-8')
    const titleMatch = content.match(/^# (.+)$/m)
    const dateMatch = content.match(/\*\*发表时间\*\*:\s*(.+)$/m)

    papers.push({
      slug: dir.name,
      title: titleMatch?.[1]?.trim() || dir.name,
      date: dateMatch?.[1]?.trim() || '',
    })
  }

  return papers
}

// 运行同步脚本
async function runSync() {
  const syncScript = join(BLOG_DIR, 'scripts', 'sync-papers.mjs')
  try {
    const { stdout, stderr } = await execAsync(`node "${syncScript}"`, {
      cwd: BLOG_DIR,
      timeout: 30000,
    })
    return { success: true, stdout, stderr }
  } catch (err) {
    return { success: false, error: err.message }
  }
}

// 解析请求体
function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = ''
    req.on('data', (chunk) => (body += chunk))
    req.on('end', () => {
      try {
        resolve(JSON.parse(body))
      } catch {
        reject(new Error('Invalid JSON'))
      }
    })
    req.on('error', reject)
  })
}

// 发送 JSON 响应
function sendJson(res, data, status = 200) {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': '*',
  })
  res.end(JSON.stringify(data))
}

export function adminApiPlugin() {
  return {
    name: 'blog-admin-api',

    configureServer(server) {
      // 处理 CORS 预检
      server.middlewares.use((req, res, next) => {
        if (req.url?.startsWith('/api/admin') && req.method === 'OPTIONS') {
          res.writeHead(204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
          })
          res.end()
          return
        }
        next()
      })

      // GET /api/admin/config — 获取配置和论文列表
      server.middlewares.use(async (req, res, next) => {
        if (req.url !== '/api/admin/config' || req.method !== 'GET') return next()

        try {
          const [config, papers] = await Promise.all([readConfig(), scanPapers()])
          sendJson(res, { config, papers })
        } catch (err) {
          sendJson(res, { error: err.message }, 500)
        }
      })

      // POST /api/admin/config — 保存配置并同步
      server.middlewares.use(async (req, res, next) => {
        if (req.url !== '/api/admin/config' || req.method !== 'POST') return next()

        try {
          const newConfig = await parseBody(req)
          await writeConfig(newConfig)
          const syncResult = await runSync()
          sendJson(res, { success: true, sync: syncResult })
        } catch (err) {
          sendJson(res, { error: err.message }, 500)
        }
      })
    },
  }
}
