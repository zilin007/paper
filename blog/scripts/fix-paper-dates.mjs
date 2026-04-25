#!/usr/bin/env node
/**
 * 批量统一所有论文解读文件中的日期格式为精确的 YYYY-MM-DD
 * 并统一使用 **发表时间** 字段名
 */

import { readFile, writeFile, readdir } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const PAPERS_DIR = join(__dirname, '..', '..', 'papers')

const DATES = {
  'AdaLoRA_Adaptive_Budget_Allocation': '2023-03-18',
  'Attention_Is_All_You_Need': '2017-06-12',
  'A_Survey_on_In-context_Learning': '2022-12-31',
  'Batch_Normalization': '2015-02-11',
  'Hard_Prompts_Made_Easy': '2023-02-07',
  'Language_Models_are_Few-Shot_Learners': '2020-05-28',
  'Layer_Normalization': '2016-07-21',
  'LoftQ_LoRA-Fine-Tuning-Aware_Quantization': '2023-10-12',
  'LongLoRA_Efficient_Fine-tuning_of_Long-Context_Large_Language_Models': '2023-09-21',
  'LoRA_Low-Rank_Adaptation_of_Large_Language_Models': '2021-06-17',
  'P-Tuning_v2_Prompt_Tuning_Can_Be_Comparable_to_Fine-tuning': '2021-10-14',
  'Parameter-Efficient_Transfer_Learning_for_NLP': '2019-02-02',
  'Prefix-Tuning_Optimizing_Continuous_Prompts_for_Generation': '2021-01-01',
  'SRLoRA_Subspace_Reorganization_Low-Rank_Adaptation': '2025-05-18',
  'Stochastic_Gradient_Descent_in_Non-Convex_Problems': '2025-04-17',
  'The_Power_of_Scale_for_Parameter-Efficient_Prompt_Tuning': '2021-04-18',
}

async function main() {
  const dirs = await readdir(PAPERS_DIR, { withFileTypes: true })
  const paperDirs = dirs.filter((d) => d.isDirectory() && !d.name.startsWith('.'))

  for (const dir of paperDirs) {
    const files = await readdir(join(PAPERS_DIR, dir.name))
    const mdFile = files.find((f) => f.endsWith('_解读.md'))
    if (!mdFile) continue

    const mdPath = join(PAPERS_DIR, dir.name, mdFile)
    let content = await readFile(mdPath, 'utf-8')

    const preciseDate = DATES[dir.name]
    if (!preciseDate) {
      console.log(`  ⚠ ${dir.name} 无精确日期，跳过`)
      continue
    }

    // 统一替换各种日期格式为 **发表时间**: YYYY-MM-DD
    // 匹配以下格式：
    // > **发表时间**: xxx
    // > - **发表**: xxx
    // > **发表**: xxx
    const oldContent = content
    content = content.replace(
      /(>\s*(?:-\s*)?\*\*发表(?:时间)?\*\*):\s*.+$/m,
      `> **发表时间**: ${preciseDate}`
    )

    if (content !== oldContent) {
      await writeFile(mdPath, content, 'utf-8')
      console.log(`  ✓ ${dir.name}: ${preciseDate}`)
    } else {
      console.log(`  ⚠ ${dir.name}: 未匹配到日期行，请手动检查`)
    }
  }

  console.log('\n日期格式统一完成')
}

main().catch((err) => {
  console.error('修复失败:', err)
  process.exit(1)
})
