# QuaRot: 基于旋转的无异常值4bit推理

> **原文**: QuaRot: Outlier-Free 4-bit Inference in Rotated LLMs
> **作者**: Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman
> **发表时间**: 2024年 (arXiv:2404.00456)
> **arXiv**: https://arxiv.org/abs/2404.00456

---

## 一句话总结

这篇论文首次实现**全4bit推理**(权重+激活+KV Cache全部INT4),通过**随机Hadamard变换**旋转LLM消除激活异常值,无需保留任何高精度通道,Llama-2-70B仅损失0.47困惑度,8bit量化甚至完全无损且无需校准数据。

## 研究背景

### 激活异常值难题

之前的量化方法(如SmoothQuant)发现:
- 激活中存在少量异常值通道(值特别大)
- 这些异常值导致量化误差巨大
- 现有方案:保留异常值通道为高精度(如256个通道保持FP16)

**问题**: 混合精度难以在硬件上高效实现。

### QuaRot的核心洞察

**如果能把异常值"旋转掉",让所有通道分布均匀,不就可以全4bit量化了吗?**

关键发现:**随机Hadamard变换**可以消除异常值,而且通过计算不变性,可以融合到权重中,运行时零开销!

## 核心方法

### 1. 数学基础

#### Hadamard矩阵

递归定义:
$$\mathbf{H}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad \mathbf{H}_{2^n} = \mathbf{H}_2 \otimes \mathbf{H}_{2^{n-1}}$$

**关键性质**:
- 正交矩阵: $\mathbf{H}\mathbf{H}^T = \mathbf{I}$
- 计算 $\mathbf{H}\mathbf{x}$ 仅需 $O(d \log_2 d)$ 操作
- 随机H矩阵: $\tilde{\mathbf{H}} = \mathbf{H} \cdot \text{diag}(\mathbf{s})$, $\mathbf{s} \in \{+1,-1\}^d$

#### 计算不变性定理

**核心公式**:
$$\text{RMSNorm}(\mathbf{X}) = \text{RMSNorm}(\mathbf{X}\mathbf{Q}^T)\mathbf{Q}$$

原因:RMSNorm按行归一化 $\mathbf{x}_i \leftarrow \mathbf{x}_i / \|\mathbf{x}_i\|$,旋转不改变范数。

因此可以:
- 输入侧权重左乘 $\mathbf{Q}$
- 输出侧权重右乘 $\mathbf{Q}^T$
- **模型输出完全不变!**

### 2. 两阶段量化流程

#### Stage 1: 权重修改与Hadamard插入(全精度)

**Step 1a: 旋转FFN输入**
- 将RMSNorm缩放参数吸收到权重
- 选择随机Hadamard矩阵 $\mathbf{Q}$
- 输入权重: $\mathbf{W} \leftarrow \mathbf{Q}^T \cdot \text{diag}(\alpha) \cdot \mathbf{W}$
- 激活变为: $\mathbf{X} \leftarrow \mathbf{X}\mathbf{Q}$ (已无异常值)

**Step 1b: 旋转FFN内部**
- 在down-projection前插入在线Hadamard
- 融合到down权重: $\mathbf{W}_{\text{down}} \leftarrow \mathbf{H} \cdot \mathbf{W}_{\text{down}} \cdot \mathbf{Q}$

**Step 1c: 旋转注意力**
- Value投影: $\mathbf{W}_v \leftarrow \mathbf{W}_v \cdot (\mathbf{I} \otimes \mathbf{H}_{d_h})$
- $\mathbf{W}_{\text{out}} \leftarrow (\mathbf{I} \otimes \mathbf{H}_{d_h}) \cdot \mathbf{W}_{\text{out}}$
- Key/Query: 在线head-wise Hadamard旋转

#### Stage 2: 量化

- **权重**: 使用GPTQ量化到INT4
- **激活**: 对称per-token量化到INT4
- **KV Cache**: 非对称量化,group size=128

**所有矩阵乘法都在INT4下执行!**

## 效果

### WikiText-2困惑度

| 方法 | 异常值通道 | Llama-2-7B | 13B | 70B |
|------|-----------|-----------|-----|-----|
| FP16基线 | - | 5.47 | 4.88 | 3.32 |
| SmoothQuant | 0 | 83.12 | 35.88 | - |
| QUIK-4B | 256 | 8.87 | 7.78 | 6.91 |
| **QuaRot** | **0** | **6.10** | **5.40** | **3.79** |

**QuaRot在零异常值通道下超越所有基线!**

### Zero-Shot任务

| 模型 | 方法 | 平均准确率 |
|------|------|-----------|
| Llama-2-7B | FP16 | 69.82 |
| | QuaRot | 65.64 (94%) |
| Llama-2-70B | FP16 | 77.07 |
| | QuaRot | 75.98 (**99%**) |

### RTN量化(无需校准!)

| 模型 | 精度 | PPL | Zero-Shot |
|------|------|-----|-----------|
| 7B | INT8 | 5.50 | 69.65 (vs 69.82 FP16) |
| 70B | INT8 | 3.33 | 77.17 (vs 77.07 FP16) |

**8bit RTN完全无损,无需任何校准数据!**

### 性能加速

**Prefill加速**(RTX 3090, seq=2048):
| 模型 | BS=1 | BS=64 |
|------|------|-------|
| Llama-2-7B | 1.97× | **2.16×** |
| Llama-2-70B | 3.16× | **3.33×** |

**解码内存节省**:
| 模型 | 配置 | 基线 | QuaRot | 节省 |
|------|------|------|--------|------|
| 7B | BS=16, seq=4096 | 1.416GB | 0.378GB | **3.75×** |
| 70B | BS=16, seq=2048 | 1.738GB | 0.447GB | **3.89×** |

## 读后感

QuaRot是**理论优雅与工程实用的完美结合**:
1. **Hadamard旋转消除异常值**: 数学上优美,工程上高效
2. **全4bit推理**: 首次实现,无需混合精度
3. **8bit无损**: 简单RTN即可,无需校准

这为LLM的极致量化开辟了新方向,后续的DFRot等工作在此基础上继续改进。
