# AWQ: 激活感知的LLM权重量化

> **原文**: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
> **作者**: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
> **发表时间**: 2023年 (MLSys)
> **arXiv**: https://arxiv.org/abs/2306.00978

---

## 一句话总结

这篇论文发现LLM中**只有1%的权重是重要的**,而且这些重要权重应该通过**激活分布**而非权重分布来识别,通过**放大显著通道**的等效变换,实现了4bit权重量化且精度无损,还配套了高效的推理引擎TinyChat实现3倍加速。

## 研究背景

### 权重量化的困境

4bit量化(16个级别)理论上可以节省4倍内存,但直接量化会导致精度下降。

**关键发现**: 不是所有权重量等重要!保持1%权重为FP16就能大幅改善量化效果。

**但问题**: 如何识别这1%的重要权重?
- 按权重大小选? ❌ 无效
- 按激活幅度选? ✓ 有效!

### 为什么激活分布能识别重要权重?

**直觉**: 如果某个输入通道的激活值经常很大,说明这个通道对输出的贡献大,对应的权重就重要。

这就像是:公司里经常接大项目的员工更重要,而不是工资高的员工更重要。

## 核心思路

### 关键Insight

**保护显著权重的最佳方式不是保持高精度,而是放大它们!**

通过数学等价变换:
$$\mathbf{Y} = \mathbf{W}\mathbf{X} = (\mathbf{W} \cdot \text{diag}(\mathbf{s})) \cdot (\text{diag}(\mathbf{s})^{-1} \cdot \mathbf{X})$$

放大显著权重通道(乘以 $s>1$),同时缩小对应激活(除以 $s$),输出不变但**相对量化误差减小**!

### 为什么放大能减少误差?

量化误差公式:
$$\text{Err} = \Delta \cdot \text{RoundErr}\left(\frac{w}{\Delta}\right) \cdot x$$

缩放后:
$$\text{Err}' = \Delta' \cdot \text{RoundErr}\left(\frac{ws}{\Delta'}\right) \cdot \frac{x}{s}$$

误差比: $\frac{\Delta'}{\Delta} \cdot \frac{1}{s} \approx \frac{1}{s}$ (因为 $\Delta' \approx \Delta$)

**当 $s=2$ 时,误差减半!**

## 具体方法

### AWQ算法流程

1. **收集激活统计**: 用少量校准数据(16个序列),记录每层输入的逐通道平均幅度 $\mathbf{s}_X$

2. **搜索最优缩放**:
   $$\mathbf{s} = \mathbf{s}_X^\alpha, \quad \alpha^* = \arg\min_\alpha \|\mathbf{W}\mathbf{X} - Q(\mathbf{W} \cdot \text{diag}(\mathbf{s}))(\text{diag}(\mathbf{s})^{-1} \cdot \mathbf{X})\|$$
   
   通过网格搜索($\alpha \in [0,1]$, 20个候选值)找到最优$\alpha$

3. **应用变换**:
   $$\mathbf{W}' = \mathbf{W} \cdot \text{diag}(\mathbf{s})$$
   然后对$\mathbf{W}'$进行量化

4. **权重裁剪**: 最小化MSE,调整量化边界

### 与SmoothQuant的对比

| 特性 | SmoothQuant | AWQ |
|------|------------|-----|
| **目标** | W8A8(权重+激活都量化) | W4A16(仅权重量化) |
| **平滑因子** | $s = \frac{\max(|X|)^\alpha}{\max(|W|)^{1-\alpha}}$ | $s = s_X^\alpha$ |
| **$\alpha$选择** | 固定0.5-0.9 | 网格搜索最优 |
| **是否需要校准** | 512个样本 | 16个序列 |
| **应用场景** | 服务器部署 | 边缘设备部署 |

## 效果

### LLaMA系列4bit量化结果

| 模型 | FP16 | RTN | GPTQ | **AWQ** |
|------|------|-----|------|---------|
| Llama-2-7B | - | 5.73 | 5.69 | **5.60** |
| Llama-2-13B | 5.47 | 4.98 | 4.98 | **4.97** |
| Llama-2-70B | 4.88 | 3.46 | 3.42 | **3.41** |
| LLaMA-7B | 5.68 | 5.96 | 6.22 | **5.78** |
| LLaMA-65B | 3.53 | 3.67 | 3.66 | **3.62** |

**AWQ在所有模型上都优于GPTQ!**

### 跨域泛化能力

**编程任务(MBPP, CodeLlama-7B)**:
| 方法 | pass@1 | pass@10 |
|------|--------|---------|
| FP16 | 38.53 | 49.77 |
| GPTQ | 31.97 | 44.75 |
| **AWQ** | **40.64** | **49.25** |

**数学任务(GSM8K)**:
| 模型 | FP16 | GPTQ | **AWQ** |
|------|------|------|---------|
| 7B | 13.87 | 12.13 | **13.57** |
| 70B | 56.41 | 56.03 | **56.40** |

**AWQ在专业任务上也保持接近FP16的性能,而GPTQ有明显下降!**

### TinyChat加速

| 平台 | FP16吞吐量 | AWQ W4A16 | 加速比 |
|------|-----------|-----------|--------|
| A100 | 81.6 tok/s | 155.3 tok/s | **1.9×** |
| RTX 4090 | 58.5 tok/s | 168.1 tok/s | **2.9×** |
| Jetson Orin | 11.5 tok/s | 35.6 tok/s | **3.1×** |

## 读后感

AWQ的核心贡献是**激活感知的量化理念**:
1. **重要权重由激活决定**,而非权重本身
2. **放大显著通道**比保持高精度更优雅
3. **无需反向传播**,避免过拟合校准集

这为边缘设备部署LLM提供了强大工具,与GPTQ正交(可组合使用)。
