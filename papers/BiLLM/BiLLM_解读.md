# BiLLM:  pushing the Limit of Post-Training Quantization for LLMs

> **原文**: BiLLM: Pushing the Limit of Post-Training Quantization for LLMs
> **作者**: Wei Huang, Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, Xiaojuan Qi
> **发表时间**: 2024年 (ICML)
> **arXiv**: https://arxiv.org/abs/2402.04291

---

## 一句话总结

这篇论文首次实现**1-bit后训练量化**,将LLM权重压缩至平均**1.08-1.11 bit**,通过**Hessian敏感度选择显著权重**+**二值残差近似**+**钟形分布分割**,在LLaMA2-70B上困惑度仅8.41,模型大小压缩近10倍。

## 研究背景

### 超低比特量化的挑战

当比特宽度≤2bit时,现有方法性能崩溃:

| 方法 | OPT-66B (WikiText PPL) |
|------|----------------------|
| FP16 | 9.34 |
| GPTQ 2bit | 82.10 (崩溃!) |
| GPTQ 1bit | 13106 (完全失效) |

**问题**: 二值化(1bit)是最激进的量化,误差极大。

### BiLLM的关键观察

1. **Hessian呈极长尾分布**: 少数权重对层输出影响巨大
2. **权重大小呈钟形分布**: 大多数权重聚集在零附近

**推论**: 少数权重重要,大多数权重冗余,但钟形分布二值化误差最大。

## 核心方法

### 1. 显著权重二值化

#### Hessian敏感度计算
$$s_i = \frac{w_i^2}{[\mathbf{H}^{-1}]_{ii}^2}$$

#### 结构化选择
发现敏感Hessian值主要集中于**特定列/行**,采用按列分割方式,搜索最优列数。

#### 二值残差近似(Binary Residual Approximation)

对显著权重进行**两次二值化**:
$$\begin{cases}
\alpha_o^*, \mathbf{B}_o^* = \arg\min \|\mathbf{W} - \alpha_o\mathbf{B}_o\|^2 \\
\alpha_r^*, \mathbf{B}_r^* = \arg\min \|(\mathbf{W} - \alpha_o^*\mathbf{B}_o^*) - \alpha_r\mathbf{B}_r\|^2
\end{cases}$$

最终近似:
$$\mathbf{W} \approx \alpha_o^*\mathbf{B}_o^* + \alpha_r^*\mathbf{B}_r^*$$

**残差误差证明**:
$$\mathcal{E}_{rb} \leq \|\mathbf{W} - \alpha_o^*\mathbf{B}_o^*\|^2$$

残差方法比直接二值化误差更低!

### 2. 钟形分布分割二值化

移除显著权重后,剩余权重保持钟形分布。

**分割策略**: 断点 $p$ 将非显著权重分为:
- **集中区域** $A_c[-p, p]$
- **稀疏区域** $A_s[-m, -p] \cup [p, m]$

**最优断点搜索**:
$$p^* = \arg\min_p (\theta_{q,p}^2)$$

通过百分位搜索(0.1, 0.2, ..., 9.0倍最大权重值)找到最优$p$。

### 3. 完整流程

```
Algorithm: BiLLM(W, X, β, λ)
1. H := 2XXᵀ                     // Hessian
2. H^c := Cholesky((H + λI)⁻¹)   // 逆Hessian
3. for each block b:
4.     rows := salient(W^b, H^c)  // 搜索显著列
5.     B̃₁ := res_approximation(W^b[rows])  // 残差近似
6.     p* := seg_search(W^b[non-rows])     // 搜索断点
7.     B̃₂ := binary(W^b[|w|≤p*])  // 集中区域
8.     B̃₃ := binary(W^b[|w|>p*])  // 稀疏区域
9.     应用OBC补偿更新剩余权重
10. end for
```

## 效果

### LLaMA系列结果

| 模型 | FP16 | GPTQ 2bit | PB-LLM 1.7bit | **BiLLM 1.08bit** |
|------|------|-----------|---------------|------------------|
| LLaMA-7B | 5.68 | 152.31 | 102.36 | **35.04** |
| LLaMA-13B | 5.09 | 20.44 | 36.60 | **15.14** |
| LLaMA-65B | 3.53 | 8.78 | 12.53 | **8.49** |
| LLaMA2-7B | 5.47 | 60.45 | 69.20 | **32.48** |
| LLaMA2-70B | 3.32 | 9.12 | 28.37 | **8.41** |

**BiLLM在1.08bit下大幅超越2bit GPTQ!**

### 模型大小对比

| 方法 | LLaMA-7B | 13B | 30B | 65B | LLaMA2-70B |
|------|----------|-----|-----|-----|-----------|
| FP16 | 13.5GB | 24.2GB | 60.5GB | 121.0GB | 129.3GB |
| **BiLLM** | **1.5GB** | **2.7GB** | **6.1GB** | **14.8GB** | **15.4GB** |

**压缩比接近10倍!**

### Zero-Shot任务(LLaMA-7B)

| 方法 | PIQA | BoolQ | Winogrande | Hellaswag |
|------|------|-------|------------|-----------|
| GPTQ 2bit | 52.8 | 50.0 | 49.3 | 26.3 |
| PB-LLM 1.7bit | 54.6 | 59.7 | 50.6 | 28.7 |
| **BiLLM 1.09bit** | **61.2** | **62.7** | **51.1** | **36.8** |

## 读后感

BiLLM突破了**后训练量化的极限**:
1. **首次实现1-bit PTQ**: 将LLM压缩到极致
2. **二值残差近似**: 用两次二值化逼近原始权重
3. **钟形分布分割**: 针对不同区域采用不同策略

虽然1-bit量化的精度仍有差距,但在极端资源受限场景下(如嵌入式设备),这为LLM部署提供了可能。
