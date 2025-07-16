# FlashAttention

之前对Attention的改进，着眼于减少计算量

FlashAttention着眼于减少IO量

## 1.目标：避免Attention Matrix从HBM的读写

1.通过分块计算，融合多个操作，减少中间结果缓存 

2.反向传播时，重新计算中间结果

![image-20250317150255519](https://raw.githubusercontent.com/wlkq151172/image_store/main/20250716112541500.png)

## 2.解决softmax的分块计算问题

1. 解决数值溢出 

2. softmax分块

	<img src="https://raw.githubusercontent.com/wlkq151172/image_store/main/20250716112058122.png" alt="image-20250317150906210" style="zoom:67%;" />

3. 算法流程

  <img src="https://raw.githubusercontent.com/wlkq151172/image_store/main/20250716112150976.png" alt="image-20250317151345123" style="zoom:67%;" />

  ![image-20250318092935488](https://raw.githubusercontent.com/wlkq151172/image_store/main/20250716112312845.png)

 

![image-20250318092954076](https://raw.githubusercontent.com/wlkq151172/image_store/main/20250716112422664.png)

<img src="https://raw.githubusercontent.com/wlkq151172/image_store/main/20250716112433540.png" alt="image-20250318093142230" style="zoom:67%;" />

## 3.FlashAttention分块的原理

块大小 `B_c` 和 `B_r` 的选取，完全来源于要把以下几部分都塞进片上 SRAM（容量为 `M` 个 scalar）：

1. 两个全尺寸的 Key/Value 块：各自大小为 `B_c * d`，共占用 `2*B_c*d`；
2. 一个 Query 块：大小为 `B_r * d`，占用 `B_r*d`；
3. 两个中间矩阵（打分矩阵 `S_{ij}` 和指数矩阵 `P_{ij}`），各自 `B_r * B_c`，共占用 `2*B_r*B_c`。

令这些加起来不超过 `M`：
$$
B_r\,d \;+\; 2\,B_c\,d \;+\; 2\,B_r\,B_c \;\le\; M
$$
一个对称且实现简单的解法是：
$$
B_c \;=\;\Bigl\lceil \frac{M}{4\,d}\Bigr\rceil,\quad
B_r \;=\;\min\!\Bigl(\lceil \tfrac{M}{4\,d}\rceil,\;d\Bigr)
$$
这样既保证了所有必须的数据能放入 SRAM，又让块尽可能大，减少对主存（HBM）的访问次数。当序列长度 `N` 比这两个值还小时，还要再加一个下界限制：
$$
B_c \le N,\quad B_r \le N
$$
最终在伪代码里就写成：

```
Set block sizes
  B_c = ⌈M/(4d)⌉,
  B_r = min(⌈M/(4d)⌉, d)
```

这样既满足
$$
\underbrace{B_r\,d}_{Q_i}+\underbrace{2B_c\,d}_{K_j,V_j}+\underbrace{2B_r B_c}_{S_{ij},\widetilde P_{ij}}\le M
$$
 又能获得最大的块维度，从而实现 FlashAttention 在 IO 上的最优性。 

```
for j = 1 … T_c:
  // 先把第 j 块的 K_j, V_j 从 HBM 载入片上 SRAM
  Load(K_j, V_j)
  
  for i = 1 … T_r:
    if 𝓜[i,j] == 0:
      continue     // 按稀疏 mask 跳过无需计算的块
    
    // 载入 Q_i 以及对应的 ℓ_i, m_i
    Load(Q_i, ℓ_i, m_i)

    // —— 计算打分矩阵（块级） ——
    S_ij = τ · Q_i · K_jᵀ    // 维度 B_r×B_c
    S_ij^masked = MASK(S_ij)

    // —— 数值稳定的分段 Softmax —— 
    // 1) 每行的最大值
    m_ij^new = row‑max(S_ij^masked)         ∈ ℝ^{B_r}
    // 2) exp 并求和
    P̃_ij = exp( S_ij^masked − m_i[:,None] )  ∈ ℝ^{B_r×B_c}
    ℓ̃_ij = row‑sum(P̃_ij)                   ∈ ℝ^{B_r}

    // 3) 合并到全局的 (m_i, ℓ_i)，保持数值稳定（log‑sum‑exp trick）
    m_i^new = max( m_i, m_ij^new )
    ℓ_i^new = exp(m_i − m_i^new)·ℓ_i
            + exp(m_ij^new − m_i^new)·ℓ̃_ij
    m_i ← m_i^new,   ℓ_i ← ℓ_i^new

    // —— Dropout & 输出累加 ——
    P^drop_ij = dropout(P̃_ij / ℓ̃_ij,  p_drop)  // 先归一化，再做 dropout
    // 累加到输出块 O_i
    O_i ← O_i + P^drop_ij · V_j

    // 写回更新后的 O_i, ℓ_i, m_i 到 HBM
    Write(O_i, ℓ_i, m_i)
  end for
end for

```

## 4.FlashAttentionV2

### 4.1改进1统一rescale

“推迟归一化／缩放（rescale）”的小优化，目的是少做几次硬件上代价较高的除法运算。我们分两步来看：

------

#### 1. 为什么要 rescale？

在做分块 Softmax 时，每个 block 会算出一个未归一化的权重矩阵
$$
\widetilde P_{ij} \;=\;\exp\bigl(S_{ij}^{\rm masked}-m_{ij}\bigr)
$$
然后要除以它对应的行和：
$$
\ell_{ij}=\sum_k\widetilde P_{ij}[\,\cdot,k]
$$
才能变成真正的概率分布：
$$
P_{ij} = \widetilde P_{ij} \;/\;\ell_{ij}\,
$$
这个
$$
/\ell_{ij}
$$
就是我们所说的 **rescale**。

------

#### 2. 传统做法 vs. FA2 做法

1. **传统做法（每个 block 都 rescale）**

	```
	for 每个 block (i,j):
	  计算  P̃ = exp(S−m)
	  计算  ℓ  = sum(P̃)
	  P  = P̃ / ℓ          ← 这里有一次除法
	  O_i += P · V_j
	```

	每迭代一个 block，就要做一次“除以 ℓ” 的操作。假设有 100 个 block，就得做 100 次除法。

2. **FA2 的做法（最后一次才 rescale）**

	- 在循环里先不做除法，只维护两组累积量：

		- 行最大值 `m_i`（用于数值稳定）
		- 未归一化的累积和 `l_i`

	- 同时把对输出 O_i 的累加，也用带指数差的乘法来替代“先除后乘”：
		$$
		O_i \;\leftarrow\; e^{m_i-m_i^{\rm new}}\,O_i \;+\; e^{m_{ij}-m_i^{\rm new}}\,\widetilde P_{ij}\,V_j
		$$
		这里完全没有做除法，只有指数和乘法。

	- **等所有 block 都跑完之后**，再用最后累计得到的 `l_i` 去一次性归一化整个输出（或者在下游需要真正概率的时候再做除法）。

------

#### 3. 节省的就是那一次次的除法

- **硬件上**，除法运算比乘法、指数都要慢很多。
- FA2 推迟 rescale，将多次“除法”变成一次“除法＋一堆乘法＋指数差”的形式，就极大减少了除法次数，整体速度就上去了。

------

**小结**：

> “先不在每个 block 的每次迭代中做完整的 rescale，而是等全部跑完才做一次 rescale”，
>  就是把每个 block 里本来要做的“`/\ell`”除法，推迟到最后一次性做，省掉了中间无数次的除法，从而加速整体计算。

------

### 4.2改进2循环调换

​		对比 FA1 和 FA2 在做 Attention forward pass 时，数据搬运和并行化策略上的一个细节优化——把两层循环的顺序给调换了。

**FA1 的写法（先 K/V 再 Q）**

```
for 每个 K‑V 块 K_j, V_j:
  load(K_j, V_j)            ← 外层循环先把 K_j、V_j 拉进来
  for 每个 Q 块 Q_i:
    load(Q_i)               ← 内层循环再把 Q_i 拉进来
    compute attention for Q_i against K_j
    read/write O_i 到全局内存
```

**问题：**

- 每次内层迭代都要从全局内存把 Q_i 和 O_i 拉进来，再写回去，造成大量不必要的全局读写。
- 而且不同的 Q_i 之间其实是完全独立的计算，彼此无需通信，却被绑在同一个外层循环里，浪费了并行化的机会。

**FA2 的改进（先 Q 再 K/V）**

```
for 每个 Q‑块 Q_i:
  load(Q_i)                 ← 外层循环先把 Q_i 拉进来
  for 每个 K‑V 块 K_j, V_j:
    load(K_j, V_j)          ← 内层循环再把 K_j、V_j 拉进来
    compute attention
    累加到 O_i（只在片上做 multiply‑add，最后再一次性写回）
```

**好处：**

1. **减少全局内存访问**
	 Q_i、O_i 一次 load 就能用完整个内层循环，不用反复读写。
2. **天然并行**
	 每个 Q_i 的 Attention 计算彼此独立，可以分给不同的 GPU thread‑block／SM 去跑，互不干扰，更好地利用硬件并行度。
3. **布局更合理**
	 先固定一个 Query 块，把所有 K/V 块都跟它做完，再切下一个 Query 块；数据局部性更好，寄存器／片上 SRAM 利用率也更高。

**总结：**
 FA2 就是把原来“先外层循环 load K/V，再内层循环 load Q”的反模式，改成“先 load Q，再循环 load 各个 K/V”，这样对每个 Q_i 只 load／write 一次全局数据，大幅降低带宽压力，并且把不同 Q_i 之间的计算切分给不同线程块，获得更好的并行效率。

------
