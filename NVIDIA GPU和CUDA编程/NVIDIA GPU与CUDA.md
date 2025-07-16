## 1. NVIDIA GPU架构

<img src="https://raw.githubusercontent.com/wlkq151172/image_store/main/20250716204930462.png" alt="image-20250716204922136" style="zoom: 80%;" />

根据您提供的英伟达GPU架构图（示意图），以下是详细的技术解析：

------

### 1.1 **核心架构层级**

1. **流多处理器（SM）**
	- 作用：GPU的核心计算单元，每个SM包含：
		- 数百个CUDA核心（整数/浮点运算单元）
		- 深度学习张量核心（Tensor Core）
		- 寄存器堆（Register File）
	- **特点**：多个SM并行工作，同时处理数千个线程。
2. **L1缓存（一级缓存）**
	- **位置**：**每个SM独占**的快速缓存
	- **容量**：通常为32-128KB
	- **用途**：存储SM内部线程的临时数据，**访问延迟仅1-3个时钟周期**
3. **L2缓存（二级缓存）**
	- **位置**：**所有SM共享**的片上缓存
	- **容量**：现代GPU可达40-96MB（如H100为50MB）
	- **用途**：协调不同SM间的数据共享，**减少DRAM访问**
4. **DRAM（显存）**
	- **位置**：GPU板载的**高容量内存**
	- **技术**：GDDR6/GDDR6X（带宽500-1000GB/s）
	- **延迟**：约300个时钟周期（比L1慢100倍）

------

#### **数据传输路径（箭头方向）**

1. **计算流程**
	 `SM → L1 → L2 → DRAM`
	- SM执行计算时，优先从L1读取数据
	- 若L1未命中，则查询L2缓存
	- 若L2仍未命中，最终访问DRAM
2. **回写流程**
	 `DRAM → L2 → L1 → SM`
	- 计算结果按相反路径写回显存
	- 数据经过L2时被缓存，供其他SM复用

------

#### **关键技术优化意义**

1. **减少DRAM访问**

	- **问题**：DRAM带宽是计算瓶颈（延迟高、耗电大）

	- 

		解法

		：通过L1/L2缓存复用数据，如：

		- FlashAttention算法通过分块计算，将中间结果保留在L1缓存
		- 避免反复读写DRAM，提速高达**9倍**

2. **并行架构设计**

	- **SM级并行**：==每个SM独立处理线程块（如128线程）==
	- **Warp调度**：==SM内32线程为一组（Warp），硬件自动调度==
	- **内存墙突破**：L2缓存统一管理数据，避免SM间争抢显存

3. **延迟隐藏机制**

	- **原理**：当线程等待显存数据时，SM立即切换其他就绪线程
	- **效果**：保持计算单元利用率 >95%

------

#### **与深度学习的关系**

1. **大模型训练优化**

	- 如GPT-3训练时：
		- 激活值存储在DRAM → 通过L2缓存分发给各SM
		- 梯度聚合时L2缓存协调全局数据 → 减少通信开销

2. **推理加速关键**

	- 

		vLLM引擎

		：利用PagedAttention技术

		- 将KV Cache分段存储在L2缓存
		- 按需加载到SM的L1缓存，吞吐量提升**24倍**

------

### 1.2 SM内部

**CUDA Cores 和 Tensor Cores 均位于 GPU 的流多处理器（Streaming Multiprocessor, SM）内部**，它们是 SM 的核心计算单元，但在功能定位和设计目标上存在显著差异。以下是详细说明：

------

#### 🔧 **1. SM 是两者的物理载体**

- SM 是 GPU 的核心计算单元，负责执行所有并行计算任务。每个 SM 包含多种硬件资源：
	- **CUDA Cores**：通用计算核心，处理浮点（FP32/FP64）和整数运算[1,4,6](https://tencent.yuanbao/@ref)。
	- **Tensor Cores**：专用加速单元，专注于矩阵乘累加运算（如 `D = A*B + C`），支持混合精度（FP16/FP32, INT8 等）[2,3](https://tencent.yuanbao/@ref)。
	- 其他组件：Warp 调度器、寄存器文件、共享内存等[4,6](https://tencent.yuanbao/@ref)。

------

#### ⚙️ **2. 功能定位对比**

| **特性**     | **CUDA Cores**               | **Tensor Cores**                                             |
| ------------ | ---------------------------- | ------------------------------------------------------------ |
| **核心功能** | 通用标量运算（加法、乘法等） | 专用矩阵运算（GEMM）                                         |
| **计算模式** | 单指令多线程（SIMT）         | 单指令多数据（SIMD）优化                                     |
| **典型场景** | 条件分支、逻辑控制等通用任务 | 深度学习训练/推理（如矩阵乘法）                              |
| **计算效率** | 1 操作/时钟周期              | 64 次乘加操作/时钟周期（Volta）[2](https://tencent.yuanbao/@ref) |

> 💡 **Tensor Cores 的高效性**：以 Volta 架构为例，单个 Tensor Core 每周期可完成 64 次浮点乘加操作（FMA），是 CUDA Core 吞吐量的数十倍[2,3](https://tencent.yuanbao/@ref)。

------

#### 🔄 **3. 协作与资源调度机制**

- **共享 SM 资源**：两者共享 SM 内的寄存器文件、共享内存和调度单元[1,4](https://tencent.yuanbao/@ref)。

- 

	并行执行能力

	：

	- 当 SM 同时调度使用 CUDA Core 的 Warp（如逻辑控制）和使用 Tensor Core 的 Warp（如矩阵乘法）时，两者可**并行工作**，提升整体吞吐量[1](https://tencent.yuanbao/@ref)。
	- 实验数据表明：混合调度 CUDA Core 和 Tensor Core 任务时，SM 利用率（Overlap Rate）可达 **45%**[1](https://tencent.yuanbao/@ref)。

------

#### ⚠️ **4. 资源竞争与优化挑战**

虽然两者可并行，但以下因素可能限制效率：

1. 资源冲突：
	- 若单一任务（如 GEMM）独占 Tensor Core，则 CUDA Core 可能闲置[1](https://tencent.yuanbao/@ref)。
	- 寄存器或共享内存不足时，即使硬件空闲也无法启动新 Warp[1,5](https://tencent.yuanbao/@ref)。
2. 调度限制：
	- 传统 CUDA Stream 按顺序调度 Kernel，导致 CUDA/Tensor Core 无法同时激活（需通过 Persistent Block 等机制优化）[1](https://tencent.yuanbao/@ref)。

------

#### 📊 **5. 架构演进与核心配置**

不同 NVIDIA 架构中 SM 的核心配比如下：

| **架构**   | **每 SM 的 CUDA Cores** | **每 SM 的 Tensor Cores** | **代表 GPU** |
| ---------- | ----------------------- | ------------------------- | ------------ |
| **Volta**  | 64 × FP32 + 32 × FP64   | 8                         | V100         |
| **Ampere** | 64 × FP32 + 64 × INT32  | 4（第三代）               | A100         |
| **Hopper** | 128 × FP32              | 4（第四代，支持 FP8）     | H100         |

> 📌 注：Ampere 及后续架构的 Tensor Cores 支持更多数据类型（如 TF32、BF16）[2,3](https://tencent.yuanbao/@ref)。

------

### 💎 **总结**

- **物理位置**：CUDA Cores 和 Tensor Cores **均集成在 SM 内部**，共享存储与调度资源。
- **功能互补**：CUDA Core 处理通用逻辑，Tensor Core 加速矩阵运算，两者可通过 Warp 级调度实现并行。
- **优化关键**：需通过内核融合（如 FlashAttention）、混合精度计算及细粒度调度（如 Persistent Block）最大化协同效率。

### 1.3 FlashAttentionV2的Warp Partitioning策略

![img](https://raw.githubusercontent.com/wlkq151172/image_store/main/20250716213308188.jpeg)

<img src="https://raw.githubusercontent.com/wlkq151172/image_store/main/20250716213138545.jpeg" alt="img" style="zoom:67%;" />

​		送进去的QKV的块，V2在实际的GPU计算单元里面还会切分到不同的warp，只不过V2划分二点是Q而不是KV，一个warp里面保留了完整的KV

“首先看fwd，相比V1，V2改进了Warp Partition：4个warp会从smem的K/V tile load同样的数据做mma计算，但是load 不同Q，把V1 sliced-K sliced-V 改成了v2 sliced-Q，V1的做法是需要warp之间产生同步通信的，因为在计算QK结果乘V的时候，如图所示需要跨warp reduction得到O的结果，而且fwd的目的是沿着行方向计算softmax，行方向信息最后要汇总的，这也需要跨warp不同。V2就不需要了，这样可以减少同步开销。”