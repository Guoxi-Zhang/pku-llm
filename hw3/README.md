# hw3

rsync -avz --delete -e "ssh -p 30470 -i /Users/guoxi/.ssh/id_rsa" pku0030@36.170.52.66:/home/pku0030/hw3/homework_3 /Users/guoxi/Coding/ML/pku-llm/hw3

## 基线运行

- 原始CPU代码
    - average 0.076 Billion Interactions / second
- 基础的GPU加速
    - BLOCK_SIZE 128: average 32.128 Billion Interactions / second
    - BLOCK_SIZE 64: average 32.307 Billion Interactions / second
    - BLOCK_SIZE 32: average 32.457 Billion Interactions / second
    - BLOCK_SIZE 30: average 32.539 Billion Interactions / second
    - BLOCK_SIZE 16: average 31.542 Billion Interactions / second
    - BLOCK_SIZE 13: average 31.866 Billion Interactions / second
    - BLOCK_SIZE 8: average 29.460 Billion Interactions / second
    - BLOCK_SIZE 7: average 32.508 Billion Interactions / second
    - BLOCK_SIZE 4: average 31.365 Billion Interactions / second
    - BLOCK_SIZE 2: average 25.995 Billion Interactions / second
    - BLOCK_SIZE 1: average 14.300 Billion Interactions / second
- 内存优化的GPU加速
    - BLOCK_SIZE 128:
      - BLOCK_STRIDE 128: average 130.664 Billion Interactions / second
      - BLOCK_STRIDE 64: average 179.435 Billion Interactions / second
      - BLOCK_STRIDE 47: average 223.101 Billion Interactions / second
      - BLOCK_STRIDE 32: average 265.043 Billion Interactions / second
      - BLOCK_STRIDE 25: average 234.646 Billion Interactions / second
      - BLOCK_STRIDE 16: average 236.966 Billion Interactions / second
      - BLOCK_STRIDE 4: average 195.084 Billion Interactions / second
      - BLOCK_STRIDE 2: average 157.385 Billion Interactions / second
      - BLOCK_STRIDE 1: average 94.626 Billion Interactions / second
    - BLOCK_SIZE 32:
      - BLOCK_STRIDE 128: average 148.209 Billion Interactions / second
      - BLOCK_STRIDE 64: 
      - BLOCK_STRIDE 47: 
      - BLOCK_STRIDE 32: average 258.509 Billion Interactions / second
      - BLOCK_STRIDE 25: 
      - BLOCK_STRIDE 16: 
      - BLOCK_STRIDE 4: average 206.362 Billion Interactions / second
      - BLOCK_STRIDE 2: average 151.556 Billion Interactions / second
      - BLOCK_STRIDE 1: average 89.336 Billion Interactions / second
    - BLOCK_SIZE 4:
      - BLOCK_STRIDE 128: average 49.651 Billion Interactions / second
      - BLOCK_STRIDE 64: 
      - BLOCK_STRIDE 47: average 65.870 Billion Interactions / second
      - BLOCK_STRIDE 32: average 67.951 Billion Interactions / second
      - BLOCK_STRIDE 25: 
      - BLOCK_STRIDE 16: 
      - BLOCK_STRIDE 4: average 57.733 Billion Interactions / second
      - BLOCK_STRIDE 2: 
      - BLOCK_STRIDE 1: average 41.456 Billion Interactions / second
    - BLOCK_SIZE 1:
      - BLOCK_STRIDE 128: average 7.382 Billion Interactions / second
      - BLOCK_STRIDE 64: 
      - BLOCK_STRIDE 47: 
      - BLOCK_STRIDE 32: average 8.500 Billion Interactions / second
      - BLOCK_STRIDE 25: 
      - BLOCK_STRIDE 16: 
      - BLOCK_STRIDE 4: average 8.532 Billion Interactions / second
      - BLOCK_STRIDE 2: 
      - BLOCK_STRIDE 1: average 6.804 Billion Interactions / second

## 性能分析与理解

- 哪几行代码的修改，使得`nbody_parallel.cu`比`01-nbody.cu`执行快？为什么？
  1. 将`bodyForce`函数改为CUDA核函数(`__global__`)并行执行 —— 使得计算可以在GPU的大量并行计算单元上同时进行,而不是在CPU上串行执行
  2. 使用`cudaMallocManaged`统一内存管理 - 简化了CPU和GPU之间的数据传输,自动处理数据迁移,提高效率
  3. 将position integration也改为并行执行的核函数 - 同样利用GPU并行性能,加速位置更新计算
  4. 使用`cudaFree`替代`free`

- `nbody_shared.cu`比`nbody_parallel.cu`执行快约6倍(195.084/32.457 ≈ 6)
  - 原因:主要是通过共享内存优化和计算分块策略大幅提升了内存访问效率,减少了对全局内存的访问次数

- 哪几行代码的修改，使得`nbody_shared.cu`比`nbody_parallel.cu`执行快？为什么？
  1. 引入共享内存(`__shared__ float3 spos[BLOCK_SIZE]`)缓存粒子位置数据 - 共享内存访问延迟远低于全局内存,可以大幅提升访问速度
  2. 使用BLOCK_STRIDE进行数据分块处理 - 通过分块提高缓存命中率,减少内存访问次数
  3. 使用cudaMallocHost替代cudaMallocManaged - 可以实现更细粒度的内存控制,减少不必要的数据传输
  4. 使用原子操作(atomicAdd)确保并发安全 - 保证多线程访问共享数据的正确性

- `nbody_shared.cu`比`01-nbody.cu`执行快约2567倍(195.084/0.076 ≈ 2567)
  1. CPU串行计算改为GPU并行计算 - GPU拥有大量并行计算单元,可以同时处理多个粒子计算
  2. 优化的内存访问模式(共享内存、分块) - 大幅降低内存访问延迟和带宽需求
  3. 高效的线程组织和同步策略 - 合理的线程分配和同步机制提高了并行效率

## 参数分析与优化

- 分析`nbody_parallel.cu`中`BLOCK_SIZE`的取值对性能的影响。
  - 从数据可以看出,BLOCK_SIZE在30左右时性能最好(32.539 Billion Interactions/second)
  - BLOCK_SIZE过大(>64)或过小(<4)都会导致性能下降
  - 原因:
    1. BLOCK_SIZE过大会导致每个线程块占用过多资源,降低并行度
    2. BLOCK_SIZE过小会导致线程调度开销增加,且无法充分利用GPU硬件资源
    3. 30左右的BLOCK_SIZE能在资源利用和调度开销之间达到较好平衡

- 分析`nbody_shared.cu`中`BLOCK_SIZE`、`BLOCK_STRIDE`的取值对性能的影响。
  - BLOCK_SIZE的影响:
    1. BLOCK_SIZE=128时整体性能最好,最高可达265.043 Billion Interactions/second
    2. 随着BLOCK_SIZE减小,性能显著下降(BLOCK_SIZE=1时仅有8.532 Billion Interactions/second)
    3. 较大的BLOCK_SIZE有利于共享内存的高效利用,减少内存访问次数
  
  - BLOCK_STRIDE的影响:
    1. 对于BLOCK_SIZE=128,BLOCK_STRIDE=32时性能最佳(265.043 Billion Interactions/second)
    2. BLOCK_STRIDE过大或过小都会导致性能下降
    3. 合适的BLOCK_STRIDE可以优化内存访问模式,提高缓存命中率
    4. BLOCK_STRIDE的最优值与BLOCK_SIZE相关,需要根据具体情况调整

## 代码优化

- 内存访问优化：
  - 添加了内存对齐（ALIGNMENT）
  - 在Body结构体中添加了padding以优化内存访问模式
  - 使用float4替代float3来提高内存访问效率

- 寄存器优化：
  - 在bodyForce内核中预取数据到寄存器
  - 减少寄存器压力，使用更少的临时变量

- 循环优化：
  - 在外层循环使用#pragma unroll 4
  - 在内层循环使用#pragma unroll 8
  - 添加了边界检查以避免越界访问

- GPU优化：
  - 添加了GPU预热阶段
  - 获取设备属性以便进行更好的配置
  - 优化了线程块的配置方式

- 数据加载优化：
  - 使用协作加载方式加载数据到共享内存
  - 一次性读取和写回数据，减少内存访问次数

- 其他优化：
  - 简化了代码结构，提高可读性
  - 优化了循环边界条件的计算
  - 保持了原有的block stride策略


这个版本的优化保持了原有代码的基本结构，但对内存访问模式和计算效率进行了优化。主要的改进在于：

- 更好的内存对齐和访问模式
- 更高效的寄存器使用
- 更优的循环展开策添加了GPU预热阶段

你可以通过调整以下参数来进一步优化性能：

- BLOCK_SIZE：可以尝试64、128、256等不同值
- BLOCK_STRIDE：可以根据具体GPU的SM数量调整
- 循环展开的因子（目前是4和8）

average 429.085 Billion Interactions / second