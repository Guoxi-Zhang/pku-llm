# hw3
同步代码：
`rsync -avz --delete -e "ssh -p 30470 -i /Users/guoxi/.ssh/id_rsa" pku0030@36.170.52.66:/home/pku0030/hw3/homework_3 /Users/guoxi/Coding/ML/pku-llm/hw3`

## 基线运行

- 原始CPU代码
  - average 0.076 Billion Interactions / second
- 基础的GPU加速
  - BLOCK_SIZE 128: average 32.128 Billion Interactions / second
- 内存优化的GPU加速
  - BLOCK_SIZE 128，BLOCK_STRIDE 4: average 195.084 Billion Interactions / second

## 性能分析与理解

- 哪几行代码的修改，使得`nbody_parallel.cu`比`01-nbody.cu`执行快？为什么？
  1. 将`bodyForce`函数改为CUDA核函数(`__global__`)并行执行 —— 使得计算可以在GPU的大量并行计算单元上同时进行,而不是在CPU上串行执行
  ```c
  __global__ void bodyForce(Body *p, float dt, int n)
      
  bodyForce<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies);
  ```
  2. 使用`cudaMallocManaged`统一内存管理 —— 简化了CPU和GPU之间的数据传输,自动处理数据迁移,提高效率

      ```c
      // buf = (float *)malloc(bytes);
      cudaMallocManaged(&buf, bytes);
      ```
  
  3. 将position integration也改为并行执行的核函数 —— 同样利用GPU并行性能,加速位置更新计算
  
      ```c
      __global__ void integrate_position(Body *p, float dt, int n)
      {
          int i = threadIdx.x + blockDim.x * blockIdx.x;
          if (i < n)
          {
              p[i].x += p[i].vx * dt;
              p[i].y += p[i].vy * dt;
              p[i].z += p[i].vz * dt;
          }
      }
      integrate_position<<<numberOfBlocks,threadsPerBlock>>>(p,dt,nBodies);
      ```
  
  4. 使用`cudaFree`替代`free`
  
      ```c
      cudaFree(buf);
      ```
  
- `nbody_shared.cu`比`nbody_parallel.cu`执行快约6倍(195.084/32.457 ≈ 6)
  - 原因:  主要是通过共享内存优化和计算分块策略大幅提升了内存访问效率,减少了对全局内存的访问次数

- 哪几行代码的修改，使得`nbody_shared.cu`比`nbody_parallel.cu`执行快？为什么？
  1. 引入共享内存缓存粒子位置数据 —— 共享内存访问延迟远低于全局内存,可以大幅提升访问速度
  
      ```c
      __shared__ float3 spos[BLOCK_SIZE];
      ```
  
  2. 使用BLOCK_STRIDE进行数据分块处理 —— 通过分块提高缓存命中率,减少内存访问次数
  
      ```c
      #define BLOCK_STRIDE 4
      // 计算要处理的数据index
      int i = threadIdx.x + (int)(blockIdx.x / BLOCK_STRIDE) * blockDim.x;
      // 计算要处理的数据块的起始位置
      int start_block = blockIdx.x % BLOCK_STRIDE;
      for (int block_num = start_block; block_num < cycle_times; block_num += BLOCK_STRIDE)
      {
          temp = p[block_num * BLOCK_SIZE + threadIdx.x];
          spos[threadIdx.x] = make_float3(temp.x, temp.y, temp.z);
          // 块内同步，防止spos提前被读取
          __syncthreads();
      // 编译优化，只有 BLOCK_SIZE 为常量时才有用
      #pragma unroll
          for (int j = 0; j < BLOCK_SIZE; j++)
          {
              dx = spos[j].x - ptemp.x;
              dy = spos[j].y - ptemp.y;
              dz = spos[j].z - ptemp.z;
              distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
              invDist = rsqrtf(distSqr);
              invDist3 = invDist * invDist * invDist;
              Fx += dx * invDist3;
              Fy += dy * invDist3;
              Fz += dz * invDist3;
          }
          // 块内同步，防止spos提前被写入
          __syncthreads();
      }
      ```
  
  3. 使用cudaMallocHost替代cudaMallocManaged —— 可以实现更细粒度的内存控制,减少不必要的数据传输
  
      ```c
      cudaMallocHost(&buf, bytes);
      ```
  
  4. 使用原子操作(atomicAdd)确保并发安全 —— 保证多线程访问共享数据的正确性
  
      ```c
      // 块之间不同步，原子加保证正确性
              atomicAdd(&p[i].vx, dt * Fx);
              atomicAdd(&p[i].vy, dt * Fy);
              atomicAdd(&p[i].vz, dt * Fz);
      ```
  
- `nbody_shared.cu`比`01-nbody.cu`执行快约2567倍(195.084/0.076 ≈ 2567)，原因：
  
  1. CPU串行计算改为GPU并行计算 —— GPU拥有大量并行计算单元,可以同时处理多个粒子计算
  2. 优化的内存访问模式(共享内存、分块) —— 大幅降低内存访问延迟和带宽需求
  3. 高效的线程组织和同步策略 —— 合理的线程分配和同步机制提高了并行效率

## 参数分析与优化

不同参数设置之下的运行时长表格如下：

基础的GPU加速：

| BLOCK_SIZE                                | 128    | 64     | 32     | 30     | 16     | 13     | 8      | 7      | 4      | 2      | 1      |
| :---------------------------------------- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- |
| **Average Billion Interactions / Second** | 32.128 | 32.307 | 32.457 | 32.539 | 31.542 | 31.866 | 29.460 | 32.508 | 31.365 | 25.995 | 14.300 |

分析`nbody_parallel.cu`中`BLOCK_SIZE`的取值对性能的影响。

- BLOCK_SIZE绝大数情况，影响不是很大
- BLOCK_SIZE过小(<4)会导致性能下降
- 原因:
    1. BLOCK_SIZE过小会导致线程调度开销增加,且无法充分利用GPU硬件资源

内存优化的GPU加速：

| BLOCK_STRIDE \ BLOCK_SIZE | 128     | 32      | 4      | 1     |
| :------------------------ | :------ | :------ | :----- | :---- |
| 128                       | 130.664 | 148.209 | 49.651 | 7.382 |
| 64                        | 179.435 | -       | -      | -     |
| 47                        | 223.101 | -       | 65.870 | -     |
| 32                        | 265.043 | 258.509 | 67.951 | 8.500 |
| 25                        | 234.646 | -       | -      | -     |
| 16                        | 236.966 | -       | -      | -     |
| 4                         | 195.084 | 206.362 | 57.733 | 8.532 |
| 2                         | 157.385 | 151.556 | -      | -     |
| 1                         | 94.626  | 89.336  | 41.456 | 6.804 |

- 分析`nbody_shared.cu`中`BLOCK_SIZE`、`BLOCK_STRIDE`的取值对性能的影响。
  - BLOCK_SIZE的影响:
    1. BLOCK_SIZE=128时整体性能最好，原因：
        - 128是GPU warp size(32)的4倍,可以充分利用硬件资源
        - 足够大的block size可以隐藏内存延迟
        - 不会过度占用每个SM的资源
    2. 随着BLOCK_SIZE减小,性能显著下降(BLOCK_SIZE=1时仅有8.532 Billion Interactions/second)
    3. 较大的BLOCK_SIZE有利于共享内存的高效利用,减少内存访问次数
  - BLOCK_STRIDE的影响:
    1. 最优值在32附近,原因是:
        - 32正好是一个warp的大小,有利于内存合并访问
        - 较小的stride可以提高缓存命中率
        - 过大的stride会导致内存访问模式不连续
        - 过小的stride会增加同步开销
    2. BLOCK_STRIDE过大或过小都会导致性能下降

## 代码优化

下面是经过进一步内存对齐后的代码分析，性能在相同参数下为：average 429.085 Billion Interactions / second，相比使用共享内存的代码又提高了两倍多。主要的改进在于：

- 更好的内存对齐和访问模式
- 更高效的寄存器使用
- 更优的循环展开策添加了GPU预热阶段

下面是具体分析：

- 内存访问优化：
  
  ```c
  // 内存对齐定义
  #define ALIGNMENT 32
  
  // Body结构体优化
  typedef struct {
      float x, y, z;    // 位置
      float pad;        // 内存对齐
      float vx, vy, vz; // 速度
      float pad2;       // 内存对齐
  } Body;
  
  // float4替代float3
  float4 pos = make_float4(p[i].x, p[i].y, p[i].z, 0.0f);
  float4 vel = make_float4(p[i].vx, p[i].vy, p[i].vz, 0.0f);
  ```
  
  - 添加了内存对齐（ALIGNMENT）
  - 在Body结构体中添加了padding以优化内存访问模式
  - 使用float4替代float3来提高内存访问效率
  
- 寄存器优化：
  
  ```c
  // 预取数据到寄存器
  float px = p[i].x;
  float py = p[i].y;
  float pz = p[i].z;
  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  ```
  
  - 在bodyForce内核中预取数据到寄存器
  - 减少寄存器压力，使用更少的临时变量
  
- 循环优化：
  
  ```c
  // 外层循环展开
  #pragma unroll 4
  for (int block_num = start_block; block_num < cycle_times; block_num += BLOCK_STRIDE) {
      // 内层循环展开
      #pragma unroll 8
      for (int j = 0; j < BLOCK_SIZE && (block_num * BLOCK_SIZE + j) < n; j++) {
          // ...
      }
  }
  ```
  
  - 在外层循环使用#pragma unroll 4
  - 在内层循环使用#pragma unroll 8
  - 添加了边界检查以避免越界访问
  
- GPU优化：
  
  ```c
  // GPU属性获取
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  
  // GPU预热
  cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
  bodyForce<<<blocks, threads>>>(d_p, dt, nBodies);
  integrate_position<<<(nBodies + threads.x - 1) / threads.x, threads>>>(d_p, dt, nBodies);
  cudaDeviceSynchronize();
  ```
  
  - 添加了GPU预热阶段
  - 获取设备属性以便进行更好的配置
  - 优化了线程块的配置方式
  
- 数据加载优化：
  
  ```c
  // 协作加载到共享内存
  __shared__ float4 shPos[BLOCK_SIZE];
  if (idx < n) {
      shPos[threadIdx.x] = make_float4(p[idx].x, p[idx].y, p[idx].z, 0.0f);
  }
  __syncthreads();
  
  // 一次性读写数据
  float4 pos = make_float4(p[i].x, p[i].y, p[i].z, 0.0f);
  float4 vel = make_float4(p[i].vx, p[i].vy, p[i].vz, 0.0f);
  ```
  
  - 使用协作加载方式加载数据到共享内存
  
  - 一次性读取和写回数据，减少内存访问次数
  
      

