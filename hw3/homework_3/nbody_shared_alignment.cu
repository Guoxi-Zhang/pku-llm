#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f
#define BLOCK_SIZE 128
#define BLOCK_STRIDE 32
#define ALIGNMENT 32  // 内存对齐优化

typedef struct {
    float x, y, z;    // 位置
    float pad;        // 内存对齐
    float vx, vy, vz; // 速度
    float pad2;       // 内存对齐
} Body;

void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__ void bodyForce(Body *p, float dt, int n) {
    // 使用更大的共享内存来减少全局内存访问
    __shared__ float4 shPos[BLOCK_SIZE];
    
    int i = threadIdx.x + (int)(blockIdx.x / BLOCK_STRIDE) * blockDim.x;
    int start_block = blockIdx.x % BLOCK_STRIDE;
    
    if (i < n) {
        // 预取数据到寄存器
        float px = p[i].x;
        float py = p[i].y;
        float pz = p[i].z;
        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;

        int cycle_times = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        #pragma unroll 4
        for (int block_num = start_block; block_num < cycle_times; block_num += BLOCK_STRIDE) {
            int idx = block_num * BLOCK_SIZE + threadIdx.x;
            // 协作加载数据到共享内存
            if (idx < n) {
                shPos[threadIdx.x] = make_float4(p[idx].x, p[idx].y, p[idx].z, 0.0f);
            }
            __syncthreads();

            // 计算当前块内的力
            #pragma unroll 8
            for (int j = 0; j < BLOCK_SIZE && (block_num * BLOCK_SIZE + j) < n; j++) {
                float dx = shPos[j].x - px;
                float dy = shPos[j].y - py;
                float dz = shPos[j].z - pz;

                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                ax += dx * invDist3;
                ay += dy * invDist3;
                az += dz * invDist3;
            }
            __syncthreads();
        }

        // 使用原子操作更新速度
        atomicAdd(&p[i].vx, dt * ax);
        atomicAdd(&p[i].vy, dt * ay);
        atomicAdd(&p[i].vz, dt * az);
    }
}

__global__ void integrate_position(Body *p, float dt, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        // 一次性读取所有数据
        float4 pos = make_float4(p[i].x, p[i].y, p[i].z, 0.0f);
        float4 vel = make_float4(p[i].vx, p[i].vy, p[i].vz, 0.0f);
        
        // 更新位置
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;
        
        // 一次性写回数据
        p[i].x = pos.x;
        p[i].y = pos.y;
        p[i].z = pos.z;
    }
}

int main(const int argc, const char **argv) {
    int nBodies = 2<<11;
    int salt = 0;
    if (argc > 1) nBodies = 2<<atoi(argv[1]);
    if (argc > 2) salt = atoi(argv[2]);

    const float dt = 0.01f;
    const int nIters = 10;

    // 确保内存对齐
    size_t bytes = nBodies * sizeof(Body);
    bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
    float *buf;
    cudaMallocHost(&buf, bytes);
    randomizeBodies(buf, 6 * nBodies);

    double totalTime = 0.0;

    // 计算最优的线程块配置
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((nBodies + threads.x - 1) / threads.x * BLOCK_STRIDE);

    float *d_buf;
    cudaMalloc(&d_buf, bytes);
    Body *d_p = (Body *)d_buf;

    // 预热GPU
    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    bodyForce<<<blocks, threads>>>(d_p, dt, nBodies);
    integrate_position<<<(nBodies + threads.x - 1) / threads.x, threads>>>(d_p, dt, nBodies);
    cudaDeviceSynchronize();

    // 主循环
    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        bodyForce<<<blocks, threads>>>(d_p, dt, nBodies);
        integrate_position<<<(nBodies + threads.x - 1) / threads.x, threads>>>(d_p, dt, nBodies);

        if (iter == nIters-1) {
            cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
        }

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

#ifdef ASSESS
    checkPerformance(buf, billionsOfOpsPerSecond, salt);
#else
    checkAccuracy(buf, nBodies);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
    salt += 1;
#endif

    cudaFree(d_buf);
    cudaFreeHost(buf);
}
