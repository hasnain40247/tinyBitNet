#pragma once
#include <cuda_runtime.h>
#include <utility>

struct LaunchConfig {
    dim3 blocks;
    dim3 threads;
};

inline LaunchConfig make_launch_1d(size_t n, int block_size = 256) {
    int blocks = static_cast<int>((n + block_size - 1) / block_size);
    return {dim3(blocks), dim3(block_size)};
}

inline LaunchConfig make_launch_2d(int M, int N, int block_size = 16) {
    dim3 threads(block_size, block_size);
    dim3 blocks((N + block_size - 1) / block_size,
                (M + block_size - 1) / block_size);
    return {blocks, threads};
}
