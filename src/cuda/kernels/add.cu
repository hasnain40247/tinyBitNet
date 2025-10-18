
#include "utils.cuh"


__global__ void add_forward_kernel(const double* A, const double* B, double* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void add_backward_kernel(double* dA, double* dB, const double* dC, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dA[idx] = dC[idx];
        dB[idx] = dC[idx];
    }
}


__global__ void add_broadcast_forward_kernel(
    const double* A, const double* B, double* C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        C[idx] = A[idx] + B[col];  
    }
}

__global__ void add_broadcast_backward_kernel(
    double* dA, double* dB, const double* dC, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        dA[idx] = dC[idx];

        atomicAdd(&dB[col], dC[idx]); // basically avoid racings
    }
}

