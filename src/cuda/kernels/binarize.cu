#include "utils.cuh"

__global__ void binarize_forward_kernel(const double* A, double* C, double alpha, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        double v = A[idx] - alpha;
        C[idx] = (v >= 0.0) ? 1.0 : -1.0;
    }
}

__global__ void binarize_backward_kernel(const double* dOut, double* dA, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dA[idx] = dOut[idx];
    }
}
