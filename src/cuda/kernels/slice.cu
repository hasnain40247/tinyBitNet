#include "utils.cuh"

__global__ void slice_cols_forward_kernel(
    const double* A, double* C,
    int M, int N, int start_col, int width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < width) {
        int in_idx  = row * N + (start_col + col);
        int out_idx = row * width + col;
        C[out_idx] = A[in_idx];
    }
}

__global__ void slice_cols_backward_kernel(
    const double* grad_out, double* grad_in,
    int M, int N, int start_col, int width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < width) {
        int in_idx  = row * width + col;
        int out_idx = row * N + (start_col + col);
        grad_in[out_idx] = grad_out[in_idx];
    }
}
