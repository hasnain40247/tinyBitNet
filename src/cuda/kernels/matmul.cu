#include "utils.cuh"

__global__ void matmul_forward_kernel(const double* A, const double* B, double* C,
                                      int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


__global__ void matmul_backward_A_kernel(const double* grad_C, const double* B,
                                         double* grad_A, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        double sum = 0.0;
        for (int n = 0; n < N; n++) {
            sum += grad_C[row * N + n] * B[col * N + n];  // B^T
        }
        grad_A[row * K + col] = sum;
    }
}

__global__ void matmul_backward_B_kernel(const double* A, const double* grad_C,
                                         double* grad_B, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < K && col < N) {
        double sum = 0.0;
        for (int m = 0; m < M; m++) {
            sum += A[m * K + row] * grad_C[m * N + col];  // A^T
        }
        grad_B[row * N + col] = sum;
    }
}




// Forward: C[i,j] = A[i,j] * B[j]
__global__ void mul_broadcast_forward_kernel(
    const double* A,
    const double* B,
    double* C,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int j = idx % N;
        C[idx] = A[idx] * B[j];
    }
}


__global__ void mul_broadcast_backward_kernel(
    const double* A,
    const double* B,
    const double* dOut,
    double* dA,
    double* dB,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int j = idx % N;

        dA[idx] = dOut[idx] * B[j];

        atomicAdd(&dB[j], dOut[idx] * A[idx]);
    }
}
