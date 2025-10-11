#include "cuda/kernels.hpp"
#include <stdio.h>

__global__ void matmul_naive_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
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



__global__ void matmul_backward_A_kernel(const double* grad_C, const double* B, double* grad_A, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        double sum = 0.0;
        for (int n = 0; n < N; n++) {
            sum += grad_C[row * N + n] * B[col * N + n]; // B^T
        }
        grad_A[row * K + col] = sum;
    }
}


__global__ void matmul_backward_B_kernel(const double* A, const double* grad_C, double* grad_B, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < K && col < N) {
        double sum = 0.0;
        for (int m = 0; m < M; m++) {
            sum += A[m * K + row] * grad_C[m * N + col]; // A^T
        }
        grad_B[row * N + col] = sum;
    }
}



void CudaOps::matmul(const double* h_A, const double* h_B, double* h_C, int M, int N, int K) {
    size_t size_A = M * K * sizeof(double);
    size_t size_B = K * N * sizeof(double);
    size_t size_C = M * N * sizeof(double);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_naive_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void CudaOps::matmul_backward(const double* h_A, const double* h_B, const double* h_grad_C,
                              double* h_grad_A, double* h_grad_B, int M, int N, int K) {
    size_t size_A = M * K * sizeof(double);
    size_t size_B = K * N * sizeof(double);
    size_t size_C = M * N * sizeof(double);

    double *d_A, *d_B, *d_grad_C, *d_grad_A, *d_grad_B;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_grad_C, size_C);
    cudaMalloc(&d_grad_A, size_A);
    cudaMalloc(&d_grad_B, size_B);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_C, h_grad_C, size_C, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks_A((K + BLOCK_SIZE - 1)/BLOCK_SIZE, (M + BLOCK_SIZE - 1)/BLOCK_SIZE);
    dim3 numBlocks_B((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (K + BLOCK_SIZE - 1)/BLOCK_SIZE);

    matmul_backward_A_kernel<<<numBlocks_A, threadsPerBlock>>>(d_grad_C, d_B, d_grad_A, M, K, N);
    matmul_backward_B_kernel<<<numBlocks_B, threadsPerBlock>>>(d_A, d_grad_C, d_grad_B, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_grad_A, d_grad_A, size_A, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_B, d_grad_B, size_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_grad_C); cudaFree(d_grad_A); cudaFree(d_grad_B);
}
