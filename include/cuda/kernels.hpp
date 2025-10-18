#pragma once

#include <cstddef> 

__global__ void add_forward_kernel(const double* A, const double* B, double* C, size_t n);
__global__ void add_broadcast_forward_kernel(const double* A, const double* B, double* C, int M, int N);
__global__ void add_backward_kernel(double* dA, double* dB, const double* dC, size_t n);
__global__ void add_broadcast_backward_kernel(double* dA, double* dB, const double* dC, int M, int N);

__global__ void relu_forward_kernel(const double* A, double* C, size_t n);
__global__ void relu_backward_kernel(const double* A, const double* dOut, double* dX, size_t n);

__global__ void gelu_forward_kernel(const double* A, double* C, size_t n);
__global__ void gelu_backward_kernel(const double* A, const double* dOut, double* dX, size_t n);

__global__ void scale_forward_kernel(const double* A, double* C, double scale, int size);
__global__ void scale_backward_kernel(const double* dC, double* dA, double scale, int size);

__global__ void matmul_forward_kernel(const double* A, const double* B, double* C, int M, int N, int K);
__global__ void matmul_backward_A_kernel(const double* grad_C, const double* B, double* grad_A, int M, int K, int N);
__global__ void matmul_backward_B_kernel(const double* A, const double* grad_C, double* grad_B, int M, int K, int N);

__global__ void mul_broadcast_forward_kernel(const double* A, const double* B, double* C, int M, int N);
__global__ void mul_broadcast_backward_kernel(const double* A, const double* B, const double* dOut,
                                              double* dA, double* dB, int M, int N);

__global__ void transpose_forward_kernel(const double* A, double* C, int M, int N);
__global__ void transpose_backward_kernel(const double* dOut, double* dX, int M, int N);

__global__ void slice_cols_forward_kernel(const double* A, double* C, int M, int N, int start_col, int width);
__global__ void slice_cols_backward_kernel(const double* grad_out, double* grad_in, int M, int N, int start_col, int width);
