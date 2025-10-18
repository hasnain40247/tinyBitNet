#pragma once

#include <cstddef>      
#include "kernels/utils.cuh"

class CudaOps {
public:

    static void add(const double* h_A, const double* h_B, double* h_C, size_t n);
    static void add_broadcast(const double* h_A, const double* h_B, double* h_C, int M, int N);

    static void add_backward(const double* h_dC, double* h_dA, double* h_dB, size_t n);
    static void add_broadcast_backward(double* h_dA, double* h_dB, const double* h_dC, int M, int N);

    static void relu(const double* h_A, double* h_C, size_t n);
    static void relu_backward(const double* h_A, const double* h_dOut, double* h_dX, size_t n);

    static void gelu(const double* h_A, double* h_C, size_t n);
    static void gelu_backward(const double* h_A, const double* h_dOut, double* h_dX, size_t n);

    static void scale(const double* h_A, double* h_C, int M, int N, double scale);
    static void scale_backward(const double* h_dC, double* h_dA, int M, int N, double scale);

    static void matmul(const double* h_A, const double* h_B, double* h_C, int M, int N, int K);
    static void matmul_backward(const double* h_A, const double* h_B, const double* h_grad_C,
                                double* h_grad_A, double* h_grad_B, int M, int N, int K);

    static void mul_broadcast(const double* h_A, const double* h_B, double* h_C, int M, int N);
    static void mul_broadcast_backward(const double* h_A, const double* h_B, const double* h_dOut,
                                       double* h_dA, double* h_dB, int M, int N);

    static void transpose(const double* h_A, double* h_C, int M, int N);
    static void transpose_backward(const double* h_dOut, double* h_dX, int M, int N);

    static void slice_cols(const double* h_A, double* h_C, int M, int N, int start_col, int width);
    static void slice_cols_backward(const double* h_grad_out, double* h_grad_in,
                                    int M, int N, int start_col, int width);
};
