#pragma once
#include <cuda_runtime.h>

class CudaOps {
public:
    static void matmul(const double* h_A, const double* h_B, double* h_C, int M, int N, int K);
    static void add(const double* h_A, const double* h_B, double* h_C, int M, int N, int K);

};
