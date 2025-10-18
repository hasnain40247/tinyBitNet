

#include "utils.cuh"

__global__ void relu_forward_kernel(
    const double* A, double* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] =  A[idx] > 0.0 ? A[idx] : 0.0; 
    }
}



__global__ void relu_backward_kernel(
    const double* A,         
    const double* dOut,      
    double* dX,               
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dX[idx] = (A[idx] > 0.0 ? 1.0 : 0.0) * dOut[idx];
    }
}


