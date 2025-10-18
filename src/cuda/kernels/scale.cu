

#include "utils.cuh"

__global__ void scale_forward_kernel(const double* A, double* C, double scale, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        C[idx] = A[idx] * scale;
    }
}



__global__ void scale_backward_kernel(
    const double* dC,  
    double* dA,       
    double scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dA[idx] = dC[idx] * scale;
    }
}
