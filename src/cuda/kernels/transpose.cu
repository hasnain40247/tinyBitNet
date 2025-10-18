

#include "utils.cuh"



__global__ void transpose_forward_kernel(
    const double* A, double* C, int M,int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < M && col < N) {
        C[col * M + row] = A[row * N + col];  
    }


}


__global__ void transpose_backward_kernel(
    const double* dOut,   
    double* dX,           
    int M, int N)          
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        dX[row * N + col] = dOut[col * M + row]; 
    }
}


