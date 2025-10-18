
#include <math_constants.h>
#include "utils.cuh"


__global__ void gelu_forward_kernel(
    const double* A, double* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


         

    if (idx < size) {
    double c = sqrt(2.0 / CUDART_PI); 

    double x = A[idx];
    double x3 = x * x * x;      
    double inner = c * (x + 0.044715 * x3);
    C[idx] = 0.5 * x * (1.0 + tanh(inner));

    }
}



__global__ void gelu_backward_kernel(
    const double* A,         
    const double* dOut,      
    double* dX,               
    int size
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

  
    if (idx < size) {
     double c = sqrt(2.0 / CUDART_PI);

        double x = A[idx];
        double x3 = x * x * x;
        double x2 = x * x;     



        double inner = c * (x + 0.044715 * x3);
        double tanh_inner = tanh(inner);
        double sech2 = 1.0 - tanh_inner * tanh_inner;

        double term1 = 0.5 * (1.0 + tanh_inner);
        double term2 = 0.5 * x * sech2 * c * (1 + 3 * 0.044715 * x2);
        double addition=term1 + term2;
        dX[idx] = addition * dOut[idx];
        
    }
}


