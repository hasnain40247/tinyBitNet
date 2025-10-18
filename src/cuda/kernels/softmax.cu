#include "utils.cuh"
#include <cuda_runtime.h>

__global__ void softmax_forward_kernel(const float* __restrict__ A, float* __restrict__ Out,
                                       int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    // find max value for numerical stability
    float max_val = -1e20f;
    for (int j = 0; j < cols; j++)
        max_val = fmaxf(max_val, A[row * cols + j]);

    // compute exponentials and sum
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float expv = expf(A[row * cols + j] - max_val);
        Out[row * cols + j] = expv;
        sum += expv;
    }

    for (int j = 0; j < cols; j++)
        Out[row * cols + j] /= sum;
}

__global__ void softmax_backward_kernel(const float* __restrict__ Out,
                                        const float* __restrict__ GradOut,
                                        float* __restrict__ GradIn,
                                        int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    for (int j = 0; j < cols; j++) {
        float yj = Out[row * cols + j];
        float sum = 0.0f;

        for (int k = 0; k < cols; k++) {
            float yk = Out[row * cols + k];
            float delta = (j == k) ? 1.0f : 0.0f;
            sum += (delta - yk) * GradOut[row * cols + k];
        }
        GradIn[row * cols + j] = yj * sum;
    }
}
