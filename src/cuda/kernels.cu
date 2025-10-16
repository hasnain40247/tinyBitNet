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


//----------------------- Kernels For Matrix Addition --------------------------

__global__ void add_forward_kernel(const double* A, const double* B, double* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void add_backward_kernel(double* dA, double* dB, const double* dC, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dA[idx] = dC[idx];
        dB[idx] = dC[idx];
    }
}

void add(const double* h_A, const double* h_B, double* h_C, size_t n) {
    double *d_A, *d_B, *d_C;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_1d(n);
    add_forward_kernel<<<cfg.grid, cfg.block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void add_broadcast_forward_kernel(
    const double* A, const double* B, double* C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        C[idx] = A[idx] + B[col];  
    }
}

__global__ void add_broadcast_backward_kernel(
    double* dA, double* dB, const double* dC, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        dA[idx] = dC[idx];

        atomicAdd(&dB[col], dC[idx]); // basically avoid racings
    }
}


void add_backward(const double* h_dC, double* h_dA, double* h_dB, size_t n) {
    double *d_dA, *d_dB, *d_dC;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_dA, bytes);
    cudaMalloc(&d_dB, bytes);
    cudaMalloc(&d_dC, bytes);

    cudaMemcpy(d_dC, h_dC, bytes, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_1d(n);
    add_backward_kernel<<<cfg.grid, cfg.block>>>(d_dA, d_dB, d_dC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dA, d_dA, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dB, d_dB, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_dA);
    cudaFree(d_dB);
    cudaFree(d_dC);
}

void CudaOps::add_broadcast(const double* h_A, const double* h_B, double* h_C, int M, int N) {
    size_t size_A = M * N * sizeof(double);
    size_t size_B = N * sizeof(double);
    size_t size_C = M * N * sizeof(double);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    auto [blocks, threads] = make_launch_1d(M * N);
    add_broadcast_forward_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void CudaOps::add_broadcast_backward(double* h_dA, double* h_dB, const double* h_dC, int M, int N) {
    size_t size_A = M * N * sizeof(double);
    size_t size_B = N * sizeof(double);
    size_t size_C = M * N * sizeof(double);

    double *d_dA, *d_dB, *d_dC;
    cudaMalloc(&d_dA, size_A);
    cudaMalloc(&d_dB, size_B);
    cudaMalloc(&d_dC, size_C);

    cudaMemcpy(d_dC, h_dC, size_C, cudaMemcpyHostToDevice);
    cudaMemset(d_dB, 0, size_B); 

    auto [blocks, threads] = make_launch_1d(M * N);
    add_broadcast_backward_kernel<<<blocks, threads>>>(d_dA, d_dB, d_dC, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dA, d_dA, size_A, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dB, d_dB, size_B, cudaMemcpyDeviceToHost);

    cudaFree(d_dA);
    cudaFree(d_dB);
    cudaFree(d_dC);
}




void relu(const double* h_A, double* h_C, size_t n) {
    //op= C=relu(X) X= MxN | C=MxN
    double *d_A, *d_C;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);


    LaunchConfig cfg = make_launch_1d(n);
    relu_forward_kernel<<<cfg.grid, cfg.block>>>(d_A, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
}

__global__ void relu_forward_kernel(
    const double* A, double* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] =  A[idx] > 0.0 ? A[idx] : 0.0; 
    }
}



void CudaOps::relu(const double* h_A, double* h_C, size_t n) {
    //op= C=relu(X) X= MxN | C=MxN
    double *d_A, *d_C;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);


    LaunchConfig cfg = make_launch_1d(n);
    relu_forward_kernel<<<cfg.grid, cfg.block>>>(d_A, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
}

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


void CudaOps::relu_backward(const double* h_A, const double* h_dOut, double* h_dX, size_t n) {
    size_t bytes = n * sizeof(double);
    double *d_A, *d_dOut, *d_dX;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_dOut, bytes);
    cudaMalloc(&d_dX, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dOut, h_dOut, bytes, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_1d(n);
    relu_backward_kernel<<<cfg.grid, cfg.block>>>(d_A, d_dOut, d_dX, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dX, d_dX, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_dOut);
    cudaFree(d_dX);
}







void CudaOps::gelu(const double* h_A, double* h_C, size_t n) {
    //op= C=gelu(X) X= MxN | C=MxN
    double *d_A, *d_C;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);


    LaunchConfig cfg = make_launch_1d(n);
    gelu_forward_kernel<<<cfg.grid, cfg.block>>>(d_A, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
}

__global__ void gelu_forward_kernel(
    const double* A, double* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double c = sqrt(2.0 / M_PI); 


         

    if (idx < size) {
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

     double c = std::sqrt(2.0 / M_PI);
  
    if (idx < size) {
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






void CudaOps::gelu_backward(const double* h_A, const double* h_dOut, double* h_dX, size_t n) {
    size_t bytes = n * sizeof(double);
    double *d_A, *d_dOut, *d_dX;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_dOut, bytes);
    cudaMalloc(&d_dX, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dOut, h_dOut, bytes, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_1d(n);
    gelu_backward_kernel<<<cfg.grid, cfg.block>>>(d_A, d_dOut, d_dX, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dX, d_dX, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_dOut);
    cudaFree(d_dX);
}

