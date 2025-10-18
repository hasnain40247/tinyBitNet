#include "kernels/utils.cuh"
#include "kernels/slice_cols.cu"




void CudaOps::add(const double* h_A, const double* h_B, double* h_C, size_t n) {
    double *d_A, *d_B, *d_C;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_1d(n);
    add_forward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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

    LaunchConfig cfg = make_launch_1d(M * N);
    add_broadcast_forward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_B, d_C, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


void CudaOps::add_backward(const double* h_dC, double* h_dA, double* h_dB, size_t n) {
    double *d_dA, *d_dB, *d_dC;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_dA, bytes);
    cudaMalloc(&d_dB, bytes);
    cudaMalloc(&d_dC, bytes);

    cudaMemcpy(d_dC, h_dC, bytes, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_1d(n);
    add_backward_kernel<<<cfg.blocks, cfg.threads>>>(d_dA, d_dB, d_dC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dA, d_dA, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dB, d_dB, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_dA);
    cudaFree(d_dB);
    cudaFree(d_dC);
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



void CudaOps::gelu(const double* h_A, double* h_C, size_t n) {
    //op= C=gelu(X) X= MxN | C=MxN
    double *d_A, *d_C;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);


    LaunchConfig cfg = make_launch_1d(n);
    gelu_forward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
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
    gelu_backward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_dOut, d_dX, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dX, d_dX, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_dOut);
    cudaFree(d_dX);
}



void CudaOps::slice_cols(
    const double* h_A, double* h_C,
    int M, int N, int start_col, int width
) {
    size_t size_A = M * N * sizeof(double);
    size_t size_C = M * width * sizeof(double);

    double *d_A, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_2d(M, width);
    slice_cols_forward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_C, M, N, start_col, width);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
}

void CudaOps::slice_cols_backward(
    const double* h_grad_out, double* h_grad_in,
    int M, int N, int start_col, int width
) {
    size_t size_grad_out = M * width * sizeof(double);
    size_t size_grad_in  = M * N * sizeof(double);

    double *d_grad_out, *d_grad_in;
    cudaMalloc(&d_grad_out, size_grad_out);
    cudaMalloc(&d_grad_in,  size_grad_in);

    cudaMemcpy(d_grad_out, h_grad_out, size_grad_out, cudaMemcpyHostToDevice);
    cudaMemset(d_grad_in, 0, size_grad_in);

    LaunchConfig cfg = make_launch_2d(M, width);
    slice_cols_backward_kernel<<<cfg.blocks, cfg.threads>>>(d_grad_out, d_grad_in, M, N, start_col, width);
    cudaDeviceSynchronize();

    cudaMemcpy(h_grad_in, d_grad_in, size_grad_in, cudaMemcpyDeviceToHost);

    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
}




void CudaOps::matmul(const double* h_A, const double* h_B, double* h_C,
                     int M, int N, int K) {
    size_t size_A = M * K * sizeof(double);
    size_t size_B = K * N * sizeof(double);
    size_t size_C = M * N * sizeof(double);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_2d(M, N, 16);
    matmul_forward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
void CudaOps::matmul_backward(const double* h_A, const double* h_B,
                              const double* h_grad_C,
                              double* h_grad_A, double* h_grad_B,
                              int M, int N, int K) {
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

    // grad_A = grad_C * B^T
    LaunchConfig cfgA = make_launch_2d(M, K, 16);
    matmul_backward_A_kernel<<<cfgA.blocks, cfgA.threads>>>(d_grad_C, d_B, d_grad_A, M, K, N);

    // grad_B = A^T * grad_C
    LaunchConfig cfgB = make_launch_2d(K, N, 16);
    matmul_backward_B_kernel<<<cfgB.blocks, cfgB.threads>>>(d_A, d_grad_C, d_grad_B, M, K, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_grad_A, d_grad_A, size_A, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_B, d_grad_B, size_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_grad_C);
    cudaFree(d_grad_A);
    cudaFree(d_grad_B);
}


void CudaOps::mul_broadcast(
    const double* h_A, const double* h_B, double* h_C,
    int M, int N
) {
    size_t bytes_A = M * N * sizeof(double);
    size_t bytes_B = N * sizeof(double);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_A);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_1d(M * N);
    mul_broadcast_forward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_B, d_C, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes_A, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void CudaOps::mul_broadcast_backward(
    const double* h_A, const double* h_B, const double* h_dOut,
    double* h_dA, double* h_dB,
    int M, int N
) {
    size_t bytes_A = M * N * sizeof(double);
    size_t bytes_B = N * sizeof(double);

    double *d_A, *d_B, *d_dOut, *d_dA, *d_dB;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_dOut, bytes_A);
    cudaMalloc(&d_dA, bytes_A);
    cudaMalloc(&d_dB, bytes_B);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dOut, h_dOut, bytes_A, cudaMemcpyHostToDevice);
    cudaMemset(d_dB, 0, bytes_B); 

    LaunchConfig cfg = make_launch_1d(M * N);
    mul_broadcast_backward_kernel<<<cfg.blocks, cfg.threads>>>(
        d_A, d_B, d_dOut, d_dA, d_dB, M, N
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_dA, d_dA, bytes_A, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dB, d_dB, bytes_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_dOut);
    cudaFree(d_dA);
    cudaFree(d_dB);
}


void CudaOps::relu(const double* h_A, double* h_C, size_t n) {
    //op= C=relu(X) X= MxN | C=MxN
    double *d_A, *d_C;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);


    LaunchConfig cfg = make_launch_1d(n);
    relu_forward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
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
    relu_backward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_dOut, d_dX, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dX, d_dX, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_dOut);
    cudaFree(d_dX);
}



void CudaOps::scale_backward(
    const double* h_dC,  
    double* h_dA,       
    int M, int N,
    double scale
) {
    int size = M * N;
    size_t bytes = size * sizeof(double);

    double *d_dC, *d_dA;
    cudaMalloc(&d_dC, bytes);
    cudaMalloc(&d_dA, bytes);

    cudaMemcpy(d_dC, h_dC, bytes, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_1d(size);
    scale_backward_kernel<<<cfg.blocks, cfg.threads>>>(d_dC, d_dA, scale, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dA, d_dA, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_dC);
    cudaFree(d_dA);
}

void CudaOps::scale(const double* h_A, double* h_C, int M, int N, double scale) {
    int numElements = M * N;
    size_t size = numElements * sizeof(double);

    double *d_A, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_1d(numElements);
    scale_forward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_C, scale, numElements);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
}


void CudaOps::transpose(const double* h_A, double* h_C, int M, int N) {
    size_t size_A = M * N * sizeof(double);
    size_t size_C = N * M * sizeof(double);

    double *d_A, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_2d(M, N);
    transpose_forward_kernel<<<cfg.blocks, cfg.threads>>>(d_A, d_C, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
}


void CudaOps::transpose_backward(const double* h_dOut, double* h_dX, int M, int N) {
    size_t size = M * N * sizeof(double);

    double *d_dOut, *d_dX;
    cudaMalloc(&d_dOut, size);
    cudaMalloc(&d_dX, size);

    cudaMemcpy(d_dOut, h_dOut, size, cudaMemcpyHostToDevice);

    LaunchConfig cfg = make_launch_2d(M, N);
    transpose_backward_kernel<<<cfg.blocks, cfg.threads>>>(d_dOut, d_dX, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dX, d_dX, size, cudaMemcpyDeviceToHost);

    cudaFree(d_dOut);
    cudaFree(d_dX);
}
