#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define N 256 // Use for matrix multiplication and array sizes for sorting
#define TILE_WIDTH 16 // For matrix multiplication kernel

// Kernel for Matrix Multiplication
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (col < width && row < width) {
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Kernel for Bitonic Sort Step
__global__ void bitonicSortStep(int* dev_values, int j, int k) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    if ((ixj) > i) {
        if ((i & k) == 0) {
            if (dev_values[i] > dev_values[ixj]) {
                // Swap
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i & k) != 0) {
            if (dev_values[i] < dev_values[ixj]) {
                // Swap
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

// Kernel for Odd-Even Sort Step
__global__ void oddEvenSortStep(int* dev_values, int n, int phase) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int i = 2 * index + phase;

    if (i + 1 < n) {
        if (dev_values[i] > dev_values[i + 1]) {
            // Swap
            int temp = dev_values[i];
            dev_values[i] = dev_values[i + 1];
            dev_values[i + 1] = temp;
        }
    }
}

// Host function to run matrix multiplication
void matrixMultiply() {
    float* A, * B, * C;
    float* d_A, * d_B, * d_C;
    int size = N * N * sizeof(float);

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    matrixMultiplyKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // For demonstration: print the first element of the result matrix
    std::cout << "Matrix Multiplication Result [0,0]: " << C[0] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
}

// Host function to run bitonic sort
void bitonicSort() {
    int* values = new int[N];
    int* d_values;
    size_t size = N * sizeof(int);

    // Initialize array with random values
    for (int i = 0; i < N; i++) {
        values[i] = rand() % N;
    }

    cudaMalloc(&d_values, size);
    cudaMemcpy(d_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks((N + 255) / 256);
    dim3 threads(256);

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep << <blocks, threads >> > (d_values, j, k);
        }
    }

    cudaMemcpy(values, d_values, size, cudaMemcpyDeviceToHost);

    // For demonstration: print the first element of the sorted array
    std::cout << "Bitonic Sort Result [0]: " << values[0] << std::endl;

    cudaFree(d_values);
    delete[] values;
}

// Host function to run odd-even sort
void oddEvenSort() {
    int* values = new int[N];
    int* d_values;
    size_t size = N * sizeof(int);

    // Initialize array with random values
    for (int i = 0; i < N; i++) {
        values[i] = rand() % N;
    }

    cudaMalloc(&d_values, size);
    cudaMemcpy(d_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks((N + 255) / 256);
    dim3 threads(256);

    for (int i = 0; i < N; i++) {
        oddEvenSortStep << <blocks, threads >> > (d_values, N, i % 2);
    }

    cudaMemcpy(values, d_values, size, cudaMemcpyDeviceToHost);

    // For demonstration: print the first element of the sorted array
    std::cout << "Odd-Even Sort Result [0]: " << values[0] << std::endl;

    cudaFree(d_values);
    delete[] values;
}

int main() {
    srand(time(NULL)); // Seed for random number generation

    std::cout << "Starting matrix multiplication..." << std::endl;
    matrixMultiply();

    std::cout << "\nStarting bitonic sort..." << std::endl;
    bitonicSort();

    std::cout << "\nStarting odd-even sort..." << std::endl;
    oddEvenSort();

    return 0;
}
