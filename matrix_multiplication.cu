#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>

#define THREADS_PER_BLOCK 16

// Kernel pentru multiplicarea matricilor pe GPU
__global__ void matrixMultiplicationGPU(float *a, float *b, float *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

// Multiplicarea matricelor pe CPU cu OpenMP
void matrixMultiplicationCPU(float *a, float *b, float *c, int width) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0;
            for (int k = 0; k < width; ++k) {
                sum += a[i * width + k] * b[k * width + j];
            }
            c[i * width + j] = sum;
        }
    }
}

// Funcție pentru inițializarea matricelor
void initializeMatrix(float *matrix, int width, float value = 1.0f) {
    for (int i = 0; i < width * width; ++i) {
        matrix[i] = value;
    }
}

// Verificarea rezultatelor
bool verifyResults(float *cpuResult, float *gpuResult, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(cpuResult[i] - gpuResult[i]) > 1e-4) {
            return false;
        }
    }
    return true;
}

int main() {
    const int sizes[] = {100, 1000}; // Dimensiunile matricelor
    for (int size : sizes) {
        int width = size;
        int numElements = width * width;

        // Alocare memorie pe CPU
        float *a = new float[numElements];
        float *b = new float[numElements];
        float *c_cpu = new float[numElements];
        float *c_gpu = new float[numElements];

        // Inițializarea matricelor
        initializeMatrix(a, width);
        initializeMatrix(b, width);

        // Calcul pe CPU
        auto cpuStart = std::chrono::high_resolution_clock::now();
        int numThreadsCPU = omp_get_max_threads();
        std::cout << "Total threads used on CPU: " << numThreadsCPU << std::endl;

        matrixMultiplicationCPU(a, b, c_cpu, width);

        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpuDuration = cpuEnd - cpuStart;
        std::cout << "CPU execution time for " << width << "x" << width << " matrix: " 
                  << cpuDuration.count() << " seconds" << std::endl;

        // Alocare memorie pe GPU
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, numElements * sizeof(float));
        cudaMalloc(&d_b, numElements * sizeof(float));
        cudaMalloc(&d_c, numElements * sizeof(float));

        cudaMemcpy(d_a, a, numElements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, numElements * sizeof(float), cudaMemcpyHostToDevice);

        // Configurare grilă și blocuri
        dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        dim3 numBlocks((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                       (width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

        int totalThreadsGPU = numBlocks.x * numBlocks.y * THREADS_PER_BLOCK * THREADS_PER_BLOCK;
        std::cout << "Total threads used on GPU: " << totalThreadsGPU << std::endl;

        // Calcul pe GPU
        auto gpuStart = std::chrono::high_resolution_clock::now();

        matrixMultiplicationGPU<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, width);
        cudaDeviceSynchronize();

        auto gpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gpuDuration = gpuEnd - gpuStart;

        cudaMemcpy(c_gpu, d_c, numElements * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "GPU execution time for " << width << "x" << width << " matrix: " 
                  << gpuDuration.count() << " seconds" << std::endl;

        // Verificare rezultate
        if (verifyResults(c_cpu, c_gpu, numElements)) {
            std::cout << "Results are correct!" << std::endl;
        } else {
            std::cout << "Results are incorrect!" << std::endl;
        }

        // Curățare memorie
        delete[] a;
        delete[] b;
        delete[] c_cpu;
        delete[] c_gpu;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        std::cout << std::endl;
    }

    return 0;
}



// Compile and run
// nvcc matrix_multiplication.cu -o matrix_multiplication
// ./matrix_multiplication