
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "Error CUDA: " \
                  << cudaGetErrorString(err) << '\n'; \
        return 1; \
    } \
} while (0)

__global__ void sumar(const float* A, const float* B,
                      float* C, long long N) {
    long long i = (long long)blockIdx.x * blockDim.x
                + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]) {
    long long N = (argc > 1) ? atoll(argv[1]) : 100000000;

    if (N <= 0) {
        std::cerr << "El tamaño debe ser positivo.\n";
        return 1;
    }

    std::vector<float> A(N), B(N), C(N);

    for (long long i = 0; i < N; i++) {
        A[i] = static_cast<float>(i) * 0.5f;
        B[i] = static_cast<float>(i) * 0.25f;
    }

    float *d_A, *d_B, *d_C;

    auto inicioTotal = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    int hilos = 256;
    int bloques = static_cast<int>((N + hilos - 1) / hilos);

    cudaEvent_t inicioKernel, finKernel;
    CUDA_CHECK(cudaEventCreate(&inicioKernel));
    CUDA_CHECK(cudaEventCreate(&finKernel));

    CUDA_CHECK(cudaEventRecord(inicioKernel));

    sumar<<<bloques, hilos>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(finKernel));
    CUDA_CHECK(cudaEventSynchronize(finKernel));

    float tiempoKernel = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &tiempoKernel, inicioKernel, finKernel));

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    auto finTotal = std::chrono::steady_clock::now();

    double tiempoTotal =
        std::chrono::duration<double, std::milli>(
            finTotal - inicioTotal).count();

    double suma = 0.0;
    for (long long i = 0; i < N; i++) {
        suma += C[i];
    }

    cudaDeviceProp propiedades;
    int dispositivo;
    CUDA_CHECK(cudaGetDevice(&dispositivo));
    CUDA_CHECK(cudaGetDeviceProperties(
        &propiedades, dispositivo));

    std::cout << "GPU: " << propiedades.name << '\n';
    std::cout << "Elementos: " << N << '\n';
    std::cout << "Hilos CUDA por bloque: " << hilos << '\n';
    std::cout << "Bloques CUDA: " << bloques << '\n';
    std::cout << "Tiempo: " << tiempoKernel << " ms\n";
    std::cout << "Checksum: " << suma << '\n';

    cudaEventDestroy(inicioKernel);
    cudaEventDestroy(finKernel);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
