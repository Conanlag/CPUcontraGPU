
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[]) {
    long long N = (argc > 1) ? atoll(argv[1]) : 100000000;

    std::vector<float> A(N), B(N), C(N);

    for (long long i = 0; i < N; i++) {
        A[i] = static_cast<float>(i) * 0.5f;
        B[i] = static_cast<float>(i) * 0.25f;
    }

    auto inicio = std::chrono::steady_clock::now();

    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    auto fin = std::chrono::steady_clock::now();

    double tiempo = std::chrono::duration<double, std::milli>(
        fin - inicio).count();

    double suma = 0.0;
    for (long long i = 0; i < N; i++) {
        suma += C[i];
    }

    std::cout << "Procesador: AMD Ryzen 7 7735HS \n";
    std::cout << "Elementos: " << N << '\n';
    std::cout << "Hilos CPU: " << omp_get_max_threads() << '\n';
    std::cout << "Tiempo: " << tiempo << " ms\n";
    std::cout << "Checksum: " << suma << '\n';

    return 0;
}
