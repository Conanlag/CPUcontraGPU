# Comparación CPU y GPU

Este proyecto compara la suma elemento a elemento de dos vectores usando CPU
(OpenMP) y GPU (CUDA). Ambos programas inicializan los vectores de la misma
forma y muestran el tiempo medido y un checksum del resultado.

## Requisitos

- Un compilador de C++ con soporte para OpenMP, como `g++`.
- NVIDIA CUDA Toolkit para compilar y ejecutar la versión GPU.
- Una GPU NVIDIA compatible con CUDA para ejecutar la versión GPU.

## Compilar

Desde el directorio del proyecto:

```bash
g++ -O2 -fopenmp cpu.cpp -o cpu
nvcc -O2 gpu.cu -o gpu
```

## Ejecutar

El tamaño predeterminado es de 10 000 000 elementos. Se puede indicar otro
tamaño positivo como argumento:

```bash
./cpu
./gpu

```
o eligiendo una cantidad personalizada
``` bash
./cpu 100000000
./gpu 100000000
```

La versión CPU mide el tiempo del bucle de suma. La versión GPU informa el
tiempo del kernel CUDA, sin incluir las transferencias de datos entre la CPU y
la GPU. Ambas muestran un checksum para comparar los resultados.
