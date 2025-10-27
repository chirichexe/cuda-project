// Copyright (c) 2025 Author. All Rights Reserved.
// Davide Chirichella

/*
================================================================================
Funzioni CUDA utilizzate e come funzionano
================================================================================

1. cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
   - Ottiene le proprietà del device GPU specificato.
   - Riempie la struttura 'prop' con informazioni come nome, memoria globale,
     numero di core, compute capability, ecc.

2. cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device)
   - Ottiene un attributo specifico del device (es. clock rate).
   - Permette di leggere parametri hardware specifici della GPU.

3. cudaMalloc(void **devPtr, size_t size)
   - Alloca memoria sulla GPU di dimensione 'size' bytes.
   - Restituisce un puntatore device (devPtr) che si usa nei kernel.

4. cudaFree(void *devPtr)
   - Libera la memoria precedentemente allocata con cudaMalloc.

5. cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
   - Copia dati tra host e device.
   - kind può essere:
     - cudaMemcpyHostToDevice: copia dati dall’host alla GPU
     - cudaMemcpyDeviceToHost: copia dati dalla GPU all’host
     - cudaMemcpyDeviceToDevice: copia dati tra due puntatori device

6. __global__ 
   - Specifica che la funzione è un kernel CUDA, eseguibile sulla GPU.
   - Può essere chiamata solo dal host (CPU) con la sintassi <<<blocks, threads>>>.

7. <<<blocksPerGrid, threadsPerBlock>>>
   - Sintassi per il lancio del kernel CUDA.
   - blocksPerGrid = numero di blocchi nella griglia
   - threadsPerBlock = numero di thread per blocco

8. blockIdx, threadIdx, blockDim, gridDim
   - Variabili built-in che permettono di calcolare l’indice globale del thread.
   - blockIdx = indice del blocco nella griglia
   - threadIdx = indice del thread all’interno del blocco
   - blockDim = dimensione del blocco
   - gridDim = dimensione della griglia

9. dim3
   - Tipo di dato CUDA per gestire dimensioni di blocchi e griglie in 2D/3D.
   - Es. dim3 threadsPerBlock(8,8) definisce un blocco 2D di 8x8 thread.

================================================================================
*/


#include <stdio.h>
#include <cuda_runtime.h>

// Kernel CUDA per somma di array 1D
__global__ void sum1D(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // indice globale
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel CUDA per somma di array 2D
__global__ void sum2D(float *a, float *b, float *c, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x; // indice lineare
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel CUDA per somma di array 3D
__global__ void sum3D(float *a, float *b, float *c, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = z * width * height + y * width + x; // indice lineare 3D
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("\n******************************************************\n\n"); 

    // Informazioni sulla GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    printf("Nome Dispositivo: %s\n", prop.name);
    printf("Memoria Globale Totale: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
    int clockKhz;
    cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, 0);
    printf("Clock Core: %.2f MHz\n", clockKhz / 1000.0f);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("\n******************************************************\n\n"); 
    
    /*
     * WARNING!!
     *
     * Il numero massimo totale di thread per blocco è 1024 per la maggior parte delle GPU (compute capability >= 2.x)
     * Un blocco può essere organizzato in 1, 2 o 3 dimensioni, ma ci sono limiti per ciascuna dimensione. Esempio:
     * x: 1024 , y: 1024, z: 64 
     * Il prodotto delle dimensioni x, y e z non può superare 1024 (queste limitazioni potrebbero cambiare in futuro)
     *
     * */

    /* ======== Esempio somma array 1D ======== */
    int N = 1024;
    size_t size1D = N * sizeof(float);

    // Allocazione memoria host
    float *h_a = (float*)malloc(size1D);
    float *h_b = (float*)malloc(size1D);
    float *h_c = (float*)malloc(size1D);

    // Inizializzazione array host
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Allocazione memoria device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size1D);
    cudaMalloc((void**)&d_b, size1D);
    cudaMalloc((void**)&d_c, size1D);

    // Copia dati host -> device
    cudaMemcpy(d_a, h_a, size1D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size1D, cudaMemcpyHostToDevice);

    // Lancio kernel 1D
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sum1D<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copia dati device -> host
    cudaMemcpy(h_c, d_c, size1D, cudaMemcpyDeviceToHost);

    printf("Somma 1D: c[0]=%.1f, c[N-1]=%.1f\n", h_c[0], h_c[N-1]);

    // Liberazione memoria device e host
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    /* ======== Esempio somma array 2D ======== */
    int width = 16, height = 16;
    size_t size2D = width * height * sizeof(float);

    float *h_a2 = (float*)malloc(size2D);
    float *h_b2 = (float*)malloc(size2D);
    float *h_c2 = (float*)malloc(size2D);

    for (int i = 0; i < width * height; i++) {
        h_a2[i] = i;
        h_b2[i] = i * 2;
    }

    float *d_a2, *d_b2, *d_c2;
    cudaMalloc((void**)&d_a2, size2D);
    cudaMalloc((void**)&d_b2, size2D);
    cudaMalloc((void**)&d_c2, size2D);

    cudaMemcpy(d_a2, h_a2, size2D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, size2D, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock2D(8, 8);
    dim3 blocksPerGrid2D((width + 7) / 8, (height + 7) / 8);
    sum2D<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_a2, d_b2, d_c2, width, height);

    cudaMemcpy(h_c2, d_c2, size2D, cudaMemcpyDeviceToHost);
    printf("Somma 2D: c[0]=%.1f, c[last]=%.1f\n", h_c2[0], h_c2[width*height-1]);

    cudaFree(d_a2); cudaFree(d_b2); cudaFree(d_c2);
    free(h_a2); free(h_b2); free(h_c2);

    /* ======== Esempio somma array 3D ======== */
    int depth = 4;
    size_t size3D = width * height * depth * sizeof(float);

    float *h_a3 = (float*)malloc(size3D);
    float *h_b3 = (float*)malloc(size3D);
    float *h_c3 = (float*)malloc(size3D);

    for (int i = 0; i < width * height * depth; i++) {
        h_a3[i] = i;
        h_b3[i] = i * 2;
    }

    float *d_a3, *d_b3, *d_c3;
    cudaMalloc((void**)&d_a3, size3D);
    cudaMalloc((void**)&d_b3, size3D);
    cudaMalloc((void**)&d_c3, size3D);

    cudaMemcpy(d_a3, h_a3, size3D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, size3D, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock3D(4, 4, 4);
    dim3 blocksPerGrid3D((width+3)/4, (height+3)/4, (depth+3)/4);
    sum3D<<<blocksPerGrid3D, threadsPerBlock3D>>>(d_a3, d_b3, d_c3, width, height, depth);

    cudaMemcpy(h_c3, d_c3, size3D, cudaMemcpyDeviceToHost);
    printf("Somma 3D: c[0]=%.1f, c[last]=%.1f\n", h_c3[0], h_c3[width*height*depth-1]);

    cudaFree(d_a3); cudaFree(d_b3); cudaFree(d_c3);
    free(h_a3); free(h_b3); free(h_c3);

    printf("\nProgramma terminato correttamente.\n");
    return 0;
}

