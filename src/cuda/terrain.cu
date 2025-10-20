#include <cstdio>
#include <cuda_runtime.h>

// --- Parametri globali ---
constexpr int WIDTH  = 512;
constexpr int HEIGHT = 512;

// --- Kernel CUDA ---
// Ogni thread genera un punto dell'heightmap a partire dal seed.
__global__ void generateTerrain(float* heightmap, int width, int height, unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Esempio: funzione fittizia per mostrare struttura
    unsigned int idx = y * width + x;
    float fx = static_cast<float>(x) / width;
    float fy = static_cast<float>(y) / height;

    // Formula arbitraria (solo dimostrativa)
    float value = sinf(fx * 11.0f + seed * 0.01f) * cosf(fy * 10.0f + seed * 0.02f);
    heightmap[idx] = value;
}

// --- Funzione principale ---
int main()
{
    // 1. Allocazione memoria GPU
    float* d_heightmap = nullptr;
    size_t size = WIDTH * HEIGHT * sizeof(float);
    cudaMalloc(&d_heightmap, size);

    // 2. Configurazione griglia e blocchi
    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
                 (HEIGHT + blockDim.y - 1) / blockDim.y);

    unsigned int seed = 1234;

    // 3. Lancio kernel
    generateTerrain<<<gridDim, blockDim>>>(d_heightmap, WIDTH, HEIGHT, seed);
    cudaDeviceSynchronize();

    // 4. Copia risultati su CPU
    float* h_heightmap = new float[WIDTH * HEIGHT];
    cudaMemcpy(h_heightmap, d_heightmap, size, cudaMemcpyDeviceToHost);

    // 5. Verifica semplice di alcuni valori
    printf("Heightmap sample values:\n");
    for (int i = 0; i < 5; ++i)
        printf("%d: %.4f\n", i, h_heightmap[i]);

    // 6. Cleanup
    // Dopo cudaMemcpy, salva i dati
    FILE* f = fopen("heightmap.bin", "wb");
    fwrite(h_heightmap, sizeof(float), WIDTH * HEIGHT, f);
    fclose(f);
    
    // Free memory
    delete[] h_heightmap;
    cudaFree(d_heightmap);

    return 0;
}

