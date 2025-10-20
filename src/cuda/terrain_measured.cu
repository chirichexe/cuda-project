#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

// --- Parametri globali ---
constexpr int WIDTH  = 512;
constexpr int HEIGHT = 512;

// --- Kernel CUDA ---
__global__ void generateTerrain(float* heightmap, int width, int height, unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    unsigned int idx = y * width + x;
    float fx = static_cast<float>(x) / width;
    float fy = static_cast<float>(y) / height;

    float value = sinf(fx * 50.0f + seed * 0.01f) * cosf(fy * 10.0f + seed * 0.02f);
    heightmap[idx] = value;
}

// --- Funzione principale ---
int main()
{
    float* d_heightmap = nullptr;
    size_t size = WIDTH * HEIGHT * sizeof(float);
    cudaMalloc(&d_heightmap, size);

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
                 (HEIGHT + blockDim.y - 1) / blockDim.y);

    unsigned int seed = 1234;

    // --- Eventi per misurazione tempo GPU ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // inizio misurazione

    // --- Lancio kernel ---
    generateTerrain<<<gridDim, blockDim>>>(d_heightmap, WIDTH, HEIGHT, seed);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);  // fine misurazione
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tempo esecuzione kernel GPU: %.3f ms\n", milliseconds);

    // --- Copia risultati su CPU ---
    float* h_heightmap = new float[WIDTH * HEIGHT];
    cudaMemcpy(h_heightmap, d_heightmap, size, cudaMemcpyDeviceToHost);

    printf("Heightmap sample values:\n");
    for (int i = 0; i < 5; ++i)
        printf("%d: %.4f\n", i, h_heightmap[i]);

    // --- Salvataggio su file ---
    FILE* f = fopen("heightmap.bin", "wb");
    fwrite(h_heightmap, sizeof(float), WIDTH * HEIGHT, f);
    fclose(f);

    delete[] h_heightmap;
    cudaFree(d_heightmap);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

