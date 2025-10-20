#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#define CHECK(call)                                                         \
    {                                                                       \
        const cudaError_t error = call;                                     \
        if (error != cudaSuccess) {                                         \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);          \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

// Kernel CUDA: converte un'immagine RGB in scala di grigi
__global__ void rgbToGrayGPU(const unsigned char *d_rgb, unsigned char *d_gray,
                             int width, int height) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;  // Coordinata X del pixel
    int iy = blockIdx.y * blockDim.y + threadIdx.y;  // Coordinata Y del pixel

    if (ix < width && iy < height) {
        int rgbOffset = (iy * width + ix) * 3;
        int grayOffset = iy * width + ix;

        unsigned char r = d_rgb[rgbOffset];
        unsigned char g = d_rgb[rgbOffset + 1];
        unsigned char b = d_rgb[rgbOffset + 2];

        // Conversione RGB → Grayscale (formula standard ITU-R BT.601)
        d_gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Conversione CPU per confronto
void rgbToGrayscaleCPU(const unsigned char *rgb, unsigned char *gray,
                       int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        unsigned char r = rgb[i * 3];
        unsigned char g = rgb[i * 3 + 1];
        unsigned char b = rgb[i * 3 + 2];
        gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <image_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("%s starting...\n", argv[0]);

    // Imposta il device CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    // Carica l'immagine con stb_image
    int width, height, channels;
    unsigned char *rgb = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!rgb) {
        fprintf(stderr, "Error loading image %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    printf("Image loaded: %dx%d, channels: %d\n", width, height, channels);

    // Alloca memoria host
    int imageSize = width * height;
    int rgbSize   = imageSize * 3;

    unsigned char *h_gray   = (unsigned char *)malloc(imageSize);
    unsigned char *cpu_gray = (unsigned char *)malloc(imageSize);

    if (!h_gray || !cpu_gray) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Conversione su CPU per confronto
    rgbToGrayscaleCPU(rgb, cpu_gray, width, height);

    // Alloca memoria device
    unsigned char *d_rgb = NULL, *d_gray = NULL;
    CHECK(cudaMalloc((void **)&d_rgb, rgbSize));
    CHECK(cudaMalloc((void **)&d_gray, imageSize));

    // Copia input su GPU
    CHECK(cudaMemcpy(d_rgb, rgb, rgbSize, cudaMemcpyHostToDevice));

    // Configura ed esegue il kernel
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rgbToGrayGPU<<<grid, block>>>(d_rgb, d_gray, width, height);
    CHECK(cudaDeviceSynchronize());

    // Copia risultato GPU su host
    CHECK(cudaMemcpy(h_gray, d_gray, imageSize, cudaMemcpyDeviceToHost));

    // Verifica risultato
    bool match = true;
    for (int i = 0; i < imageSize; ++i) {
        if (abs(cpu_gray[i] - h_gray[i]) > 1) { // tolleranza ±1
            match = false;
            printf("Mismatch at pixel %d: CPU %d, GPU %d\n",
                   i, cpu_gray[i], h_gray[i]);
            break;
        }
    }

    if (match) {
        printf("CPU and GPU results match.\n");
    }

    // Salva il risultato in scala di grigi
    if (!stbi_write_png("output_gray.png", width, height, 1, h_gray, width)) {
        fprintf(stderr, "Error writing output_gray.png\n");
    } else {
        printf("Output saved to output_gray.png\n");
    }

    // Libera la memoria
    stbi_image_free(rgb);
    free(h_gray);
    free(cpu_gray);
    CHECK(cudaFree(d_rgb));
    CHECK(cudaFree(d_gray));

    // Reset del device CUDA
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

