#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <time.h>

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Error checking macro
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Kernel downsampling 2×2 → 1
__global__ void downsample2x2(unsigned char* d_input, unsigned char* d_output,
                               int input_width, int input_height, int channels)
{
    
    // TODO

}

// Versione CPU naive per confronto
void downsample2x2_cpu(unsigned char* input, unsigned char* output,
                       int input_width, int input_height, int channels)
{
    
    // TODO

}

// Funzione per verificare correttezza
bool verifyResults(unsigned char* cpu_result, unsigned char* gpu_result, 
                   int size)
{
    int errors = 0;
    for (int i = 0; i < size; i++) {
        // Tolleriamo differenze di ±1 dovute ad arrotondamenti
        int diff = abs((int)cpu_result[i] - (int)gpu_result[i]);
        if (diff > 1) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at index %d: CPU=%d, GPU=%d (diff=%d)\n", 
                       i, cpu_result[i], gpu_result[i], diff);
            }
        }
    }
    
    if (errors > 0) {
        printf("Total errors: %d / %d (%.2f%%)\n", 
               errors, size, 100.0f * errors / size);
    }
    
    return errors == 0;
}

int main(int argc, char **argv) 
{
    if (argc < 3) {
        printf("Usage: %s <input_image> <block_size>\n", argv[0]);
        printf("Example: %s input.png 16\n", argv[0]);
        printf("Note: Input image dimensions must be even numbers\n");
        return 1;
    }

    const char* inputFile = argv[1];
    int blockSize = atoi(argv[2]);

    // ========== Caricamento immagine ==========
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 0);
    if (!h_input) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    // Verifica che le dimensioni siano pari
    if (width % 2 != 0 || height % 2 != 0) {
        printf("Warning: Image dimensions should be even for 2x downsampling\n");
        printf("Output will use dimensions: %dx%d\n", width/2, height/2);
    }

    // ========== Allocazione memoria ==========
    
    // TODO


    // ========== Configurazione kernel ==========
    
    // TODO


    // ========== Esecuzione GPU ==========
    
    // TODO


    // ========== Esecuzione CPU ==========
    printf("\nEsecuzione CPU...\n");
    
    clock_t cpu_start = clock();
    downsample2x2_cpu(h_input, h_output_cpu, width, height, channels);
    clock_t cpu_end = clock();
    
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo CPU: %.3f ms\n", cpu_time);

    // ========== Verifica correttezza ==========
    printf("\nVerifica correttezza...\n");
    bool correct = verifyResults(h_output_cpu, h_output_gpu, output_size);
    
    if (correct) {
        printf("✓ Test PASSATO: GPU e CPU producono lo stesso risultato\n");
    } else {
        printf("✗ Test FALLITO: GPU e CPU producono risultati diversi\n");
    }

    // ========== Salvataggio immagini ==========
    stbi_write_png("output_gpu.png", output_width, output_height, channels, 
                   h_output_gpu, output_width * channels);
    stbi_write_png("output_cpu.png", output_width, output_height, channels, 
                   h_output_cpu, output_width * channels);
    printf("\nImmagini salvate: output_gpu.png, output_cpu.png\n");
    printf("Dimensioni output: %dx%d (%.1f%% dell'originale)\n", 
           output_width, output_height, 
           100.0 * output_size / input_size);

    // ========== Cleanup ==========
    
    // TODO


    return 0;
}
