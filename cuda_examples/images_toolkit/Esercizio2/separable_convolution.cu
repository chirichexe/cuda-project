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

// Helper function to clamp values
__device__ __host__ unsigned char clamp(int value) {
    return (unsigned char)(value < 0 ? 0 : (value > 255 ? 255 : value));
}

// Kernel convoluzione 2D naive (non ottimale ma corretta - vista a lezione)
__global__ void convolution2D(unsigned char* d_input, unsigned char* d_output, 
                               float* d_filter, int width, int height, 
                               int channels, int filterSize) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = filterSize / 2;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float result = 0.0f;
            
            for (int fy = 0; fy < filterSize; fy++) {
                for (int fx = 0; fx < filterSize; fx++) {
                    int imageX = x + fx - radius;
                    int imageY = y + fy - radius;
                    
                    if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                        float pixelValue = d_input[(imageY * width + imageX) * channels + c];
                        float filterValue = d_filter[fy * filterSize + fx];
                        result += pixelValue * filterValue;
                    }
                }
            }
            
            d_output[(y * width + x) * channels + c] = clamp((int)result);
        }
    }
}

// Kernel convoluzione separabile - Passata ORIZZONTALE (sulle righe)
__global__ void convolutionHorizontal(unsigned char* d_input, unsigned char* d_output,
                                       float* d_filter1D, int width, int height,
                                       int channels, int filterSize)
{
    // TODO
}

// Kernel convoluzione separabile - Passata VERTICALE (sulle colonne)
__global__ void convolutionVertical(unsigned char* d_input, unsigned char* d_output,
                                     float* d_filter1D, int width, int height,
                                     int channels, int filterSize)
{
    // TODO
}

// Versione CPU convoluzione 2D
void convolution2D_cpu(unsigned char* input, unsigned char* output,
                       float* filter, int width, int height,
                       int channels, int filterSize)
{
    int radius = filterSize / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float result = 0.0f;
                
                for (int fy = 0; fy < filterSize; fy++) {
                    for (int fx = 0; fx < filterSize; fx++) {
                        int imageX = x + fx - radius;
                        int imageY = y + fy - radius;
                        
                        if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                            float pixelValue = input[(imageY * width + imageX) * channels + c];
                            float filterValue = filter[fy * filterSize + fx];
                            result += pixelValue * filterValue;
                        }
                    }
                }
                
                output[(y * width + x) * channels + c] = clamp((int)result);
            }
        }
    }
}

// Crea filtro box (media) separabile
void createBoxFilter1D(float* filter1D, int size)
{
    float value = 1.0f / size;
    for (int i = 0; i < size; i++) {
        filter1D[i] = value;
    }
}

// Crea filtro box 2D da filtro 1D (per test naive)
void createBoxFilter2D(float* filter2D, float* filter1D, int size)
{
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            filter2D[i * size + j] = filter1D[i] * filter1D[j];
        }
    }
}

// Funzione per verificare correttezza
bool verifyResults(unsigned char* result1, unsigned char* result2, 
                   int size, const char* label1, const char* label2)
{
    int errors = 0;
    int max_diff = 0;
    
    for (int i = 0; i < size; i++) {
        int diff = abs((int)result1[i] - (int)result2[i]);
        if (diff > max_diff) max_diff = diff;
        
        // Tolleriamo differenze di ±1 dovute ad arrotondamenti
        if (diff > 1) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at index %d: %s=%d, %s=%d (diff=%d)\n", 
                       i, label1, result1[i], label2, result2[i], diff);
            }
        }
    }
    
    printf("Max difference: %d pixel\n", max_diff);
    
    if (errors > 0) {
        printf("Total errors: %d / %d (%.2f%%)\n", 
               errors, size, 100.0f * errors / size);
    }
    
    return errors == 0;
}

void printFilter(float* filter, int size, const char* name)
{
    printf("\n%s (%dx%d):\n", name, size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%7.4f ", filter[i * size + j]);
        }
        printf("\n");
    }
}

void printFilter1D(float* filter, int size, const char* name)
{
    printf("\n%s (1x%d): [ ", name, size);
    for (int i = 0; i < size; i++) {
        printf("%.4f ", filter[i]);
    }
    printf("]\n");
}

int main(int argc, char **argv) 
{
    if (argc < 4) {
        printf("Usage: %s <input_image> <filter_size> <block_size>\n", argv[0]);
        printf("Example: %s input.png 5 16\n", argv[0]);
        printf("Filter_size must be odd (3, 5, 7, ...)\n");
        return 1;
    }

    const char* inputFile = argv[1];
    int filterSize = atoi(argv[2]);
    int blockSize = atoi(argv[3]);

    if (filterSize % 2 == 0) {
        printf("Error: filter_size must be odd\n");
        return 1;
    }

    // ========== Caricamento immagine ==========
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 0);
    if (!h_input) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    // ========== Creazione filtri ==========
    int filterElements2D = filterSize * filterSize;
    
    // Filtro 1D per convoluzione separabile
    float* h_filter1D = (float*)malloc(filterSize * sizeof(float));
    createBoxFilter1D(h_filter1D, filterSize);
    printFilter1D(h_filter1D, filterSize, "Filtro 1D (box filter)");
    
    // Filtro 2D per convoluzione naive (prodotto esterno del filtro 1D)
    float* h_filter2D = (float*)malloc(filterElements2D * sizeof(float));
    createBoxFilter2D(h_filter2D, h_filter1D, filterSize);
    printFilter(h_filter2D, filterSize, "Filtro 2D equivalente");

    // ========== Allocazione memoria ==========

    // TODO

    // ========== Configurazione kernel ==========

    // TODO

    // ========== Esecuzione GPU - Convoluzione SEPARABILE ==========
    printf("\n=== Esecuzione GPU - Convoluzione Separabile (2 passate) ===\n");
    
    // Passata 1: Orizzontale (input -> temp)
    
    // TODO (attenzione alle sincronizzazioni)
    
    // Passata 2: Verticale (temp -> output)

    // TODO
    
    printf("Kernel separabile completato con successo\n");
    // NOTA: Per misurare accuratamente le performance del kernel,
    // utilizzare NVIDIA Nsight Compute con il comando:
    // ncu --set full ./separable_convolution input.png 5 16

    // ========== Esecuzione CPU ==========
    printf("\n=== Esecuzione CPU - Convoluzione 2D ===\n");
    
    clock_t cpu_start = clock();
    convolution2D_cpu(h_input, h_output_cpu, h_filter2D, width, height, 
                      channels, filterSize);
    clock_t cpu_end = clock();
    
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo CPU: %.3f ms\n", cpu_time);

    // ========== Verifica correttezza ==========
    printf("\n=== Verifica Correttezza ===\n");
    
    printf("\n1. Confronto GPU 2D vs CPU:\n");
    bool correct_2d = verifyResults(h_output_cpu, h_output_2D, imageSize, "CPU", "GPU-2D");
    
    printf("\n2. Confronto GPU Separabile vs CPU:\n");
    bool correct_sep = verifyResults(h_output_cpu, h_output_sep, imageSize, "CPU", "GPU-Sep");
    
    printf("\n3. Confronto GPU 2D vs GPU Separabile:\n");
    bool same_gpu = verifyResults(h_output_2D, h_output_sep, imageSize, "GPU-2D", "GPU-Sep");
    
    printf("\n=== Risultati Finali ===\n");
    if (correct_2d && correct_sep && same_gpu) {
        printf("✓ Test PASSATO: Tutti i metodi producono lo stesso risultato!\n");
        printf("✓ Verifica separabilità: CONFERMATA\n");
    } else {
        printf("✗ Test FALLITO: Differenze rilevate\n");
    }

    // ========== Salvataggio immagini ==========
    stbi_write_png("output_2D.png", width, height, channels, 
                   h_output_2D, width * channels);
    stbi_write_png("output_separable.png", width, height, channels, 
                   h_output_sep, width * channels);
    stbi_write_png("output_cpu.png", width, height, channels, 
                   h_output_cpu, width * channels);
    printf("\nImmagini salvate:\n");
    printf("  - output_2D.png (convoluzione 2D naive)\n");
    printf("  - output_separable.png (convoluzione separabile)\n");
    printf("  - output_cpu.png (riferimento CPU)\n");

    // ========== Cleanup ==========
    
    // TODO

    return 0;
}
