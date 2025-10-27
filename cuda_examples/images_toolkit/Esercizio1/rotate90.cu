#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <time.h>

// STB image libraries (single-file includes)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Macro di controllo errori CUDA (termina immediatamente in caso di errore)
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

/*
   KERNEL CUDA: rotazione 90°
   - unica implementazione per CW (direction=1) e CCW (direction=-1)
   - ogni thread mappa un pixel di input (x = ix, y = iy)
   - la dimensione dell'output è la stessa in byte ma con dimensioni logiche scambiate:
       output_width  = height
       output_height = width
*/
__global__ void rotate90(unsigned char* d_input, unsigned char* d_output,
                         int width, int height, int channels, int direction)
{
    // Coordinate del pixel (input)
    int ix = blockIdx.x * blockDim.x + threadIdx.x;  // x nell'immagine di input [0..width-1]
    int iy = blockIdx.y * blockDim.y + threadIdx.y;  // y nell'immagine di input [0..height-1]

    if (ix >= width || iy >= height) return; // fuori range

    // Per ogni canale (RGB / RGBA / ecc.)
    for (int c = 0; c < channels; c++) {
        // indice lineare nel buffer di input (row-major)
        int input_idx = (iy * width + ix) * channels + c;

        // Calcolare dove va il pixel nell'immagine di output
        // output_width = height; output_height = width
        // Usare formule consolidate:
        //  - Clockwise (90° CW): (x,y) -> (new_x, new_y) = (H - 1 - y, x)
        //    indice_output = new_y * new_width + new_x = x * height + (height - 1 - y)
        //  - CounterClockwise (90° CCW): (x,y) -> (new_x, new_y) = (y, W - 1 - x)
        //    indice_output = new_y * new_width + new_x = (width - 1 - x) * height + y
        int output_idx;
        if (direction == 1) { // CW
            output_idx = (ix * height + (height - 1 - iy)) * channels + c;
        } else { // CCW
            output_idx = ((width - 1 - ix) * height + iy) * channels + c;
        }

        d_output[output_idx] = d_input[input_idx];
    }
}

/* ============================
   Funzione CPU: versione singola
   - riproduce esattamente la stessa trasformazione del kernel CUDA
   - implementata per confronto / verifica
   ============================ */
void rotate90_cpu(unsigned char* input, unsigned char* output,
                  int width, int height, int channels, int direction)
{
    // Nota: output ha la stessa dimensione in byte (width*height*channels)
    // ma la geometria è scambiata quando interpretata come immagine.
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int input_idx = (y * width + x) * channels + c;
                int output_idx;
                if (direction == 1) { // CW
                    // (x,y) -> (new_x = H-1-y, new_y = x)
                    output_idx = (x * height + (height - 1 - y)) * channels + c;
                } else { // CCW
                    // (x,y) -> (new_x = y, new_y = W-1-x)
                    output_idx = ((width - 1 - x) * height + y) * channels + c;
                }
                output[output_idx] = input[input_idx];
            }
        }
    }
}

/* ============================
   Verifica byte-a-byte fra CPU e GPU
   ============================ */
bool verifyResults(unsigned char* cpu_result, unsigned char* gpu_result, int size)
{
    for (int i = 0; i < size; i++) {
        if (cpu_result[i] != gpu_result[i]) {
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n",
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

/* ============================
   main: parsing, allocazioni, lancio kernel, verifica, cleanup
   ============================ */
int main(int argc, char **argv)
{
    if (argc < 4) {
        printf("Usage: %s <input_image> <block_size> <direction>\n", argv[0]);
        printf("  direction: cw (clockwise) or ccw (counterclockwise)\n");
        printf("Example: %s input.png 16 cw\n", argv[0]);
        return 1;
    }

    const char* inputFile = argv[1];
    int blockSize = atoi(argv[2]);
    const char* dirStr = argv[3];
    const int blockSize_x = blockSize;
    const int blockSize_y = blockSize;

    int direction;
    if (strcmp(dirStr, "cw") == 0) {
        direction = 1;
        printf("Rotazione: senso orario (cw)\n");
    } else if (strcmp(dirStr, "ccw") == 0) {
        direction = -1;
        printf("Rotazione: senso antiorario (ccw)\n");
    } else {
        printf("Errore: direzione non valida. Usare 'cw' o 'ccw'\n");
        return 1;
    }

    // == Caricamento immagine (host) ==
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 0);
    if (!h_input) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    // Dimensione in byte (host & device): width * height * channels
    // NB: la geometria dell'immagine ruotata è (height x width) ma i byte totali restano identici.
    int output_size = width * height * channels;

    // == Allocazioni host per output ==
    unsigned char* h_output_cpu = (unsigned char*)malloc(output_size);
    unsigned char* h_output_gpu = (unsigned char*)malloc(output_size);
    if (!h_output_cpu || !h_output_gpu) {
        fprintf(stderr, "Errore allocazione memoria per output\n");
        stbi_image_free(h_input);
        free(h_output_cpu);
        free(h_output_gpu);
        return 1;
    }

    // == Allocazioni device ==
    unsigned char *d_input = NULL, *d_output = NULL;
    CHECK(cudaMalloc(&d_input, output_size));
    CHECK(cudaMalloc(&d_output, output_size));
    // Copia input host -> device (tutti i byte dell'immagine originale)
    CHECK(cudaMemcpy(d_input, h_input, output_size, cudaMemcpyHostToDevice));

    /* ============================
       Configurazione kernel: grid e block
       - ogni thread elabora esattamente un pixel di input
       - blockDim = (blockSize_x, blockSize_y)
       - gridDim calcolata per coprire tutta l'immagine di input
       ============================ */
    dim3 block(blockSize_x, blockSize_y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Esecuzione kernel GPU
    printf("\nEsecuzione GPU...\n");
    clock_t gpu_start = clock();
    // Lancio kernel
    rotate90<<<grid, block>>>(d_input, d_output, width, height, channels, direction);
    clock_t gpu_end = clock();
    double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo GPU: %.3f ms\n", gpu_time);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copia risultato da device a host
    CHECK(cudaMemcpy(h_output_gpu, d_output, output_size, cudaMemcpyDeviceToHost));

    /* ============================
       Esecuzione CPU per confronto
       ============================ */
    printf("\nEsecuzione CPU...\n");
    clock_t cpu_start = clock();
    rotate90_cpu(h_input, h_output_cpu, width, height, channels, direction);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo CPU: %.3f ms\n", cpu_time);

    /* ============================
       Verifica correttezza (byte-by-byte)
       ============================ */
    printf("\nVerifica correttezza...\n");
    bool correct = verifyResults(h_output_cpu, h_output_gpu, output_size);
    printf(correct ? "✓ Test PASSATO\n" : "✗ Test FALLITO\n");

    /* ============================
       Salvataggio immagini risultanti
       - Attenzione: la geometria dell'output è (height x width)
         perche' l'immagine e' ruotata: quindi passiamo width/height invertiti
       - stride = output_width * channels = height * channels
       ============================ */
    int out_width = height;
    int out_height = width;
    int out_stride = out_width * channels;

    stbi_write_png("output_gpu.png", out_width, out_height, channels, h_output_gpu, out_stride);
    stbi_write_png("output_cpu.png", out_width, out_height, channels, h_output_cpu, out_stride);
    printf("\nImmagini salvate: output_gpu.png, output_cpu.png\n");

    /* ============================
       Cleanup risorse
       ============================ */
    stbi_image_free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    return 0;
}
