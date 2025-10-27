#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

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

// Kernel grayscale con loop sui frame (2D grid)
// Ogni thread processa un pixel in TUTTI i frame
__global__ void grayscale_loop(unsigned char* d_input, unsigned char* d_output,
                                int width, int height, int num_frames)
{
    
    // TODO

}

// Kernel grayscale con griglia 3D
// Ogni thread processa UN pixel in UN frame
__global__ void grayscale_3D(unsigned char* d_input, unsigned char* d_output,
                              int width, int height, int num_frames)
{
    
    // TODO

}

// Versione CPU per confronto
void grayscale_cpu(unsigned char* input, unsigned char* output,
                   int width, int height, int num_frames)
{
    
    // TODO

}

// Funzione per verificare correttezza
bool verifyResults(unsigned char* cpu_result, unsigned char* gpu_result, 
                   int size, const char* label)
{
    int errors = 0;
    for (int i = 0; i < size; i++) {
        // Tolleriamo differenze di ±1 dovute ad arrotondamenti
        int diff = abs((int)cpu_result[i] - (int)gpu_result[i]);
        if (diff > 1) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at index %d: CPU=%d, %s=%d (diff=%d)\n", 
                       i, cpu_result[i], label, gpu_result[i], diff);
            }
        }
    }
    
    if (errors > 0) {
        printf("Total errors: %d / %d (%.2f%%)\n", 
               errors, size, 100.0f * errors / size);
    }
    
    return errors == 0;
}

// Funzione per caricare "video" (sequenza di immagini)
unsigned char* loadVideoFrames(const char* pattern, int* width, int* height, 
                               int* num_frames, int max_frames)
{
    unsigned char** frames = (unsigned char**)malloc(max_frames * sizeof(unsigned char*));
    int channels;
    int frame_count = 0;
    
    printf("Caricamento frame:\n");
    for (int i = 0; i < max_frames; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), pattern, i);
        
        int w, h, c;
        unsigned char* frame = stbi_load(filename, &w, &h, &c, 3);  // Forza RGB
        
        if (!frame) {
            if (i == 0) {
                printf("Errore: impossibile caricare il primo frame %s\n", filename);
                free(frames);
                return NULL;
            }
            break;  // Fine dei frame
        }
        
        // Verifica dimensioni consistenti
        if (i == 0) {
            *width = w;
            *height = h;
        } else {
            if (w != *width || h != *height) {
                printf("Errore: frame %d ha dimensioni diverse (%dx%d vs %dx%d)\n", 
                       i, w, h, *width, *height);
                stbi_image_free(frame);
                break;
            }
        }
        
        frames[frame_count++] = frame;
        printf("  Frame %d: %dx%d\n", i, w, h);
    }
    
    if (frame_count == 0) {
        free(frames);
        return NULL;
    }
    
    *num_frames = frame_count;
    
    // Crea buffer contiguo per tutti i frame
    size_t frame_size = (*width) * (*height) * 3;
    size_t total_size = frame_size * frame_count;
    unsigned char* video = (unsigned char*)malloc(total_size);
    
    for (int i = 0; i < frame_count; i++) {
        memcpy(video + i * frame_size, frames[i], frame_size);
        stbi_image_free(frames[i]);
    }
    
    free(frames);
    return video;
}

int main(int argc, char **argv) 
{
    if (argc < 4) {
        printf("Usage: %s <frame_pattern> <block_size> <method> [max_frames]\n", argv[0]);
        printf("Method: loop (2D grid + loop) or 3d (3D grid)\n");
        printf("Max_frames: numero massimo di frame da processare (default: 1000)\n");
        printf("Example: %s \"../Videos/frame_%%03d.jpg\" 16 loop 100\n", argv[0]);
        printf("         %s \"../Videos/frame_%%03d.jpg\" 16 3d 50\n", argv[0]);
        return 1;
    }

    const char* framePattern = argv[1];
    int blockSize = atoi(argv[2]);
    const char* method = argv[3];
    int max_frames = (argc >= 5) ? atoi(argv[4]) : 1000;
    
    bool use_3d = (strcmp(method, "3d") == 0);

    // ========== Caricamento video ==========
    int width, height, num_frames;
    unsigned char* h_input = loadVideoFrames(framePattern, &width, &height, 
                                             &num_frames, max_frames);
    if (!h_input) {
        printf("Error loading video frames\n");
        return 1;
    }
    
    printf("\nVideo caricato: %d frame di %dx%d\n", num_frames, width, height);
    printf("Metodo: %s\n", use_3d ? "Griglia 3D" : "Loop 2D");

    // ========== Allocazione memoria ==========
        
    // TODO

    // ========== Configurazione kernel ==========
    if (use_3d) {
 
        // TODO
        
    } else {

        // TODO

    }

    // ========== Esecuzione CPU ==========
    printf("\nEsecuzione CPU...\n");
    
    clock_t cpu_start = clock();
    grayscale_cpu(h_input, h_output_cpu, width, height, num_frames);
    clock_t cpu_end = clock();
    
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo CPU: %.3f ms\n", cpu_time);

    // ========== Verifica correttezza ==========
    printf("\nVerifica correttezza...\n");
    bool correct = verifyResults(h_output_cpu, h_output_gpu, output_size, "GPU");
    
    if (correct) {
        printf("✓ Test PASSATO: GPU e CPU producono lo stesso risultato\n");
    } else {
        printf("✗ Test FALLITO: GPU e CPU producono risultati diversi\n");
    }

    // ========== Salvataggio frame output ==========
    printf("\nSalvataggio frame output...\n");
    
    // Crea directory output (ignora errore se esiste già)
    #ifdef _WIN32
        system("mkdir output_frames 2>nul");
    #else
        system("mkdir -p output_frames 2>/dev/null");
    #endif
    
    // Salva alcuni frame di esempio
    int frames_to_save = (num_frames < 10) ? num_frames : 10;
    for (int i = 0; i < frames_to_save; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "output_frames/frame_%03d_gpu.png", i);
        
        unsigned char* frame_data = h_output_gpu + i * width * height;
        stbi_write_png(filename, width, height, 1, frame_data, width);
        
        if (i == 0) {
            printf("  Salvato frame %d", i);
        } else if (i == frames_to_save - 1) {
            printf(", ..., %d\n", i);
        }
    }
    
    printf("Frame salvati in: output_frames/\n");
    printf("Total frames processed: %d\n", num_frames);
    printf("Compression ratio: %.2fx (RGB -> Grayscale)\n", 
           (float)input_size / output_size);

    // ========== Cleanup ==========

        
    // TODO


    return 0;
}
