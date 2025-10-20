#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define WIDTH  512
#define HEIGHT 512

int main() {
    float* heightmap = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    if (!heightmap) {
        fprintf(stderr, "Errore allocazione memoria\n");
        return 1;
    }

    unsigned int seed = 1234;

    // --- Misura tempo inizio ---
    clock_t start = clock();

    // --- Generazione heightmap seriale ---
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            float fx = (float)x / WIDTH;
            float fy = (float)y / HEIGHT;
            float value = sinf(fx * 11.0f + seed * 0.01f) * cosf(fy * 10.0f + seed * 0.02f);
            heightmap[y * WIDTH + x] = value;
        }
    }

    // --- Misura tempo fine ---
    clock_t end = clock();
    double elapsed_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo generazione seriale: %.3f ms\n", elapsed_ms);

    // --- Salvataggio su file ---
    FILE* f = fopen("heightmap_serial.bin", "wb");
    if (!f) {
        fprintf(stderr, "Errore apertura file\n");
        free(heightmap);
        return 1;
    }
    fwrite(heightmap, sizeof(float), WIDTH * HEIGHT, f);
    fclose(f);

    // --- Verifica sample ---
    printf("Heightmap sample values:\n");
    for (int i = 0; i < 5; ++i)
        printf("%d: %.4f\n", i, heightmap[i]);

    free(heightmap);
    return 0;
}

