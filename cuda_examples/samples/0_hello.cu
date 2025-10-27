// Copyright (c) 2025 Author. All Rights Reserved.
// Davide Chirichella

#include <stdio.h>
#include <cuda_runtime.h>

// Global significa che può essere richiamata dall'host
// ed eseguita sul device
__global__ void helloCUDA() {

  // Indice locale del thread nel blocco
  int thread_local_idx = threadIdx.x;
  // Indice del blocco nella griglia
  int block_index = blockIdx.x;
  // Dimensione del blocco
  int block_size = blockDim.x;
  // Indice globale del thread nella griglia
  int thread_global_idx = thread_local_idx + block_index * block_size;

  printf("Hello from GPU! Block %d, Thread %d (local %d)\n",
         block_index, thread_global_idx, thread_local_idx);
}

int main() {

  /* Stampa info sulla GPU */
  // ottiene proprità sul device "0"
  printf("\n******************************************************\n\n"); 

  cudaDeviceProp prop;
  
  cudaGetDeviceProperties(&prop, 0); // <- device: 0 
  printf("Nome Dispositivo: %s\n", prop.name);
  printf("Memoria Globale Totale: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
  int clockKhz;
  cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, 0);
  printf("Clock Core: %.2f MHz\n", clockKhz / 1000.0f);

  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("\n******************************************************\n\n"); 

  /* Inizio programma */ 
  // la CPU richiama questo kernel
  // Bisogna definire quanti thread eseguiranno la specifica
  // funzione. Esiste una gerarchia di thread suddivisa in
  // griglie e blocchi
  helloCUDA<<<2,50>>>();
  
  // Si attende che la GPU finisca
  cudaDeviceSynchronize();


  /* FIne programma */
  return 0;
}

