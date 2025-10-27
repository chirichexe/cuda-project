/*
 * cuda_runner.c - run a cuda backend
 *
 */

/*
 * Copyright 2025 Davide Chirichella, Filippo Giulietti
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "options.h"
#include <stdio.h>
#include <inttypes.h>

#include "cuda_kernel.h"

/**
 * @brief Run the CUDA backend
 * @param opts Program options
 * @return 0 on success, error code on failure
 */
int run_on_cuda(const ProgramOptions *opts) {
    printf("\n\n ---------- RUNNING CUDA CODE -----------\n");
    
    if (opts->verbose) {
        printf("CUDA Configuration:\n");
        printf("  Image size: %d x %d\n", opts->width, opts->height);
        printf("  Octaves: %d\n", opts->octaves);
        printf("  Seed: %" PRIu64 "\n", opts->seed);
    }
    
    // Launch a simple CUDA kernel that just prints thread info
    // Using image width as number of threads for demonstration
    int num_threads = opts->width;

    return run_cuda_kernel(num_threads);

}
