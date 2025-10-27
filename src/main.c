/*
 * main.c - common main function
 *
 * Davide Chirichella, Filippo Giulietti
 * Alma Mater Studiorum - University of Bologna
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

#define _POSIX_C_SOURCE 200809L

#include "options.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <cuda_runner.h>

int main(int argc, char **argv) {
    ProgramOptions opts;
    int r = parse_program_options(argc, argv, &opts);

    if (r == 1) {
        /* Help printed by parser — exit success as expected by Linux conventions */
        return 0;
    } else if (r == -1) {
        /* Parse error; diagnostics already printed; return conventional exit code 2 */
        return 2;
    }

    /* Strict summary when verbose */
    if (opts.verbose) {
        fprintf(stderr, "Configuration (strict):\n");
        fprintf(stderr, "  Size:        %d x %d\n", opts.width, opts.height);
        fprintf(stderr, "  Octaves:     %d\n", opts.octaves);
        fprintf(stderr, "  Format:      %s\n", opts.format);
        fprintf(stderr, "  Output file: %s\n", opts.output_filename);
        fprintf(stderr, "  Backend:     %s\n", opts.cpu_mode ? "CPU (forced)" : "CUDA (default)");
        if (opts.seed_provided) {
            fprintf(stderr, "  Seed:        %" PRIu64 " (provided)\n", opts.seed);
        } else {
            fprintf(stderr, "  Seed:        %" PRIu64 " (auto-generated)\n", opts.seed);
        }
        fprintf(stderr, "  Verbose:     enabled\n");
    } else {
        fprintf(stderr, "Configuration accepted. Use --verbose for details.\n");
    }

    // test run cuda
    run_on_cuda(&opts);
    // end test run cuda

    /* Delegation point -- do not implement heavy work in this translation unit.
     * Example (commented):
     *
     * if (opts.cpu_mode) {
     *     return run_on_cpu(&opts);
     * } else {
     *     return run_on_cuda(&opts);
     * }
     *
     * Implement run_on_cpu/run_on_cuda in separate modules and link at build time.
     */

    return 0;
}

