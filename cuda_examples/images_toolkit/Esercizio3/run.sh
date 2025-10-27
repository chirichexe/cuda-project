#!/bin/bash

# Verifica argomenti
if [ $# -lt 1 ]; then
    echo "Usage: $0 <num_frames> [methods]"
    echo "Example: $0 100          # Processa 100 frame con entrambi i metodi"
    echo "         $0 50 loop      # Processa 50 frame solo con metodo loop"
    echo "         $0 200 3d       # Processa 200 frame solo con metodo 3d"
    exit 1
fi

NUM_FRAMES=$1
METHODS=${2:-"loop 3d"}  # Default: entrambi i metodi

# Compila
echo "Compilazione..."
# TODO
echo ""

# Crea directory
mkdir -p ncu_reports output_frames

# Pattern per i frame del video
FRAME_PATTERN="../Videos/frame_%03d.jpg"

echo "Testing video grayscale conversion"
echo "Frame pattern: $FRAME_PATTERN"
echo "Numero frame: $NUM_FRAMES"
echo "Metodi: $METHODS"
echo ""

# Test con i metodi specificati
for method in $METHODS; do
    echo "=== Metodo: $method ==="
    
    for bs in 4 8 16 32; do
        echo "  Block Size ${bs}x${bs}"
        
        # Profiling
        # TODO
        
        # Rinomina output per distinguere i metodi
        if [ -d "output_frames" ]; then
            mv output_frames "output_frames_${method}_bs${bs}_f${NUM_FRAMES}" 2>/dev/null
        fi
    done
    echo ""
done

# Processa 100 frame con entrambi i metodi
# ./run.sh 100

# Processa 50 frame solo con metodo loop
# ./run.sh 50 loop

# Processa 200 frame solo con metodo 3d
# ./run.sh 200 3d

# Processa 10 frame con entrambi i metodi (veloce per test)
# ./run.sh 10
