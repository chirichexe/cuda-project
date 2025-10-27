#!/bin/bash

# Compila
echo "Compilazione..."
# TODO
echo ""

# Crea directory
mkdir -p ncu_reports output_images

# Test tutte le immagini con tutti i block size
for img in ../Images/img*.jpg; do
    IMG_NAME=$(basename "$img" .jpg)
    echo "Testing: $IMG_NAME"
    
    for bs in 4 8 16 32; do
        echo "  Block Size ${bs}x${bs}"
        
        # Profiling
        # TODO
        
        # Sposta output
        mv output_gpu.png "output_images/${IMG_NAME}_bs${bs}_gpu.png" 2>/dev/null
        mv output_cpu.png "output_images/${IMG_NAME}_bs${bs}_cpu.png" 2>/dev/null
    done
    echo ""
done

echo "Report salvati in:   ncu_reports/"
echo "Immagini salvate in: output_images/"
