#!/bin/bash

# Compila
echo "Compilazione..."
# TODO
echo ""

# Crea directory
mkdir -p ncu_reports output_images

# Test tutte le immagini con tutti i block size e direzioni
for img in ../Images/img*.jpg; do
    IMG_NAME=$(basename "$img" .jpg)
    echo "Testing: $IMG_NAME"
    
    for bs in 8 16 32; do
        echo "  Block Size ${bs}x${bs}"
        
        for dir in cw ccw; do
            echo "    Direzione: $dir"
            
            # Profiling
            # TODO
            
            # Sposta output
            mv output_gpu.png "output_images/${IMG_NAME}_bs${bs}_${dir}_gpu.png" 2>/dev/null
            mv output_cpu.png "output_images/${IMG_NAME}_bs${bs}_${dir}_cpu.png" 2>/dev/null
        done
    done
    echo ""
done

echo "Report salvati in:   ncu_reports/"
echo "Immagini salvate in: output_images/"

