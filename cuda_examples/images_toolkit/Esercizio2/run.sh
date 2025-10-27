#!/bin/bash

# Compila
echo "Compilazione..."
# TODO
echo ""

# Crea directory
mkdir -p ncu_reports output_images

# Test tutte le immagini con varie dimensioni di filtro e block size
for img in ../Images/img*.jpg; do
    IMG_NAME=$(basename "$img" .jpg)
    echo "Testing: $IMG_NAME"
    
    # Test con diversi filter size
    for fs in 3 5 7; do
        echo "  Filter Size ${fs}x${fs}"
        
        for bs in 4 8 16 32; do
            echo "    Block Size ${bs}x${bs}"
            
            # Profiling (solo per la combinazione più interessante)
            if [ $fs -eq 5 ] && [ $bs -eq 16 ]; then
                # TODO
            else
                ./separable_convolution "$img" $fs $bs > /dev/null 2>&1
            fi
            
            # Sposta output
            mv output_2D.png "output_images/${IMG_NAME}_fs${fs}_bs${bs}_2D.png" 2>/dev/null
            mv output_separable.png "output_images/${IMG_NAME}_fs${fs}_bs${bs}_sep.png" 2>/dev/null
            mv output_cpu.png "output_images/${IMG_NAME}_fs${fs}_bs${bs}_cpu.png" 2>/dev/null
        done
    done
    echo ""
done

echo "Report salvati in:   ncu_reports/"
echo "Immagini salvate in: output_images/"
echo ""
echo "Nota: Le immagini _2D.png e _sep.png dovrebbero essere identiche!"
