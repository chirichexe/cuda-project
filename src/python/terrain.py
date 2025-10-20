import numpy as np
import pygame

WIDTH, HEIGHT = 512, 512  # sostituire con le tue dimensioni

# 1. Leggi il file binario
heightmap = np.fromfile("../cuda/heightmap.bin", dtype=np.float32)
heightmap = heightmap.reshape((HEIGHT, WIDTH))

# 2. Normalizza tra 0 e 255 per Pygame
min_val = heightmap.min()
max_val = heightmap.max()
heightmap_normalized = ((heightmap - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# 3. Inizializza Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Heightmap Viewer")

# 4. Crea una superficie Pygame dall'array
surf = pygame.surfarray.make_surface(heightmap_normalized.T)  # trasposta per orientamento corretto

# 5. Loop principale (solo per mostrare, non aggiornare)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(surf, (0, 0))
    pygame.display.flip()

pygame.quit()

