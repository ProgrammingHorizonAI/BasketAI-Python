import pygame
import numpy as np

class Basket:
    img = pygame.image.load('basket.png')
    scaled_image = pygame.transform.scale_by(img, .1)
    w, h = scaled_image.get_size()
    print(w)

    def draw(self, surface, flip):
        if flip:
            surface.blit(pygame.transform.flip(self.scaled_image, flip_x=True, flip_y=False), (720, 100))
        else:
            surface.blit(self.scaled_image, (-10, 100))