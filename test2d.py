#!/usr/bin/env python3

import numpy as np
import pygame
from starship import spaceship


pygame.init()
W, H = 800, 800
screen = pygame.display.set_mode((W, H))

frame_center = np.array([W/2, H/2])
ship = spaceship(img_file='imgs/shuttle.png',
                 pos = frame_center)

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            running = False

    ship.handle_keys()

    screen.fill((0,0,0))
    ship.move()
    ship.draw(screen)
    pygame.display.update()

    clock.tick(60)
