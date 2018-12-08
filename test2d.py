#!/usr/bin/env python3

import numpy as np
import pygame
from libgrav import spaceship, body, eccentricity


pygame.init()
W, H = 800, 800
screen = pygame.display.set_mode((W, H))

frame_center = np.array([W/2, H/2])
imgs = {'normal': 'imgs/shuttle.png',
        'left': 'imgs/shuttle_left.png',
        'right': 'imgs/shuttle_right.png',
        'back': 'imgs/shuttle_back.png'
        }
ship = spaceship(img_files = imgs,
                 pos = frame_center + np.array([300,0]),
                 power = 5)
asteroid = body(img_file = 'imgs/asteroid.png',
                pos = frame_center.copy(),
                mass = 20000)


G_univ = 10

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
                pygame.quit()

    ship.handle_keys()

    screen.fill((0,0,0))

    asteroid.draw(screen)

    ship.gravity(asteroid, G=G_univ, dt=0.1)
    ship.move()
    ship.draw(screen)

    pygame.display.update()

    print('\r{:0.4f}'.format(eccentricity(ship, asteroid, G_univ)), end='')

    clock.tick(60)
