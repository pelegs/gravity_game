#!/usr/bin/env python3

import numpy as np
import pygame
import sys
from libgrav import *


power = float(sys.argv[1])
ang_power = float(sys.argv[2])

pygame.init()
W, H = 800, 800
screen = pygame.display.set_mode((W, H))
x0, y0 = 0., 200.

frame_center = np.array([W/2, H/2])
imgs = {'normal': 'imgs/shuttle.png',
        'left': 'imgs/shuttle_left.png',
        'right': 'imgs/shuttle_right.png',
        'back': 'imgs/shuttle_back.png'
        }
ship = spaceship(img_files = imgs,
                 pos = frame_center + np.array([x0,y0]),
                 power = power,
                 ang_power = ang_power)
asteroid = body(img_file = 'imgs/asteroid.png',
                pos = frame_center.copy(),
                mass = 20000)
G_univ = 10
r0 = np.linalg.norm(ship.pos - asteroid.pos)
vx = np.sqrt(G_univ*asteroid.mass/r0) * 0.9995
vy = 0.0
ship.vel = np.array([vx, vy]).astype(np.float64)

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
    ship.move(asteroid, G=G_univ, dt=0.1)

    # Get orbit shape
    r1 = ship.pos - asteroid.pos
    perp_vec = py_rotate(ship.vel, np.pi/2)
    a, b = ellipse_axes(ship, asteroid, G_univ)
    r1_angle = np.arctan2(r1[1], r1[0])

    c = clockwise(r1, perp_vec)
    da = -c*py_angle_between(r1, perp_vec)
    r2 = -py_rotate(r1, 2*da)
    make_norm(r2, 2*a - np.linalg.norm(r1))

    second_center = ship.pos + r2
    r12_angle = -get_angle(second_center - asteroid.pos)
    ellipse_center = 0.5*(asteroid.pos + second_center)
    ellipse_list = get_ellipse(ellipse_center, a, b, r12_angle, 1000)

    # Draw stuff
    screen.fill((0,0,0))
    for point in ellipse_list:
        pygame.draw.circle(screen, (55,0,150), point.astype(int), 1, 0)
    pygame.draw.circle(screen, (255, 0, 0), asteroid.pos.astype(int), 15, 0)
    ship.draw(screen)

    pygame.display.update()

    #print('\r{:0.4f}, {:0.4f}'.format(ship.vel[0], ship.vel[1]), end='')
    print('\r{:0.4f}, ({:0.4f}, {:0.4f})'.format(ship.ang_vel, ship.dir[0], ship.dir[1]), end='')

    clock.tick(60)
