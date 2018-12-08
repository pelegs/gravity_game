#!/usr/bin/env python3

import numpy as np
import pygame
from libgrav import *


def draw_vector(surface, vec, start=np.zeros(2), color=(255, 255, 255), width=1):
    pygame.draw.line(surface, color, start, start+vec, width)


pygame.init()
W, H = 800, 800
screen = pygame.display.set_mode((W, H))

frame_center = np.array([W/2, H/2])
x0 = 50
imgs = {'normal': 'imgs/blue_dot.png',
        'left': 'imgs/blue_dot.png',
        'right': 'imgs/blue_dot.png',
        'back': 'imgs/blue_dot.png'
        }
ship = spaceship(img_files = imgs,
                 pos = frame_center + np.array([x0,0]),
                 vel = np.array([0., 10.]),
                 power = 5)
asteroid = body(img_file = 'imgs/red_dot.png',
                pos = frame_center.copy(),
                mass = 10000)


dt = 0.05
G_univ = 10
V0 = np.sqrt(G_univ * asteroid.mass/x0) * 1.2
ship.vel = np.array([0,V0]).astype(np.float64)
r2 = np.zeros(2).astype(np.float64)

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    ship.handle_keys()
    ship.move(asteroid, G=G_univ, dt=dt)
    r1 = ship.pos - asteroid.pos
    perp_vec = pyrotate(ship.vel, np.pi/2)
    a, b = ellipse_axes(ship, asteroid, G_univ)
    r1_angle = np.arctan2(r1[1], r1[0])

    c = clockwise(r1, perp_vec)
    da = -c*py_angle_between(r1, perp_vec)
    r2 = -py_rotate(r1, 2*da)
    make_norm(r2, 2*a - np.linalg.norm(r1))
    second_center = ship.pos + r2

    # Draw stuff
    screen.fill((0,0,0))
    pygame.draw.circle(screen, (255,0,0), asteroid.pos.astype(int), 15, 0)
    pygame.draw.circle(screen, (0,100,255), ship.pos.astype(int), 5, 0)
    #draw_vector(screen, r1, start=asteroid.pos, width=1)
    #draw_vector(screen, ship.vel.astype(int), start=ship.pos, color=(0,200,255), width=1)
    #draw_vector(screen, perp_vec.astype(int), start=ship.pos, color=(0,255,55), width=1)
    #draw_vector(screen, r2.astype(int), start=ship.pos, color=(255,0,200), width=1)
    pygame.draw.circle(screen, (255,0,200), second_center.astype(int), 15, 0)

    pygame.display.update()
    clock.tick(60)

pygame.quit()
