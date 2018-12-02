import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pi, atan2, sin, cos
from tqdm import tqdm
import pygame


CW = 1
CCW = -1


cdef double dot2d(np.ndarray[double, ndim=1] v1,
                  np.ndarray[double, ndim=1] v2):
    return v1[0]*v2[0] + v1[1]*v2[1]


cdef double get_xy_angle(np.ndarray[double, ndim=1] vec):
    angle = atan2(vec[1], vec[0])
    if angle < 0:
        angle += 2*pi
    return angle


cdef double rad2deg(double angle):
    return angle * 180/pi


cdef double deg2rad(double angle):
    return angle * pi/180


cdef np.ndarray[double, ndim=1] mat_vec_dot(np.ndarray[double, ndim=2] matrix,
                                            np.ndarray[double, ndim=1] vec):
    cdef np.ndarray[double, ndim=1] v_new = np.zeros(2).astype(np.float64)
    v_new[0] = dot2d(matrix[0], vec)
    v_new[1] = dot2d(matrix[1], vec)
    return v_new


cdef np.ndarray[double, ndim=1] rotate(np.ndarray[double, ndim=1] vec,
                                       double angle):
    cdef double s = sin(angle)
    cdef double c = cos(angle)
    cdef np.ndarray[double, ndim=2] R_matrix = np.array([[c, -s],
                                                         [+s, c]])
    return mat_vec_dot(R_matrix, vec)


class spaceship:
    def __init__(self,
                 img_file,
                 rot_angle = deg2rad(5),
                 power = 5,
                 pos = np.zeros(2),
                 vel = np.zeros(2),
                 dir = np.array([1, 0]).astype(np.float64)):

        self.image = pygame.image.load(img_file)
        self.active_image = pygame.image.load(img_file)

        self.rot_angle = rot_angle
        self.power = power
        self.pos = pos
        self.vel = vel
        self.dir = dir

        # Aligning ship
        self.rotate(direction=CW)
        self.rotate(direction=CCW)

    def rotate(self, direction=CW):
        self.dir = rotate(self.dir, self.rot_angle*direction)
        heading = rad2deg(get_xy_angle(self.dir))
        self.active_image = pygame.transform.rotate(self.image, 180-heading)

    def accelerate(self, dt=0.1):
        self.vel += self.power * self.dir * dt

    def move(self, dt=0.1):
        self.pos += self.vel * dt

    def handle_keys(self):
        key = pygame.key.get_pressed()
        if key[pygame.K_RIGHT]:
            self.rotate(direction=CW)
        elif key[pygame.K_LEFT]:
            self.rotate(direction=CCW)
        if key[pygame.K_UP]:
            self.accelerate()

    def draw(self, surface, ref=np.zeros(2)):
        pos = (self.pos + ref).astype(int)
        surface.blit(self.active_image, pos)
