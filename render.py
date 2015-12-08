#!/usr/bin/env python2
import pandas as pd
import pygame
from pygame import gfxdraw
import sys

ITERATIONS = 8000 / 60

PARTICLES = 1000 

size = [800, 800]
screen = pygame.display.set_mode(size)

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
color = pygame.Color.g

clock = pygame.time.Clock()

locations = pd.read_csv('output.csv')

x = locations.ix[:,0]
y = locations.ix[:,1]
z = locations.ix[:,2]
xv = locations.ix[:,3]
yv = locations.ix[:,4]
zv = locations.ix[:,5]

scale = 7e8 

def to_pygame(x, y):
    return (size[0] / 2 + (x / scale) * size[0], size[1] / 2 + (y / scale) * size[0])

while True:
    for t in [u for u in xrange(ITERATIONS - 1)]:# if u % 50 == 0]:
        #clock.tick(10)
        print str(t) 
        screen.fill(BLACK)
        for l in range(PARTICLES):
            i = t * PARTICLES + l
            #print (x[i]/scale,y[i]/scale)
            p = to_pygame(x[i], y[i])
            p_v = to_pygame(x[i] + xv[i], y[i] + yv[i])
            pygame.draw.line(screen, GREEN, p, p_v, 1)
            if l == 3:
                pygame.draw.line(screen, RED, p, p_v, 1)
	#gfxdraw.pixel(surface, l.x, l.y, color)
   
        pygame.display.update() 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 pygame.quit(); sys.exit();

