#!/usr/bin/env python2
import pandas as pd
import pygame
from pygame import gfxdraw
import sys


size = [800, 800]
screen = pygame.display.set_mode(size)

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
color = pygame.Color.g

clock = pygame.time.Clock()

locations = pd.read_csv('locations.csv')
#import pdb; pdb.set_trace()

x = locations.ix[:,2]
y = locations.ix[:,3]
xv = locations.ix[:,4]
yv = locations.ix[:,5]

scale = 10e7

while True:
    clock.tick(10)
    for t in range(1000):
        print 'tick'
        screen.fill(BLACK)
        for l in range(512):
            i = t + l
            #print (x[i]/scale,y[i]/scale)
            pygame.draw.line(screen, GREEN, [x[i]/scale, y[i]/scale], [x[i]/scale + xv[i], y[i]/scale + yv[i]], 1)
	#gfxdraw.pixel(surface, l.x, l.y, color)
   
        pygame.display.update() 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 pygame.quit(); sys.exit();

