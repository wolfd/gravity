#!/usr/bin/env python2
import pandas as pd
import pygame
import sys
from optparse import OptionParser

size = [800, 800]
screen = pygame.display.set_mode(size)

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
YELLOW = (150, 150, 0)
RED =   (255,   0,   0)
background_color = BLACK
particle_color = GREEN
planet_color = RED 
sun_color = YELLOW 

clock = pygame.time.Clock()

parser = OptionParser()
parser.add_option("-f", "--file", dest="csv_name",
                  help="read from FILE", metavar="FILE",
		  default="output.csv")
parser.add_option("-n", "--num", dest="num_particles",
                  help="particles in sim", type="int")
parser.add_option("-s", "--skip", dest="skip",
                  help="iterations to skip", type="int",
                  default=1)
parser.add_option("-i", "--save-image", dest="save",
                  default=None)

(options, args) = parser.parse_args()

PARTICLES = options.num_particles


reader = pd.read_csv(options.csv_name, chunksize=PARTICLES, header=None)#iterator=True,
scale = 1e-3
scale = 16.0 
scale = 0.0009765625
scale = 2.0

sun_index = 0

focus_planet = 2
#focus_planet = 0
offset = (-363300, 0)
#offset = (0, 0)

#draw_background = False if options.save else True
draw_background = True

def save_it(pend=""):
    if options.save != None:
        from StringIO import StringIO
        from PIL import Image
        data = pygame.image.tostring(screen, 'RGBA')
        img = Image.frombytes('RGBA', (800, 800), data)
        with open(options.save + pend + ".png", 'wb') as f:
            img.save(f, 'PNG')

def to_pygame(x, y):
    scale_screen = scale / size[0]
    return ((x * scale_screen) + (size[0] / 2), (y * scale_screen) + (size[1] / 2))
    #return (size[0] / 2 + (x / scale) * size[0], size[1] / 2 + (y / scale) * size[0])

def get_particle(df, index):
    return (df['x'][index],  df['y'][index],  df['z'][index], 
            df['xv'][index], df['yv'][index], df['zv'][index])

def subtract_particles(a, b):
    return [a_i - b_i for a_i, b_i in zip(a, b)]

def draw_particle(p, color=particle_color):
    a = to_pygame(p[0], p[1])#, p[3], p[4]) 
    b = to_pygame(p[0], p[1])#, p[3], p[4]) 

    pygame.draw.line(screen, color, a, b, 1)

screen.fill(background_color)
iteration = 0
for d in reader:
    iteration += 1
    if iteration % options.skip != 0:
        continue
    d.columns = ['x', 'y', 'z', 'xv', 'yv', 'zv']
    ref = get_particle(d, focus_planet)
    pygame.display.set_caption(str(iteration))

    if draw_background:
        screen.fill(background_color)

    #import pdb; pdb.set_trace()

    for i in range(d.shape[0]):
        focus = get_particle(d, focus_planet)
        cur = get_particle(d, i)

        rel = subtract_particles(cur, focus)
        rel[0] = rel[0] + offset[0]
        rel[1] = rel[1] + offset[1]
        if i == sun_index:
            draw_particle(rel, sun_color)
        elif i in range(6):
            draw_particle(rel, planet_color)
        else:
            draw_particle(rel)

    pygame.display.update() 
    if iteration % 12 == 1:
        save_it("-2min-" + str(iteration / 12))
        #break

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            screen.fill(background_color)
            if event.button == 1:
                focus_planet = (focus_planet + 1) % PARTICLES 
            elif event.button == 3:
                focus_planet = (focus_planet - 1) % PARTICLES 
            print focus_planet
        if event.type == pygame.KEYDOWN:
            screen.fill(background_color)
            if event.key == pygame.K_PERIOD:
                scale *= 2
            elif event.key == pygame.K_COMMA:
                scale *= 0.5
            elif event.key == pygame.K_b:
                draw_background = not draw_background
            print scale
        elif event.type == pygame.QUIT:
             pygame.quit(); sys.exit();


save_it()
