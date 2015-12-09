#!/usr/bin/env python2
import numpy as np
import pandas as pd
import random

total_mass = 5.9e3

num_particles = 600

num_ring = num_particles / 4
num_other = num_particles - num_ring

initial_pos = (0, 0, 0)

particles = np.zeros((num_particles, 7))

vel = 6.453 # km / s

# or alderaan
radius_earth = 6367.0 # km
first_timestep = radius_earth / escape_vel # seconds

def random_vec():
    vec = [random.random() - 0.5, random.random() - 0.5, random.random() - 0.5]
    mag = np.linalg.norm(vec)
    return [v / mag for v in vec]

for i in range(num_particles):
    mag = vel
    #import pdb; pdb.set_trace()
    velocity = [mag * v for v in random_vec()]
    position = [v * first_timestep for v in velocity]
    particles[i, 0] = total_mass / num_particles
    particles[i, 1:4] = position
    particles[i, 4:7] = velocity


alderaan_index = 2
orbits = pd.read_csv('orbits.csv', header=None)

p_f = pd.DataFrame(particles)
alderaan = orbits.ix[alderaan_index]
# Positions
p_f[1] = p_f[1].map(lambda x: x + alderaan[1])
p_f[2] = p_f[2].map(lambda y: y + alderaan[2])
p_f[3] = p_f[3].map(lambda z: z + alderaan[3])
# Velocities
p_f[4] = p_f[4].map(lambda x: x + alderaan[4])
p_f[5] = p_f[5].map(lambda y: y + alderaan[5])
p_f[6] = p_f[6].map(lambda z: z + alderaan[6])
orbits.drop(orbits.index[alderaan_index], inplace=True)
orbits = orbits.append(p_f)

orbits.to_csv('input.csv', index=False, header=False)

print orbits.shape

#print alderaan


#total_mass_left = total_mass
#for i in range(num_particles):
#    this_mass = random.uniform(0, total_mass_left / 8)
#    total_mass_left -= this_mass
#    particle_masses[i] = this_mass

#print particles
#
#import numpy as np
#import scipy.stats as stats
#import pylab as pl
#
#particle_masses = particles[:, 4]
#
#h = sorted(particle_masses)
#fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
#
#pl.plot(h,fit,'-o')
#
#pl.hist(h,normed=True)      #use this to draw histogram of your data
#
#pl.show()                   #use may also need add this 
