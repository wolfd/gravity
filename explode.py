#!/usr/bin/env python2
import numpy
import pandas
import random

total_mass = 5.9e3

num_particles = 600

num_ring = num_particles / 4
num_other = num_particles - num_ring

initial_pos = (0, 0, 0)

particle_masses = numpy.zeros((num_particles))



#total_mass_left = total_mass
#for i in range(num_particles):
#    this_mass = random.uniform(0, total_mass_left / 8)
#    total_mass_left -= this_mass
#    particle_masses[i] = this_mass

print particle_masses

import numpy as np
import scipy.stats as stats
import pylab as pl

h = sorted(particle_masses)
fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

pl.plot(h,fit,'-o')

pl.hist(h,normed=True)      #use this to draw histogram of your data

pl.show()                   #use may also need add this 
