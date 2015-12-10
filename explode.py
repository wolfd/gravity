#!/usr/bin/env python2
import numpy as np
import pandas as pd
import random
import math
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-m", "--multiplier", dest="sweep_multiplier",
                  help="what to multiply by", type="float", 
                  default=1.0)
parser.add_option("-n", "--num", dest="num_particles",
                  help="particles in sim", type="int", default=(1000-6))

(options, args) = parser.parse_args()

total_mass = 5.97e3

num_particles = options.num_particles 

G = 66.7 # km^3 / (Yg * s^2)

mass_unit = 1e24 / 1e3 # Yg to Kg
length_unit = 1e3 # Km to m


particles = np.zeros((num_particles, 7))

#vel = 6.453 # km / s
#vel = 6378.100 # km / s from movie
#vel = 11.2 # km / s more accurate
#vel = 9.126 # km /s new sam


# or alderaan
radius_earth = 6367.0 # km

def what_is_vel(r):
    return math.sqrt((6 * G * total_mass) / (5 * r))

def random_vec():
    vec = [random.random() - 0.5, random.random() - 0.5, random.random() - 0.5]
    mag = np.linalg.norm(vec)
    return [v / mag for v in vec]

total_ke = 0

def ke(m, v):
    return 0.5 * (m * mass_unit) * ((v * length_unit) ** 2)

for i in range(num_particles):
    #import pdb; pdb.set_trace()
    particle_mass = total_mass / num_particles
    random_radius = random.random() * radius_earth
    unit_vec = random_vec()
    position = [vc * random_radius for vc in unit_vec] 
    cur_vel = what_is_vel(random_radius)
    total_ke += ke(particle_mass, cur_vel)
    velocity = [cur_vel * vc * options.sweep_multiplier for vc in unit_vec]

    particles[i, 0] = particle_mass 
    particles[i, 1:4] = position
    particles[i, 4:7] = velocity

alderaan_index = 2
orbits = pd.read_csv('orbits.csv', header=None)

p_f = pd.DataFrame(particles)
alderaan = orbits.ix[alderaan_index]
orbits.drop(orbits.index[alderaan_index], inplace=True)
# Shift Planets
orbits[1] = orbits[1].map(lambda x: x - alderaan[1])
orbits[2] = orbits[2].map(lambda y: y - alderaan[2])
orbits[3] = orbits[3].map(lambda z: z - alderaan[3])
# Velocities
#orbits[4] = orbits[4].map(lambda x: x - alderaan[4])
#orbits[5] = orbits[5].map(lambda y: y - alderaan[5])
#orbits[6] = orbits[6].map(lambda z: z - alderaan[6])

# Positions
#p_f[1] = p_f[1].map(lambda x: x + alderaan[1])
#p_f[2] = p_f[2].map(lambda y: y + alderaan[2])
#p_f[3] = p_f[3].map(lambda z: z + alderaan[3])
# Velocities
p_f[4] = p_f[4].map(lambda x: x + alderaan[4])
p_f[5] = p_f[5].map(lambda y: y + alderaan[5])
p_f[6] = p_f[6].map(lambda z: z + alderaan[6])
orbits = orbits.append(p_f)

orbits.to_csv('input.csv', index=False, header=False)

with open("summary-{0}.txt".format(options.sweep_multiplier), 'w') as summary:
    summary.write("multiplier: {0}\n".format(options.sweep_multiplier))
    summary.write("ke: {0} J\n".format(total_ke))


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
