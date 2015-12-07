import numpy as np
from sam import Body

sun_mass = 1.989e+9 # yg
masses = [0.3301e+3, 5.9726e+3, 2.7547e+3, 0.64174e+3, 0.4859e+3] # yg
distances = [46e+6, 147.09e+6, 194.5e+6, 239.4e+6, 269.3e+6] # km
velocities = [58.98, 30.29, 26.12, 23.54, 22.2] # km/s

planets = ['Raisa', 'Alderaan', 'Delaya', 'Avirandel', 'Avishan']

for i in range(0,len(planets)):
	if i % 2 != 0:
		exec "planet%s = Body(planets[i], masses[i], distances[i], 0, 0, velocities[i])" %(i+1)
	else:
		exec "planet%s = Body(planets[i], masses[i], 0, distances[i], velocities[i], 0)" %(i+1)

def distance(a, b):
	return math.sqrt( (a.x-b.x)**2 + (a.y-b.y)**2 )