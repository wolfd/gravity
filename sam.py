import numpy as np
from dotmap import DotMap
import math

# 

class Body(object):
	"Create a celestial body object with given mass, position, and velocity"
	def __init__(self, name, mass, x, y, vx, vy):
		self.name = name
		self.mass = mass
		self.x, self.y = x, y
		self.v = DotMap()
		self.v.x, self.v.y = vx, vy
		self.velocity = math.sqrt(self.v.x**2 + self.v.y**2)
		self.kinetic = 0.5 * self.mass * self.velocity**2

	def dist(self, body):
		return math.sqrt( (self.x-body.x)**2 + (self.y-body.y)**2 )