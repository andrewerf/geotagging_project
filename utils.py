from math import cos

def convert_accuracy(pos, acc_m):
	acc_m = acc_m/100

	lat_deg = acc_m / 360.0
	long_deg = cos(pos[0]) * acc_m / 360.0

	return (lat_deg, long_deg)

class Area:
	def __init__(self, pos, accuracy_m = 200):
		self.pos = pos
		self.delta = convert_accuracy(pos, accuracy_m)

	def __contains__(self, pos):
		ok = True

		for i in [0,1]:
			ok = ok and (self.pos[i] - self.delta[i] <= pos[i] <= self.pos[i] + self.delta[i])

		return ok

	def __add__(self, other):
		pass

	def __str__(self):
		return str(self.pos) + ' ' + str(self.delta)
