import numpy as np
import random
import time

datafile = "hw1_15_train.dat"

def load_data(datafile):
	X = []
	Y = []
	data = open(datafile,"r").readlines()
	for line in data:
		x = line.split("\t")[0].split(" ")
		x = [1] + x
		X.append( np.asarray(map( float,x )) )
		Y.append( float(line.split("\t")[1].strip()) )
	return np.asarray(X), np.asarray(Y)

def randomize(X,Y):
	random_X = []
	random_Y = []
	indexlist = range(len(X))
	for i in random.sample( indexlist, len(X) ):
		random_X.append(X[i])
		random_Y.append(Y[i])
	return random_X, random_Y
	
def sign(h):
	if (h <= float(0)):
		return  float(-1)
	else:
		return float(1)

def train(X, Y, TunelearnrateOrNot):
	if TunelearnrateOrNot:
		alpha = 0.5
	else:
		alpha = 1
	w = np.asarray( [0.0,0.0,0.0,0.0,0.0] )
	N = len(Y)  # number of data
	t = 0       # error times until a full cycle
	k = 0       # the k line of data 
	flag = True
	while True:
		if (k == N):
			if flag:
				break
			else:
				k = 0 
				flag = True 	
		# error occurs : 
		if (sign(np.inner(X[k],w)) != Y[k]):
				w = w + alpha * Y[k] * X[k]
				t += 1
				flag = False
					
		k += 1
	return t

def naive_cycle():
	X, Y = load_data(datafile) 
	t = train(X, Y, False)
	return t

def random_cycle(ITERATION):
	timelist = []
	X, Y = load_data(datafile) 
	for i in range(ITERATION):
		random_X, random_Y = randomize(X, Y)
		t = train(random_X, random_Y, False)
		timelist.append(t)
	t = sum(timelist)/len(timelist)
	return t

def tunelearn_cycle(ITERATION):
	timelist = []
	X, Y = load_data(datafile) 
	for i in range(ITERATION):
		random_X, random_Y = randomize(X, Y)
		t = train(random_X, random_Y, True)
		timelist.append(t)
	t = sum(timelist)/len(timelist)
	return t
	
	

def main():
	# HW 15
	naive_t = naive_cycle()
	print "Naive : Number of update is ", naive_t  # 45 times
	'''
	# HW 16
	ITERATION = 2000
	random_t = random_cycle(ITERATION) 
	print "Random : Number of update is ", random_t # approximately 40 times
	# HW 17
	ITERATION = 2000
	tune_t = tunelearn_cycle(ITERATION) 
	print "Tune : Number of update is ", tune_t # approximately 40 times
	'''

if __name__ == "__main__":
	main()

