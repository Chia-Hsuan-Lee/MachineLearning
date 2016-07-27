import numpy as np
import random
import time

datafile = "hw1_18_train.dat"
testfile = "hw1_18_test.dat"

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

def pocket_train(X, Y, updates):
	w = np.asarray( [0.0,0.0,0.0,0.0,0.0] )
	wg = w # wg is the current best w 
	N = len(Y)  # number of data
	current_error = test(X, Y, wg)
	for i in range(updates):
		for i in random.sample(range(N) , N):
			if (sign(np.inner(X[i],w)) != Y[i]):
				w = w + Y[i] * X[i]
				error = test(X, Y, w)
				if error < current_error:
					current_error = error 
					wg = w 
				break
	return wg


def pla_train(X, Y, updates):
	w = np.asarray( [0.0,0.0,0.0,0.0,0.0] )
	N = len(Y)  # number of data
	t = 0       # error times until a full cycle
	k = 0       # the k line of data 
	flag = True
	while True:
		if (t==50):
			#print "pla reaches 50 updates !"
			break
		if (k == N):
			if flag:
				print 'pla ends with no error in one full cycle ! '
				break
			else:
				k = 0 
				flag = True 	
		# error occurs : 
		if (sign(np.inner(X[k],w)) != Y[k]):
				w = w + Y[k] * X[k]
				t += 1
				flag = False
		k += 1
	return w

def test(test_X, test_Y, w):
	sumerr = 0 
	#sumerr =  sum([ 1 for i in xrange(len(test_Y)) if ( sign(np.inner(test_X[i],w) != test_Y[i] )) ])
	for i in range(len(test_Y)):
		if (sign(np.inner(test_X[i], w))!= test_Y[i]):
			sumerr+=1
	error = float(sumerr)/len(test_Y)
	return error 

def pure_pocket50(updates, ITERATION):
	X, Y = load_data(datafile) 
	test_X, test_Y = load_data(testfile) 
	errorlist = []
	for i in range(ITERATION):
		random_X, random_Y = randomize(X, Y) 
		wg = pocket_train(random_X, random_Y, updates)
		error = test(test_X, test_Y, wg)
		errorlist.append(error)
	avg_error = float(sum(errorlist))/len(errorlist)
	return avg_error

def pla_50(updates, ITERATION):
	X, Y = load_data(datafile) 
	test_X, test_Y = load_data(testfile) 
	errorlist = []
	for i in range(ITERATION):
		random_X, random_Y = randomize(X, Y) 
		wg = pla_train(random_X, random_Y, updates)
		error = test(test_X, test_Y, wg)
		print error
		errorlist.append(error)
	avg_error = float(sum(errorlist))/len(errorlist)
	return avg_error
	


def pure_pocket100(updates, ITERATION):
	X, Y = load_data(datafile) 
	test_X, test_Y = load_data(testfile) 
	errorlist = []
	for i in range(ITERATION):
		random_X, random_Y = randomize(X, Y) 
		wg = pocket_train(random_X, random_Y, updates)
		error = test(test_X, test_Y, wg)
		errorlist.append(error)
	avg_error = float(sum(errorlist))/len(errorlist)
	return avg_error
'''
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
'''	
	

def main():
	'''
	# HW 18
	updates = 50
	ITERATION = 2000
	avg_error = pure_pocket50(updates, ITERATION)
	print "Pocket : Average error rate is ", avg_error  # 0.132956

	# HW 19
	updates = 50
	ITERATION = 2000
	avg_error = pla_50(updates, ITERATION) 
	print "Pla 50 updates : Average error rate is ", avg_error   # 0.368648
	'''
	# HW 20
	updates = 100
	ITERATION = 2000
	avg_error = pure_pocket100(updates, ITERATION) 
	print "Pocket 100 iteration : Average error rate is ", avg_error # 0.115731

if __name__ == "__main__":
	main()

