import numpy as np
import math

trainfile = open("/home/chiahsuan/ml/foundation/hw3/hw3_train.dat").readlines()
testfile = open("/home/chiahsuan/ml/foundation/hw3/hw3_test.dat").readlines()

def loaddata(filename):
	X = []
	Y = []
	for line in filename:
		x =  map(float,line.split(" ")[1:-1]) 
		X.append( np.asarray(x) )
		Y.append( float(line.split(" ")[-1].strip()))
	return np.asarray(X), np.asarray(Y)

# theta (s) = 1 / 1+ exp(-s)
def countgradient(X, Y, w):
	N = len(Y)
	sumein = 0
	for i in range(N):
		s = (-1.0) * Y[i] * np.dot(w,X[i]) 
		theta = 1.0 / (1 + math.exp(-s))
		secondterm = (-1.0) * Y[i]* X[i]
		sumein += theta * secondterm 
	gradient = sumein / N
	return gradient

def logisticregression(ita, ITERATION, stochastic):
	w = np.zeros(20)
	X, Y = loaddata(trainfile)
	index = 0
	for i in range(ITERATION):
		if stochastic:
			s = (-1.0) * Y[index] * np.dot(w,X[index]) 
			theta = 1.0 / (1 + math.exp(-s))
			secondterm = (-1.0) * Y[index]* X[index]
			gradient = theta * secondterm
			w = w - ita * gradient	
			index += 1
			if (index == 1000):
				index = 0
		else:
			gradient = countgradient(X, Y, w)
			w = w - ita * gradient
	return w 
  
def CountEout(w):
	testdata, testoutput = loaddata(testfile)
	N = len(testoutput)
	error = 0.0
	for i in range(N):
		s = np.dot( w, testdata[i] ) 
		scores = 1/ ( 1 + math.exp(-s))
		h = np.where(scores>=0.5, 1.0, -1.0)
		if h != testoutput[i]:
			error += 1
	Eout = float(error) / N
	return Eout  

def main():
	'''
	# HW 18
	ita = 0.001
	ITERATION = 2000
	X,Y = loaddata(trainfile)
	w = logisticregression(ita, ITERATION, False)
	Eout = CountEout(w)  # 0.471666666667

	# HW 19
	ita = 0.01
	ITERATION = 2000
	X,Y = loaddata(trainfile)
	w = logisticregression(ita, ITERATION, False)
	Eout = CountEout(w) # 0.220666666667
	'''	
	# HW 20
	ita = 0.001
	ITERATION = 2000
	X,Y = loaddata(trainfile)
	w = logisticregression(ita, ITERATION, True)
	Eout = CountEout(w) # 0.471666666667
	print Eout
	
if __name__ == '__main__':
	main()
