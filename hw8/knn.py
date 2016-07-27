import numpy as np
from scipy.spatial import distance
import math

trainfile = open("/home/chiahsuan/ml/foundation/hw8/hw8_train.dat")
testfile = open("/home/chiahsuan/ml/foundation/hw8/hw8_test.dat")

k = 5
gamma = 0.1

def readdata(filename):
	xlist = []
	ylist = []
	for line in filename.readlines():
		x= map(float, line.strip().split(" ")[0:-1])
		xlist.append(np.array(x))
		ylist.append(float(line.strip().split(" ")[-1]))
	return np.array(xlist), np.array(ylist)	

def knn(k, x, y, test_sample):
	distancelist = []
	for i in range(len(x)):
		distancevalue = distance.euclidean(x[i], test_sample)
		distancelist.append(distancevalue)
	order = np.argsort(distancelist)
	ret = 0
	for i in range(k):
		ret += y[order[i]]
	if ret > 0:
		return 1.0
	else:
		return -1.0

def uniform_g(x, y, test_sample, gamma):
	valuelist = []
	for i in range(len(x)):
		distancevalue = distance.euclidean(x[i], test_sample)
		value= 1.0*y[i]*math.exp( (-1.0)*gamma*distancevalue)
		valuelist.append(value)
	ret = sum(valuelist)
	if ret > 0:
		return 1.0
	else:
		return -1.0
	  

def main():
	xlist, ylist = readdata(trainfile)
	# HW12: count Ein   0.16 
	error = 0
	for i in range(len(xlist)):
		predict_y = knn(k, xlist, ylist, xlist[i]) 
		if predict_y != ylist[i]:
			error += 1
	print "Ein :", 1.0*error / len(xlist)

	# HW14: count Eout  0.316
	test_xlist, test_ylist = readdata(testfile)
	error = 0
	for i in range(len(test_xlist)):
		predict_y = knn(k, xlist, ylist, test_xlist[i]) 
		if predict_y != test_ylist[i]:
			error += 1
	print "Eout :", 1.0*error / len(test_xlist)
	
	# HW 16 : count Ein with uniform_g 
	error = 0
	for i in range(len(xlist)):
		predict_y = uniform_g(xlist, ylist, xlist[i], gamma) 
		if predict_y != ylist[i]:
			error += 1
	print "Ein with uniform_g :", 1.0*error / len(xlist)
	
	 
if __name__ == '__main__':
    main()
