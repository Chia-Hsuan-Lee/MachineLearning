#encoding=utf8
import numpy as np
import math
import random

trainfile = open("/home/chiahsuan/ml/foundation/hw2/hw2_train.dat")
testfile = open("/home/chiahsuan/ml/foundation/hw2/hw2_test.dat")

def generatedata(time_seed):
	np.random.seed(time_seed)
	raw_x = np.sort( np.random.uniform(-1, 1, 20))
	noise_y = np.sign(raw_x) * np.where(np.random.random(raw_x.shape[0])<0.2,-1,1)
	return raw_x, noise_y

def readdata(filename):
	xlist = []
	ylist = []
	for line in filename.readlines():
		x = line.strip().split(" ")[0:-1]
		x = np.array(map(float, x))
		xlist.append(x)
		y = float( line.strip().split(" ")[-1] )
		ylist.append(y)
	return np.array(xlist), np.array(ylist)
		
# h(x) = s * sign (x - theta)
def stump(x, y):
	# there are 20 points, in between there are 19 intervals, and between -inf and inf there are two, so there are 21 intervals total
	#usually theta is the middle of the range
	thetaset = np.array( [ float("-inf")]+[ (x[i]+x[i+1])/2 for i in range(0, x.shape[0]-1) ]+[float("inf") ] )
	s = 1
	target_theta = 0.0
	current_Ein = x.shape[0]
	for theta in thetaset:
		y_pos = np.where(x>theta, 1, -1)
		y_neg = np.where(x<theta, 1, -1)	
		e_pos = sum(y_pos!=y)
		e_neg = sum(y_pos!=y)
		if e_pos > e_neg:
			if e_neg < current_Ein:
				current_Ein = e_neg
				target_theta = theta
				s = -1
		else:
			if e_pos < current_Ein:
				current_Ein = e_pos
				target_theta = theta
				s = 1
	if target_theta == float("inf"):
		target_theta = 1.0
	if target_theta == float("-inf"):
		target_theta = -1.0
	Einrate = float(current_Ein)/x.shape[0]
	return target_theta, s, Einrate 

def CountEout(theta, s):
	Eoutrate = 0.5+0.3*s*(abs(theta)-1)
	return Eoutrate

def main():
	'''	
	# HW 17
	Einlist = []
	Eoutlist = []
	for i in range(5000):
		raw_x, noised_y = generatedata(i)
		theta, s, Einrate = stump (raw_x, noised_y)
		Eoutrate = CountEout(theta, s)
		Einlist.append(Einrate)
		Eoutlist.append(Eoutrate)
	Ein = sum(Einlist)/float(5000)
	Eout = sum(Eoutlist)/float(5000)
	print Ein # 0.17049 
	print Eout # 0.25216906704
	'''
	# HW 18
	xlist, ylist = readdata(trainfile)
	target_theta = 1.0
	target_s = 1 
	current_Ein = float("inf")
	for i in range(xlist.shape[1]):
		raw_x = np.sort(xlist[:,i].transpose())
		raw_y = ylist
		raw_y = np.array([ y for (x, y) in sorted(zip(raw_x, raw_y)) ])
		theta, s, Einrate = stump(raw_x, raw_y)
		if Einrate < current_Ein:
			current_Ein = Einrate
			target_theta = theta 
			target_s = s
			index = i 
	print "decision stump : theta is %d, sign is %d" %(target_theta, target_s)
	print "Ein", current_Ein # 0.44
	test_xlist, test_ylist = readdata(testfile)
	test_xlist = test_xlist[:,index]
	predict_y = float(target_s) * np.sign( test_xlist-target_theta )
	Eout = sum(predict_y != test_ylist) / float(len(test_ylist))
	print "Eout is : ", Eout # 0.466
		

	
	
if __name__ == '__main__':
	main()

