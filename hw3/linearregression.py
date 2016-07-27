import numpy as np
import random
import math

def sign(value):
	if (value > float(0)):
		return 1
	else:
		return -1

def generatedata(N,transformornot):
	# generate data 
	if (transformornot is True):
		featuresize=6
		data = np.zeros((N,featuresize)) # list of data points
		output = np.zeros(N) # list of output value
		for i in range(N):
			data[i][0] = 1
			data[i][1] = random.uniform(-1,1)  # x1
			data[i][2] = random.uniform(-1,1)  # x2
			data[i][3] = data[i][1]*data[i][2] # x1*x2
			data[i][4] = data[i][1]*data[i][1] # x1^2
			data[i][5] = data[i][1]*data[i][1] # x2^2
			output[i] = sign( math.pow(data[i][0],2)+math.pow(data[i][1],2)-0.6) # y
		# produce noise on 10% data
		for i in sorted(random.sample(xrange(len(data)), int(0.1*float(N)) )):
			output[i] = -( output[i] )

	elif (transformornot is False):
		featuresize=2
		data = np.zeros((N,featuresize)) # list of data points
		output = np.zeros(N) # list of output value
		for i in range(N):
			data[i][0] = random.uniform(-1,1)  # x1
			data[i][1] = random.uniform(-1,1)  # x2
			output[i] = sign( math.pow(data[i][0],2)+math.pow(data[i][1],2)-0.6) # y
		# produce noise on 10% data
		for i in sorted(random.sample(xrange(len(data)), int(0.1*float(N)) )):
			output[i] = -( output[i] )
	return data, output	
		

# compute Wlin using Moore-Penrose pseudo-inverse of a matrix	
def FindWlin(N,data,output):
	for i in range(N):
		Xinv = np.linalg.pinv(data)
		Wlin = np.dot ( Xinv, output )
	return Wlin


def CountEin(N,ITERATION,data,output,Wlin):
	Einlist = []
	for i in range(ITERATION):
		error = float(0)
		for i in range(N):
			h = sign ( np.dot( Wlin, data[i] ) )
			if h != output[i]:
				error += max(  0, 1- np.dot( output[i] , np.dot(Wlin,data[i]) )  )
		Einlist.append( error / float(N) )
	Ein = float( np.sum(np.asarray(Einlist)) ) / float(1000)
	return Ein  
	# Ein is 0.498989943748 : to HW2 Q 14

def CountEout(N,ITERATION,Wlin,transformornot):
	Eoutlist = [] 
	for i in range(ITERATION):
		testdata, testoutput = generatedata(N, transformornot)
		error = float(0)
		for i in range(N):
			h = sign ( np.dot( Wlin, testdata[i] ) )
			if h != testoutput[i]:
				error += max(0 , 1-np.dot(testoutput[i] , np.dot(Wlin , testdata[i])))
		Eoutlist.append(error / float(N))
	Eout = float( np.sum(np.asarray(Eoutlist)) ) / float(1000)
	return Eout  


def NoTransformRegression(N,ITERATION,transformornot):
	data, output = generatedata(N,transformornot)
	Wlin = FindWlin(N,data,output)
	Ein = CountEin(N,ITERATION,data,output,Wlin)
	Eout = CountEout(N,ITERATION,Wlin,transformornot)
	return Ein, Eout 

def TransformRegression(N,ITERATION,transformornot):
	data, output = generatedata(N,transformornot)
	Wlin = FindWlin(N,data,output)
	Ein = CountEin(N,ITERATION,data,output,Wlin)
	Eout = CountEout(N,ITERATION,Wlin,transformornot)
	return Ein, Eout 


if __name__ == '__main__':
	# HW13
	N = 1000
	ITERATION = 1000
	Ein, Eout = NoTransformRegression(N, ITERATION, transformornot = False)
	print "Ein is :", Ein
	print "Eout is :", Eout
	
	# HW14
	N = 1000
	ITERATION = 1000
	Ein, Eout = NoTransformRegression(N, ITERATION, transformornot = True)
	print "Ein is :", Ein
	print "Eout is :", Eout
