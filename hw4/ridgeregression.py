import numpy as np
import math

trainfile = open("/home/chiahsuan/ml/foundation/hw4/hw4_train.dat").readlines()
testfile = open("/home/chiahsuan/ml/foundation/hw4/hw4_test.dat").readlines()

def loaddata(filename):
	X = []
	Y = []
	for line in filename:
		x =  map(float,line.split(" ")[0:-1]) 
		X.append( np.asarray(x) )
		Y.append( float(line.split(" ")[-1].strip()))
	X = np.asarray(X)
	X = np.hstack((np.ones(X.shape[0]).reshape(-1,1),X))
	return X, np.asarray(Y)

def validation(X,Y): # 120:80
	train_X = X[0:120]
	train_Y = Y[0:120]
	val_X = X[120:200]
	val_Y = Y[120:200]
	return np.array(train_X), np.array(train_Y), np.array(validation_X), np.array(validation_Y)

def CVERR(X, Y, V, lamda):
	totalCV = 0
	for i in range(V):
		train_X = []
		train_Y = []
		val_X = np.array( X[ i*X.shape[0]/V : (i+1)*X.shape[0]/V  ] )
		val_Y = np.array( Y[ i*Y.shape[0]/V : (i+1)*Y.shape[0]/V  ] )
		for j in range(V):
                	if j!=i :
                    		train_X.extend( X[range(j*X.shape[0]/V, (j+1)*X.shape[0]/V) ].tolist() )
                    		train_Y.extend( Y[range(j*Y.shape[0]/V, (j+1)*Y.shape[0]/V) ].tolist() )
		train_X = np.array(train_X)
		train_Y = np.array(train_Y)
		w = train(train_X, train_Y, math.pow(10, lamda))
		Ecv = CountError(val_X, val_Y, w)
		totalCV += Ecv
	return float(totalCV) / V

# w = ( x^Tx + labdaI )^(-1) X^T Y
def train(X, Y, lamda):
	inverse = np.linalg.inv( np.dot(X.transpose(),X) + lamda * np.eye(X.shape[1], dtype = float) )
	w = np.dot( inverse, np.dot( X.transpose(), Y) )
	return w 

def CountError(test_X, test_Y, w):
	scores = np.dot(w, test_X.transpose())
	predicts = np.where(scores >= 0, 1.0, -1.0)
	Eout = sum(predicts!=test_Y)
	return (Eout*1.0) / predicts.shape[0]
	
def main():
	'''
	# HW 13
	lamda = 11.26
	X, Y = loaddata(trainfile)
	w = train(X, Y, lamda)
	test_X, test_Y = loaddata(testfile)
	Ein = CountError(X, Y, w)
	Eout = CountError(test_X, test_Y, w)
	print Ein   # 0.055
	print Eout  # 0.052
	# HW 14
	X, Y = loaddata(trainfile)
	test_X, test_Y = loaddata(testfile)
	min_Ein = 1
	min_Eout = 1
	target_lamda = 2
	lamdaset = [ i for i in range(2, -11, -1) ]
	for lamda in lamdaset:
		w = train(X, Y, math.pow(10, lamda))
		Ein = CountError(X, Y, w)
		Eout = CountError(test_X, test_Y, w)
		if Ein < min_Ein:
			min_Ein = Ein 
			min_Eout = Eout
			target_lamda  = lamda 
	print min_Ein  # 0.015
	print min_Eout # 0.02 
	print target_lamda # lamda = -8
	
	# HW 15
	X, Y = loaddata(trainfile)
	test_X, test_Y = loaddata(testfile)
	min_Ein = 1
	min_Eout = 1
	target_lamda = 2
	lamdaset = [ i for i in range(2, -11, -1) ]
	for lamda in lamdaset:
		w = train(X, Y, math.pow(10, lamda))
		Ein = CountError(X, Y, w)
		Eout = CountError(test_X, test_Y, w)
		if Eout < min_Eout:
			min_Ein = Ein 
			min_Eout = Eout
			target_lamda  = lamda 
	print min_Ein  # 0.03
	print min_Eout # 0.015 
	print target_lamda # lamda = -7
	
	# HW 16 	
	X, Y = loaddata(trainfile)
	test_X, test_Y = loaddata(testfile)
	train_X, train_Y, val_X, val_Y = validation(X, Y)
	
	min_Ein = 1
	min_Eval = 1
	min_Eout = 1
	target_lamda = 2
	lamdaset = [ i for i in range(2, -11, -1) ]
	for lamda in lamdaset:
		w = train(train_X, train_Y, math.pow(10, lamda))
		Ein = CountError(train_X, train_Y, w)
		Eval = CountError(val_X, val_Y, w)
		Eout = CountError(test_X, test_Y, w)
		if Ein < min_Ein:
			min_Ein = Ein 
			min_Eval = Eval 
			min_Eout = Eout
			target_lamda  = lamda 
	print min_Ein  # 0.0
	print min_Eval  # 0.05
	print min_Eout # 0.025 
	print target_lamda # lamda = -8
	
	# HW 17	
	X, Y = loaddata(trainfile)
	test_X, test_Y = loaddata(testfile)
	train_X, train_Y, val_X, val_Y = validation(X, Y)
	
	min_Ein = 1
	min_Eval = 1
	min_Eout = 1
	target_lamda = 2
	lamdaset = [ i for i in range(2, -11, -1) ]
	for lamda in lamdaset:
		w = train(train_X, train_Y, math.pow(10, lamda))
		Ein = CountError(train_X, train_Y, w)
		Eval = CountError(validation_X, validation_Y, w)
		Eout = CountError(test_X, test_Y, w)
		if Eval < min_Eval:
			min_Ein = Ein 
			min_Eval = Eval 
			min_Eout = Eout
			target_lamda  = lamda 
	print min_Ein  # 0.0333
	print min_Eval  # 0.0375
	print min_Eout # 0.028
	print target_lamda # lamda = 0

	# HW 18
	lamda = 0
	X, Y = loaddata(trainfile)
	test_X, test_Y = loaddata(testfile)
	w = train(X, Y, math.pow(10, lamda))
	Ein = CountError(X, Y, w)
	Eout = CountError(test_X, test_Y, w)
	print Ein  # 0.035
	print Eout # 0.02

	X, Y = loaddata(trainfile)
	test_X, test_Y = loaddata(testfile)
	train_X, train_Y, val_X, val_Y = validation(X, Y)
	
	min_Ein = 1
	min_Eval = 1
	min_Eout = 1
	target_lamda = 2
	lamdaset = [ i for i in range(2, -11, -1) ]
	for lamda in lamdaset:
		w = train(train_X, train_Y, math.pow(10, lamda))
		Ein = CountError(train_X, train_Y, w)
		Eval = CountError(val_X, val_Y, w)
		Eout = CountError(test_X, test_Y, w)
		if Eval < min_Eval:
			min_Ein = Ein 
			min_Eval = Eval 
			min_Eout = Eout
			target_lamda  = lamda 
	print min_Ein  # 0.0333
	print min_Eval  # 0.0375
	print min_Eout # 0.028
	print target_lamda # lamda = 0
	'''	
	
	# HW 19 
	V = 5 # fiveflod validation	
	X, Y = loaddata(trainfile)
	min_Ecv = 1
	target_lamda = 2
	lamdaset = [ i for i in range(2, -11, -1) ]
	for lamda in lamdaset:
		Ecv = CVERR(X, Y, V, lamda)
		if Ecv < min_Ecv:
			min_Ecv = Ecv
			target_lamda  = lamda 
	print min_Ecv # 0.03
	print target_lamda # lamda = -8
	
	# HW 20
	lamda = -8 
	X, Y = loaddata(trainfile)
	test_X, test_Y = loaddata(testfile)
	w = train(X, Y, math.pow(10, lamda))
	Ein = CountError(X, Y, w)
	Eout = CountError(test_X, test_Y, w)
	print Ein
	print Eout

	
if __name__ == '__main__':
	main()
