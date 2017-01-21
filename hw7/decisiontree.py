#encoding=utf-8
import numpy as np
import random
import math
# CART : a recursive algorithm that uses decision stump to split data . 
# There are two loops in this algorithm , at each node we need to compare each dimension's B and then in each dimension we need to find the best theta to split the data

trainfile = open("hw7_train.dat")
testfile = open("hw7_test.dat")

class Treenode:
	def __init__(self, theta, index, B, left=None, right=None, parent=None):
		self.theta = theta 
		self.index = index
		self.B = B
		self.sign = 0.0
		self.left = None
		self.right = None

def readdata(filename):
	xlist = []
	ylist = []
	for line in filename.readlines():
		x = line.strip().split(" ")[0:-1]
		x = np.array(map(float, x))
		xlist.append(x)
		y = float(line.strip().split(" ")[-1])
		ylist.append(y)
	return np.array(xlist), np.array(ylist)

# the input data has to be sorted
def indimension_stump(x, y):
	thetaset = np.array([ (x[i]+x[i+1])/2 for i in range(0, x.shape[0]-1) ] )
	target_theta = 0.0
	indim_B = float("inf")
	for theta in thetaset:
		left_y = y[np.where(x>theta)]
		right_y = y[np.where(x<theta)]
		b = left_y.shape[0]*count_gini(left_y) + right_y.shape[0]*count_gini(right_y) 
		if indim_B > b:
			indim_B = b
			target_theta = theta
	return target_theta, indim_B

# compare B from different dimension and split the data into two branches
def comparedimension_stump(X, Y):
	sorted_index = [ np.argsort(X[:,i]) for i in range(X.shape[1])]
	B = float("inf")
	target_theta = 0.0
	index = 0
	for i in range(X.shape[1]):
		x = X[sorted_index[i],i]
		y = Y[sorted_index[i]]
		indim_theta, indim_B = indimension_stump(x, y)
		if indim_B < B:
			B = indim_B
			target_theta = indim_theta
			index = i
	right_X = X[ np.where(X[:,index] > target_theta) ]
	right_Y = Y[ np.where(X[:,index] > target_theta) ]
	left_X =  X[ np.where(X[:,index] < target_theta) ]
	left_Y =  Y[ np.where(X[:,index] < target_theta) ]
	return right_X, right_Y, left_X, left_Y, B, target_theta, index
		 
# Gini - index : 1 - sigma (each class's data proportion)^2
def count_gini(y):
	if (y.shape[0]==0):
		return 0
	n1 = sum(y==1.0)
	n2 = sum(y==-1.0)
	gini = 1- ( math.pow(1.0*n1/(n1+n2),2) + math.pow(1.0*n2/(n1+n2),2))
	return float(gini)

# CART : bi-branching by purifying 
# return a binary search tree according to the learned model 
# Notice that there has to be a stoping criteria
def CART(X, Y):
	if (X.shape[0]==0): # there are no data left to be split
		return None
	if (count_gini(Y)==0.0): # there is only one category
		node = Treenode(theta=0, index=-1, B=-100)
		node.sign = 1.0 if Y[0]==1.0 else -1.0
		return node
	right_X, right_Y, left_X, left_Y, B, target_theta, index = comparedimension_stump(X, Y)
	node = Treenode(target_theta, index, B)	
	node.left = CART(left_X, left_Y)
	node.right = CART(right_X, right_Y)
	return node

def random_forest(x, y, T):
	trees = []
	for i in range(T):
		num = range(x.shape[0])
		index = np.random.choice(num, size=x.shape[0], replace=True)
		xi = x[index]
		yi = y[index]
		node = CART(xi,yi)
		trees.append(node)	 
	return trees

# HW 18
def Eout_RF(trees):
	errorlist = []
	test_xlist, test_ylist = readdata(testfile)
	for i in range(test_xlist.shape[0]):
		pos = 0
		neg = 0
		for tree in trees:
			predict_y = predict(tree, test_xlist[i])
			if (predict_y == 1.0):
				pos+=1
			else:
				neg+=1
		if (pos>neg):
			final_predict = 1.0
		else:
			final_predict = -1.0
		if (final_predict == test_ylist[i]):
			errorlist.append(0)
		else:
			errorlist.append(1)
	Eout = 1.0*sum(errorlist)/len(errorlist)
	return Eout

def predict(root, x):
	if (root.B == -100):
		return root.sign
	if (x[root.index]>root.theta):
		if (root.right) is None:
			return (-1.0)*predict(root.left, x)
		return predict(root.right, x)
	else:
		if (root.left) is None:
			return (-1.0)*predict(root.right, x)
		return predict(root.left, x)
		
def Count_Eout(root):
	test_xlist, test_ylist = readdata(testfile)
	errorlist = []
	for i in range(test_xlist.shape[0]):
		if ( predict(root, test_xlist[i]) != test_ylist[i] ):
			errorlist.append(1)
		else:
			errorlist.append(0)
	return 1.0*sum(errorlist)/len(errorlist)
		
def main():
	'''
	# Implement the simple C&RT algorithm without pruning using the Gini index as the impurity measure
	# For the decision stump used in branching, if you are branching with feature i and direction s, please sort all the x n,i values to form (at most) N + 1 segments of equivalent θ, and then pick θ within the median of the segment.
	xlist, ylist = readdata(trainfile)
	root = CART(xlist, ylist)
	# HW 15
	Eout = Count_Eout(root)
	print "Eout is : ", Eout
	'''
	# HW 18 T = 30000 Eout is 0.071
	T = 30000
	xlist, ylist = readdata(trainfile)
	trees = random_forest(xlist, ylist, T)	
	Eout = Eout_RF(trees)
	print "RF with T = 30000 is :", Eout
	
if __name__ == "__main__":
	main()
		
		
