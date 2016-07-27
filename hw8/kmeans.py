import numpy as np
from scipy.spatial import distance
import math
import random

trainfile = open("/home/chiahsuan/ml/foundation/hw8/hw8_nolabel_train.dat")
k = 5 
T = 500

def readdata(filename):
	xlist = []
	for line in filename.readlines():
		x= map(float, line.strip().split(" "))
		xlist.append(np.array(x))
	return np.array(xlist)	

def init_centers(xlist, k):
	centers = random.sample(xlist, k)	
	return centers

# memberlist stores set for each center
def partition(xlist, centers):
	memberlist = [ [] for i in range(k) ]
	for i in range(len(xlist)):
		distancelist = []
		for j in range(k):
			distancelist.append( distance.euclidean(xlist[i], centers[j]))
		u = distancelist.index(min(distancelist))
		memberlist[u].append(i)
	return memberlist

def updatecenters(xlist, memberlist):
	centers = []
	for i in range(k):
		centers.append( sum(xlist[ memberlist[i]]) / len(memberlist[i]))
	return centers

def main():
	xlist = readdata(trainfile)
	centers = init_centers(xlist, k)
	for t in range(T):
		memberlist = partition(xlist, centers)
		centers = updatecenters(xlist, memberlist)		
	errorlist = []
	for i in range(len(xlist)):
		for j in range(k):
			if i in memberlist[j]:
				centerindex = j
		errorlist.append( distance.euclidean(xlist[i], centers[centerindex]))
	error = 1.0*sum(errorlist)/len(errorlist)
	print "k =5, T = 500 , error is :", error 	
	# HW 19,20  K = 5 , ERROR IS 1.42195159891

if __name__ == '__main__':
    main()	
