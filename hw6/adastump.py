import numpy as np
import math
import random

# there are three layers in this algorithm 
# 1. 300 iterations      - > mini_Ein
# 2. each dimension of x data as imput  -> E_local (the best in all dimensions)
# 3. in each dimesion, we need to itetate all the theta and sign to find the best -> E_little
trainfile = open("hw6_adaboost_train.dat")
testfile = open("hw6_adaboost_test.dat")

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
def stump(x, y, u_next):  
	# fist part : find the best theta and sign pair  
	thetaset = np.array( [ float("-inf")]+[ (x[i]+x[i+1])/2 for i in range(0, x.shape[0]-1) ]+[float("inf") ] )
	s = 1
	target_theta = 0.0
	E_little = sum(u_next)/sum(u_next)

	for theta in thetaset:
		e_pos = sum((np.where(x>theta, 1, -1) != y) *u_next) / sum (u_next)
		e_neg = sum((np.where(x<theta, 1, -1) != y) *u_next) / sum (u_next)
		if e_pos > e_neg:
			if e_neg < E_little:
				E_little = e_neg
				target_theta = theta
				s = -1
		else:
			if e_pos < E_little:
				E_little = e_pos
				target_theta = theta
				s = 1
	if target_theta == float("inf"):
		target_theta = 1.0
	if target_theta == float("-inf"):
		target_theta = -1.0
	# second part : count the parameters 
	scalingfactor = math.sqrt((1-E_little)/E_little)
	# update the next run of u
	if (s == 1):
		u_next = (np.where(x>target_theta, 1,-1) != y)*u_next*scalingfactor + (np.where(x>target_theta, 1, -1) == y) * u_next / scalingfactor
	else:
		u_next = (np.where(x<target_theta, 1,-1) != y)*u_next*scalingfactor + (np.where(x<target_theta, 1, -1) == y) * u_next / scalingfactor
	alpha = math.log(scalingfactor, math.e)
	return target_theta, s, E_little, u_next, alpha 

def main():
	T = 300  # 300 iterations 
	xlist, ylist = readdata(trainfile)

	for i in range(0, xlist.shape[1]): 
		sorted_index.append(np.argsort(xlist[:,i]))
	alphalist = np.ones(T)
	thetalist = np.ones(T)
	signlist = np.ones(T)
	indexlist = np.ones(T)
	
	N = len(xlist[:,0])
	mini_Ein = float("inf")

	for t in range(T):
		
		E_local = float("inf")
		
		for i in range(xlist.shape[1]):
			raw_x = xlist[sorted_index[i], i]
			raw_y = ylist[sorted_index[i]]
			
			'''
			# preprocess the data : sort x data because stump needs to check the optimized theta
			raw_x = np.sort(xlist[:,i].transpose())
			raw_y = ylist
			raw_y = np.array([ y for (x, y) in sorted(zip(raw_x, raw_y)) ])
			'''
			# stump algorithm 
			theta_i, sign_i, E_little, u_i, alpha_i = stump(raw_x, raw_y, u[sorted_index[i]])
			# check the condition to find the best Ein in one t
			if E_little < E_local:
				E_local = E_little
				alpha_t = alpha_i
			        theta_t = theta_i
			        sign_t = sign_i
			        index_t = int(i)
				u_next = u_i
			# check to find best Ein in all t
				if E_little < mini_Ein:
					mini_Ein = E_little
	
		alphalist[t] = alpha_t
		thetalist[t] = theta_t
		signlist[t] = sign_t
		indexlist[t] = index_t
		ulist[index_t] = u_next
		
		# HW 12 Ein(g1)
		if (t==0):
			if signlist[t]==1:
            			Ein = 1.0*sum(np.where(xlist[:,index_t]>theta_t,1,-1)!=y)/xlist.shape[0]
            		else:
                		Ein = 1.0*sum(np.where(xlist[:,index_t]<theta_t,1,-1)!=y)/xlist.shape[0]
	            	print "Ein1:"+str(Ein)
		# HW 15		
		if (t==0):
			print "U2:"+str(sum(u_next))
		if (t==T-2):
			print "UT:"+str(sum(u_next))

	# Q14 
	predict_y = np.zeros(xlist.shape[0])
	for t in range(0,T):
        	if signlist[t]==1:
            		predict_y = predict_y + alphalist[t]*np.where( xlist[:,indexlist[t]]>thetalist[t], 1, -1 )
        	else:
            		predict_y = predict_y + alphalist[t]*np.where( xlist[:,indexlist[t]]<thetalist[t], 1, -1 )
    	EinG = 1.0*sum( np.where( predict_y>0, 1, -1) != y ) / xlist[:,0].shape[0]
	print "Ein(G):"+str(EinG)

	# Q15
	print "mini error rate:"+str(mini_Ein)

	# Q17 Eoutg1 Q18 EoutG
	test_xlist, test_ylist = readdata(testfile)
	predict_y = np.zeros(test_xlist.shape[0])

	# Q17
	if signlist[0]==1:
        	predict_y = np.where(test_xlist[:,indexlist[0]]>thetalist[0],1,-1)
	else:
		predict_y = np.where(test_xlist[:,indexlist[0]]<thetalist[0],1,-1)
	Eoutg1 = sum(predict_y != test_ylist)
	print "Eout1:" + str(1.0*Eoutg1 / test_xlist.shape[0])

	# Q18
	for t in range(0,T):
		if signlist[t]==1:
			predict_y = predict_y + np.where(test_xlist[:,indexlist[t]]>thetalist[t],1,-1)*alphalist[t]
		else:
 			predict_y = predict_y + np.where(test_xlist[:,indexlist[t]]<thetalist[t],1,-1)*alphalist[t]
	Eout = sum(np.where(predict_y>0,1,-1)!=test_ylist)
	print "Eout:"+str(Eout*1.0/test_xlist.shape[0])

if __name__=='__main__':
	main()
