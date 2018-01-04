import numpy as np
import math
import matplotlib.pyplot as plt

trainfile = open("./hw3_train.dat").readlines()
testfile = open("./hw3_test.dat").readlines()

def loaddata(filename):
    X = []
    Y = []
    for line in filename:
        x =  line.split(" ")[1:-1]
        x = [ float(i) for i in x ]
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
        secondterm = np.multiply((-1.0) * Y[i],X[i])
        sumein += theta * secondterm 
    gradient = sumein / N
    return gradient

def logisticregression(ita, ITERATION, stochastic):
    w = np.zeros(20)
    X, Y = loaddata(trainfile)
    index = 0
    Einlist = [] 
    Eoutlist = [] 
    for i in range(ITERATION):
        if stochastic:
            s = (-1.0) * Y[index] * np.dot(w,X[index]) 
            theta = 1.0 / (1 + math.exp(-s))
            secondterm = (-1.0) * Y[index]* X[index]
            #secondterm = np.multiply((-1.0) * Y[index],X[index])
            gradient = theta * secondterm
            w = w - ita * gradient	
            index += 1
            if (index == 1000):
                index = 0
        else:
            gradient = countgradient(X, Y, w)
            w = w - ita * gradient
        Ein = CountEin(w)
        Eout = CountEout(w)
        Einlist.append(Ein)
        Eoutlist.append(Eout)
    return w, Einlist, Eoutlist

def CountEin(w):
    traindata, trainoutput = loaddata(trainfile)
    N = len(trainoutput)
    error = 0.0
    for i in range(N):
        s = np.dot( w, traindata[i] ) 
        scores = 1/ ( 1 + math.exp(-s))
        h = np.where(scores>=0.5, 1.0, -1.0)
        if h != trainoutput[i]:
            error += 1
    Ein = float(error) / N
    return Ein

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
    w , Einlist, Eoutlist= logisticregression(ita, ITERATION, False)
    Eout = CountEout(w)  # 0.471666666667
    print(Eout)
    '''
    # HW 19
    ita = 0.01
    ITERATION = 2000
    w , Einlist, Eoutlist= logisticregression(ita, ITERATION, False)
    Eout = CountEout(w) # 0.220666666667
    print(Eout)

    # HW 20
    ita = 0.001
    ITERATION = 2000
    w , Einlist_sgd, Eoutlist_sgd = logisticregression(ita, ITERATION, True)
    Eout = CountEout(w) # 0.471666666667
    print(Eout)
    t_list = range(ITERATION)
    
    EinFig = plt.figure(1)
    plt.plot(t_list, Einlist, 'r') # plotting t, a separately 
    plt.plot(t_list, Einlist_sgd, 'b') # plotting t, b separately 
    plt.title("Ein for GD(red) and SGD(blue)")
    plt.savefig('Ein.png')

    EoutFig = plt.figure(2)
    plt.plot(t_list, Eoutlist, 'r') # plotting t, a separately 
    plt.plot(t_list, Eoutlist_sgd, 'b') # plotting t, b separately 
    plt.title("Eout for GD(red) and SGD(blue)")
    plt.savefig('Eout.png')

    
    plt.show()

if __name__ == '__main__':
    main()
