import numpy as np

  #################################################
 ### Code to read vocab file and build matrix ####
#################################################
def read_mult(f_in='data/mult.dat',D=8000,ndocs=16980):
    fp = open(f_in)
    lines = fp.readlines()
    X = np.zeros((len(lines),D))
    #Populate X matrix from mult.dat
    for i,line in enumerate(lines):
        strs = line.strip().split(' ')[1:]
        for strr in strs:
            segs = strr.split(':')
            X[i,int(segs[0])] = float(segs[1])
    #IDF Calculation log2(N/DF)
    Y = np.log2(ndocs/np.count_nonzero(X,axis=0))
    #TF Calculation 1+log2(TF)
    X = 1+np.ma.log2(X)
    X = X.filled(0)
    #TF-IDF
    X = X*Y
    #Normalize
    arr_max = np.amax(X,axis=1)
    X = (X.T/arr_max).T
    return X

#######
#def read_mult(f_in='data/citeulike-a/mult.dat',D=8000):
#    fp = open(f_in)
#    lines = fp.readlines()
#    X = np.zeros((len(lines),D))
#    for i,line in enumerate(lines):
#        strs = line.strip().split(' ')[1:]
#        for strr in strs:
#            segs = strr.split(':')
#            X[i,int(segs[0])] = float(segs[1])
#    arr_max = np.amax(X,axis=1)
#    X = (X.T/arr_max).T
#    return X
#######
