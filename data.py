import numpy as np
import os
from mult import read_mult
import sys

def downloadData():
    data_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/cdl'
    for filename in ('mult.dat', 'train_P10_1.dat', 'test_P10_1.dat', 'raw-data.csv'):
        if not os.path.exists(filename):
            os.system("wget %s/%s" % (data_url, filename))

def get_mult(mult_file,vocab_size,ndocs):
    if not os.path.exists(mult_file):
        print "data.py:: get_mult File does not exist."
        sys.exit(2)
    X = read_mult(mult_file,vocab_size,ndocs).astype(np.float32)
    return X

def get_dummy_mult():
    X = np.random.rand(100,100)
    X[X<0.9] = 0
    return X

def read_user(f_in='data/citeulike-a/P10/train_P10_1.dat',num_u=5551,num_v=16980):
    if not os.path.exists(f_in):
        #downloadData()
        print("datapy:: read_user File Not present")
        sys.exit(2)
    fp = open(f_in)
    R = np.mat(np.zeros((num_u,num_v)))
    for i,line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        for seg in segs:
            R[i,int(seg)] = 1
    return R

def read_dummy_user():
    R = np.mat(np.random.rand(100,100))
    R[R<0.9] = 0
    R[R>0.8] = 1
    return R

