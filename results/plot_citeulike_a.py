import matplotlib.pyplot as plt
import numpy as np

M_low = 50
M_high = 300

f1 = np.loadtxt("recall/tf-idf/citeulike-a/citeulike_a_L2_P1.txt")
f2 = np.loadtxt("recall/tf-idf/citeulike-a/citeulike_a_L2_P5.txt")
f3 = np.loadtxt("recall/tf-idf/citeulike-a/citeulike_a_L2_P10.txt")
plt.plot(range(M_low,M_high+1),f1)
plt.plot(range(M_low,M_high+1),f2)
plt.plot(range(M_low,M_high+1),f3)
plt.ylabel("Recall")
plt.xlabel("M")
plt.title("CDL: Recall@M")
plt.legend(['P = 1(Sparse)', 'P = 10(Dense)'], loc='upper left')
plt.show()
