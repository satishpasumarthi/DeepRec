import matplotlib.pyplot as plt
import numpy as np

M_low = 50
M_high = 300

f1 = np.loadtxt("recall/tf-idf/citeulike-t/citeulike_t_L2_P1.txt")
f2 = np.loadtxt("recall/tf-idf/citeulike-t/citeulike_t_L2_P3.txt")
plt.plot(range(M_low,M_high+1),f1)
plt.ylabel("Recall")
plt.plot(range(M_low,M_high+1),f2)
plt.xlabel("M")
plt.title("CDL: Recall@M")
plt.legend(['P = 1(Sparse)', 'P = 3(Dense)'], loc='upper left')
plt.show()

