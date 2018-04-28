# DeepRec
Collaborative Deep Learning for Rec. Systems

Environment Dependencies:
=========================
Python 2.7 <br>
MXNet <br>

Library Dependencies:
=====================
getopt <br>
matplotlib <br>
numpy <br>
pickle <br>
sys <br>

Directory Structure:
====================
data - contains all the necessary input data files. Each dataset directory has separate readme file describing each file <br>
results - directory where we saved our recall values and also various comparision plots <br>
experiments - contains files related to all the experiments we did <br>

src code: <br>
=======
Dataset creation <br>
-----------------
generate_datasets.py : To generate the datasets with given sparseness settings <br>
mult.py: Reads the item-vocab files and feeds to the model as a matrix input<br>
data.py: Wrapper for data.py<br>

Evaluation 
-----------
evaluate_CDL.py: To calculate recall values <br>
#To calculate recall for citeulike-a(d=a) with dense setting(p=10) and number of encoder layers(l=2)<br>
**python evaluate_CDL.py -p 10 -l 2 -d a** <br>

cal_precision.py : To calculate mean Average Precision mAP <br>
#To calculate mAP for citeulike-a(d=a) with dense setting(p=10) and number of encoder layers(l=2)<br>
#c defines where to cut off the recommended articles <br>
**python cal_precision.py -p 10 -l 2 -d a -c 250** <br>

show_recommendation.py : To show the recommendations for a particular user<br>
#To display recommendations for userid 8 (u=8) citeulike-a(d=a) with dense setting(p=10) and number of encoder layers(l=2)<br>
**python show_recommendation.py -p 10 -l 2 -d a -u 8** <br>

Main Model
----------
autoencoder.py : Stacked Denoising AutoEncoder Model (SDAE)  <br>
BCD_one.py: Block Coordinate Descent optimization algorithm <br>
cdl.py : Main Collaborative Deep Learning (CDL) wrapper file <br>
model.py: Feature extraction and preparing data for SDAE <br>
solver.py: Finutune and parameter update <br>

Run:
---
Case 1: <br>
For citeulike-a dataset (d=a) , with dense setting (P=10) and two encoder layers in SDAE(l=2) <br>
**python cdl.py -p 10 -l 2 -d a** <br>

Case 2: <br>
For citeulike-t dataset (d=t) , with dense setting (P=3) and two encoder layers in SDAE(l=2) <br>
**python cdl.py -p 3 -l 2 -d t** <br>

Directory Tree: (output of 'tree -L 3')
-------------------
```bash
.
├── autoencoder.py
├── BCD_one.py
├── cal_precision.py
├── cdl.py
├── data
│   ├── citeulike-a
│   │   ├── mult.dat
│   │   ├── P1
│   │   ├── P10
│   │   ├── P5
│   │   ├── raw_inputs
│   │   └── README.txt
│   └── citeulike-t
│       ├── mult.dat
│       ├── P1
│       ├── P10
│       ├── P3
│       ├── raw_inputs
│       └── README.txt
├── data.py
├── evaluate_CDL.py
├── experiments
│   ├── citeulike-a
│   │   ├── L2_P1
│   │   ├── L2_P10
│   │   ├── L2_P5
│   │   ├── L3_P1
│   │   ├── L3_P10
│   │   └── L3_P5
│   └── citeulike-t
│       ├── L2_P1
│       ├── L2_P3
│       ├── L3_P1
│       └── L3_P3
├── generate_datasets.py
├── model.py
├── mult.py
├── README.md
├── results
│   ├── plot_citeulike_a.py
│   ├── plot_citeulike_t.py
│   ├── plots
│   │   ├── citeulike-a
│   │   ├── citeulike-t
│   │   └── reco.png
│   └── recall
│       ├── baseline
│       └── tf-idf
├── show_recommendation.py
└── solver.py
```
Git references:
==============
- https://github.com/js05212/CDL
- https://github.com/akash13singh/mxnet-for-cdl
- https://github.com/zoujun123/Keras-CDL
