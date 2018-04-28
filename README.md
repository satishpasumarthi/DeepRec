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
data - contains all the necessary input data files. Each dataset directory has a separate readme file describing each file <br>
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
python evaluate_CDL.py -p 10 -l 2 -d a <br>

cal_precision.py : To calculate mean Average Precision mAP <br>
#To calculate mAP for citeulike-a(d=a) with dense setting(p=10) and number of encoder layers(l=2)<br>
#c defines where to cut off the recommended articles <br>
python evaluate_CDL.py -p 10 -l 2 -d a -c 250 <br>

show_recommendation.py : To show the recommendations for a particular user<br>
#To display recommendations for userid 8 (u=8) citeulike-a(d=a) with dense setting(p=10) and number of encoder layers(l=2)<br>
python show_recommendation.py -p 10 -l 2 -d a -u 8<br>

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
python cdl.py -p 10 -l 2 -d a <br>

Case 2: <br>
For citeulike-t dataset (d=t) , with dense setting (P=3) and two encoder layers in SDAE(l=2) <br>
python cdl.py -p 3 -l 2 -d t

Directory Structure:
-------------------

├── autoencoder.py
├── autoencoder.pyc
├── BCD_one.py
├── BCD_one.pyc
├── cal_precision.py
├── cdl.py
├── data
│   ├── citeulike-a
│   │   ├── mult.dat
│   │   ├── P1
│   │   │   ├── test_P1_1.dat
│   │   │   ├── test_P1_2.dat
│   │   │   ├── test_P1_3.dat
│   │   │   ├── train_P1_1.dat
│   │   │   ├── train_P1_2.dat
│   │   │   └── train_P1_3.dat
│   │   ├── P10
│   │   │   ├── test_P10_1.dat
│   │   │   ├── test_P10_2.dat
│   │   │   ├── test_P10_3.dat
│   │   │   ├── train_P10_1.dat
│   │   │   ├── train_P10_2.dat
│   │   │   └── train_P10_3.dat
│   │   ├── P5
│   │   │   ├── test_P5_1.dat
│   │   │   ├── test_P5_2.dat
│   │   │   ├── test_P5_3.dat
│   │   │   ├── train_P5_1.dat
│   │   │   ├── train_P5_2.dat
│   │   │   └── train_P5_3.dat
│   │   ├── raw_inputs
│   │   │   ├── raw-data.csv
│   │   │   ├── users.dat
│   │   │   └── vocabulary.dat
│   │   └── README.txt
│   └── citeulike-t
│       ├── mult.dat
│       ├── P1
│       │   ├── test_P1_1.dat
│       │   ├── test_P1_2.dat
│       │   ├── test_P1_3.dat
│       │   ├── train_P1_1.dat
│       │   ├── train_P1_2.dat
│       │   └── train_P1_3.dat
│       ├── P10
│       │   ├── test_P10_1.dat
│       │   ├── test_P10_2.dat
│       │   ├── test_P10_3.dat
│       │   ├── train_P10_1.dat
│       │   ├── train_P10_2.dat
│       │   └── train_P10_3.dat
│       ├── P3
│       │   ├── test_P3_1.dat
│       │   ├── test_P3_2.dat
│       │   ├── test_P3_3.dat
│       │   ├── train_P3_1.dat
│       │   ├── train_P3_2.dat
│       │   └── train_P3_3.dat
│       ├── raw_inputs
│       │   ├── citations.dat
│       │   ├── rawtext.dat
│       │   ├── users.dat
│       │   └── vocabulary.dat
│       └── README.txt
├── data.py
├── data.pyc
├── evaluate_CDL.py
├── experiments
│   ├── citeulike-a
│   │   ├── L2_P1
│   │   │   ├── final-theta.dat
│   │   │   ├── final-U.dat
│   │   │   ├── final-V.dat
│   │   │   └── rec-list.dat
│   │   ├── L2_P10
│   │   │   ├── cdl.log
│   │   │   ├── final-theta.dat
│   │   │   ├── final-U.dat
│   │   │   ├── final-V.dat
│   │   │   └── rec-list.dat
│   │   ├── L2_P5
│   │   │   ├── final-theta.dat
│   │   │   ├── final-U.dat
│   │   │   ├── final-V.dat
│   │   │   └── rec-list.dat
│   │   ├── L3_P1
│   │   │   ├── final-theta.dat
│   │   │   ├── final-U.dat
│   │   │   ├── final-V.dat
│   │   │   └── rec-list.dat
│   │   ├── L3_P10
│   │   │   ├── final-theta.dat
│   │   │   ├── final-U.dat
│   │   │   ├── final-V.dat
│   │   │   └── rec-list.dat
│   │   └── L3_P5
│   │       ├── final-theta.dat
│   │       ├── final-U.dat
│   │       ├── final-V.dat
│   │       └── rec-list.dat
│   └── citeulike-t
│       ├── L2_P1
│       │   ├── final-theta.dat
│       │   ├── final-U.dat
│       │   ├── final-V.dat
│       │   └── rec-list.dat
│       ├── L2_P3
│       │   ├── cdl.log
│       │   ├── final-theta.dat
│       │   ├── final-U.dat
│       │   ├── final-V.dat
│       │   └── rec-list.dat
│       ├── L3_P1
│       │   ├── final-theta.dat
│       │   ├── final-U.dat
│       │   ├── final-V.dat
│       │   └── rec-list.dat
│       └── L3_P3
│           ├── final-theta.dat
│           ├── final-U.dat
│           ├── final-V.dat
│           └── rec-list.dat
├── generate_datasets.py
├── model.py
├── model.pyc
├── mult.py
├── mult.pyc
├── README.md
├── results
│   ├── plot_citeulike_a.py
│   ├── plot_citeulike_t.py
│   ├── plots
│   │   ├── citeulike-a
│   │   │   ├── BOW_TFIDF_L2.png
│   │   │   ├── L2.png
│   │   │   └── L3.png
│   │   ├── citeulike-t
│   │   │   ├── BOW_TFIDF_L2.png
│   │   │   ├── L2.png
│   │   │   └── L3.png
│   │   └── reco.png
│   └── recall
│       ├── baseline
│       │   ├── citeulike_a_BOW_L2_P10.txt
│       │   └── citeulike_t_BOW_L2_P3.txt
│       └── tf-idf
│           ├── citeulike-a
│           │   ├── citeulike_a_L2_P10.txt
│           │   ├── citeulike_a_L2_P1.txt
│           │   ├── citeulike_a_L2_P5.txt
│           │   ├── citeulike_a_L3_P10.txt
│           │   ├── citeulike_a_L3_P1.txt
│           │   └── citeulike_a_L3_P5.txt
│           └── citeulike-t
│               ├── citeulike_t_L2_P1.txt
│               ├── citeulike_t_L2_P3.txt
│               ├── citeulike_t_L3_P1.txt
│               └── citeulike_t_L3_P3.txt
├── show_recommendation.py
├── solver.py
└── solver.pyc

