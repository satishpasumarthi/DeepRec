from __future__ import  division
from data import read_user
import matplotlib.pyplot as plt
import numpy as np
import sys,getopt

   ##########################################
  ### Code to calculate recall and plot ####
 ### Inputs: U and V dat files         ####
##########################################

def cal_rec(p,cut,u_file,v_file,rec_file,test_file):
    R_true = read_user(test_file)
    U = np.mat(np.loadtxt(u_file))
    V = np.mat(np.loadtxt(v_file))
    R = U * V.T

    print "Recommendations shape: "+str(R.shape)
    num_u = R.shape[0]
    num_hit = 0
    fp = open(rec_file, 'w')
    print 'Total Users ' + str(num_u)
    for i in range(num_u):
        if i != 0 and i % 500 == 0:
            print 'Processed ' + str(i) + ' users'
        l_score = R[i, :].A1.tolist()
        pl = sorted(enumerate(l_score), key=lambda d: d[1], reverse=True)
        l_rec = list(zip(*pl)[0])[:cut]
        s_rec = set(l_rec)
        s_true = set(np.where(R_true[i, :] > 0)[1])
        cnt_hit = len(s_rec.intersection(s_true))
        fp.write('%d:' % cnt_hit)
        fp.write(' '.join(map(str, l_rec)))
        fp.write('\n')
    fp.close()

def main(argv):
    # give the same p as given in cdl.py
    try:
      opts, args = getopt.getopt(argv,"hp:l:d:")
    except getopt.GetoptError:
      print 'python evaluate_CDL.py -p <setting_value> -l <num_layers> -d <dataset_type>'
      print '\n'
      print 'Example: For 2 Layer dense setting in citeulike-a dataset'
      print 'python evaluate_CDL.py -p 10 -l 2 -d a'
      print '\n'
      print 'Example: For 2 Layer dense setting in citeulike-t dataset'
      print 'python evaluate_CDL.py -p 3 -l 2 -d t'
      sys.exit(2)
    for opt,arg in opts:
        if opt == '-h':
           print 'python evaluate_CDL.py -p <setting_value> -l <num_layers> -d <dataset_type> -u <userid>'
           print '\n'
           print 'Example: For 2 Layer dense setting in citeulike-a dataset'
           print 'python evaluate_CDL.py -p 10 -l 2 -d a '
           print '\n'
           print 'Example: For 2 Layer dense setting in citeulike-t dataset'
           print 'python evaluate_CDL.py -p 3 -l 2 -d t '
           sys.exit()
        elif opt in ("-d", "-D"):
           if arg not in ("a","t"):
             print 'not valid, allowed is only a or t. Please re-try'
             sys.exit()
           if arg == "a":
             dataset = "citeulike-a/"
           elif arg == "t":
             dataset = "citeulike-t/"
        elif opt in ("-P", "-p"):
           p = int(arg)
        elif opt in ("-L", "-l"):
           if arg not in ("2","3"):
             print 'not valid, allowed is only 2 or 3. Please re-try'
             sys.exit()
           l = arg

    #some default settings
    M_low = 50
    M_high = 300
    #vars for recall calculation
    total = 0
    correct = 0
    users = 0
    total_items_liked = 0
    #Preparing the files to be read
    data_dir = 'P%d' % p
    test_file   = 'data/'+dataset+data_dir+'/test_'+data_dir+'_1.dat'
    u_file      = 'experiments/'+dataset+'L'+l+'_'+data_dir+'/final-U.dat'
    v_file      = 'experiments/'+dataset+'L'+l+'_'+data_dir+'/final-V.dat'
    #Preparing the files to be written
    rec_file    = 'experiments/'+dataset+'L'+l+'_'+data_dir+'/rec-list.dat'
    recall_file = 'experiments/'+dataset+'L'+l+'_'+data_dir+'/recall.txt'

    #call the calculation part
    cal_rec(p,M_high,u_file,v_file,rec_file,test_file)

    R_test = read_user(test_file)
    fp = open(rec_file)
    lines = fp.readlines()

    num_users = len(range(R_test.shape[0]))

    # recall@M is calculated for M = 50 to 300
    recall_levels = M_high-M_low + 1
    recallArray = np.zeros(shape=(num_users,recall_levels))

    for user_id in range(num_users):

        s_test = set(np.where(R_test[user_id, :] > 0)[1])
        total_items_liked = len(s_test)
        l_pred = map(int, lines[user_id].strip().split(':')[1].split(' '))
        num_items_liked_in_top_M = 0
        M = 0;

        # array to store the likes at each M
        likesArray = np.zeros(recall_levels)

        for item in l_pred:
            M += 1
            total=total+1

            if item in s_test:
                correct=correct+1
                num_items_liked_in_top_M += 1


            if M >= M_low:

                #M-M_low as array indices start from 0
                likesArray[M-M_low] = num_items_liked_in_top_M

        if total_items_liked > 0:
            recallArray[user_id] = likesArray/total_items_liked
            users +=1
        else:
            recallArray[user_id] = np.nan

    fp.close()

    print " total predicted %d" % (total)
    print " correct %d" % (correct)
    print " users %d" %(users)
    print " Recall at M"
    print "recall@300 " +str(np.nanmean(recallArray,axis=0))
    np.savetxt(recall_file,np.nanmean(recallArray,axis=0))
    print "Recall values saved to " +recall_file
    plt.plot(range(M_low,M_high+1),np.nanmean(recallArray,axis=0))
    plt.ylabel("Recall")
    plt.xlabel("M")
    plt.title("CDL: Recall@M")
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
