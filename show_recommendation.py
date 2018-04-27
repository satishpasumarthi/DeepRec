import csv
from data import read_user
import numpy as np
import sys,getopt

  ##########################################
 ### Code to show recomm. for a user   ####
##########################################

def main(argv):
    # read predicted results
    try:
      opts, args = getopt.getopt(argv,"hp:l:d:u:")
    except getopt.GetoptError:
      print 'python show_recommendation.py -p <setting_value> -l <num_layers> -d <dataset_type> -uid <userid>'
      print 'Example: For 2 Layer dense setting in citeulike-a dataset'
      print 'python show_recommendation.py -p 10 -l 2 -d a -uid 8'
      print 'Example: For 2 Layer dense setting in citeulike-t dataset'
      print 'python show_recommendation.py -p 3 -l 2 -d t -uid 8'
      sys.exit(2)
    for opt,arg in opts:
        if opt == '-h':
           print 'python show_recommendation.py -p <setting_value> -l <num_layers> -d <dataset_type> -u <userid>'
           print 'Example: For 2 Layer dense setting in citeulike-a dataset'
           print 'python show_recommendation.py -p 10 -l 2 -d a -u 8'
           print 'Example: For 2 Layer dense setting in citeulike-t dataset'
           print 'python show_recommendation.py -p 3 -l 2 -d t -u 8'
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
        elif opt in ("-U", "-u"):
           uid = int(arg)

    #Preparing the files to be read
    data_dir = 'P%d' % p
    dir_save = 'P%d' % p
    csv_file   = 'data/'+dataset+'raw_inputs/raw-data.csv'
    test_file  = 'data/'+dataset+data_dir+'/test_'+data_dir+'_1.dat'
    train_file = 'data/'+dataset+data_dir+'/train_'+data_dir+'_1.dat'
    rec_file   = 'experiments/'+dataset+'L'+l+'_'+data_dir+'/rec-list.dat'
    #Call the recommendations function
    show_recommendation(uid,csv_file,train_file,test_file,rec_file)

def show_recommendation(user_id,csv_file,train_file,test_file,rec_file):
    
    csvReader = csv.reader(open(csv_file,'rb'))
    d_id_title = dict()
    for i,row in enumerate(csvReader):
        if i==0:
            continue
        d_id_title[i-1] = row[3]

    #Read the files
    R_test = read_user(test_file)
    R_train = read_user(train_file)
    fp = open(rec_file)
    lines = fp.readlines()
    
    s_test = set(np.ravel(np.where(R_test[user_id, :] > 0)[1]))
    l_train = np.ravel(np.where(R_train[user_id, :] > 0)[1]).tolist()
    
    l_pred = map(int,lines[user_id].strip().split(':')[1].split(' '))
    print '#####  Articles in the Training Sets  #####'
    for i in l_train:
        print d_id_title[i]
    print '\n#####  Articles Recommended (Correct Ones Marked by Stars)  #####'
    for i in l_pred:
        if i in s_test:
            print '* '+d_id_title[i]
        elif i not in l_train:
            print d_id_title[i]
    fp.close()

if __name__ == "__main__":
   main(sys.argv[1:])
