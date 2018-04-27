import numpy as np
import sys,getopt
from data import read_user

   ##########################################
  ### Code to calculate mAP             ####
 ### Inputs: U and V dat files         ####
##########################################

def cal_precision(p,cut,u_file,v_file,rec_file,test_file):

    R_true = read_user(test_file)
    #Calculate predicted rating
    U = np.mat(np.loadtxt(u_file))
    V = np.mat(np.loadtxt(v_file))
    R = U*V.T

    num_u = R.shape[0]
    num_hit = 0
    fp = open(rec_file,'w')
    for i in range(num_u):
        if i!=0 and i%500==0:
            print 'Processed '+str(i)+' users'#+' : '+str(float(num_hit)/i/cut)
        l_score = R[i,:].A1.tolist()
        pl = sorted(enumerate(l_score),key=lambda d:d[1],reverse=True)
        l_rec = list(zip(*pl)[0])[:cut]
        s_rec = set(l_rec)
        #s_true = set(np.where(R_true[i,:]>0)[1].A1)
        s_true = set(np.where(R_true[i,:]>0)[1])
        cnt_hit = len(s_rec.intersection(s_true))
        num_hit += cnt_hit
        fp.write('%d:' % cnt_hit)
        fp.write(' '.join(map(str,l_rec)))
        fp.write('\n')
    fp.close()
    print 'Precision: %.3f' % (float(num_hit)/num_u/cut)

def main(argv):
    #Parse the args
    try:
      opts, args = getopt.getopt(argv,"hp:l:d:c:")
    except getopt.GetoptError:
      print 'python evaluate_CDL.py -p <setting_value> -l <num_layers> -d <dataset_type> -c <cut_value>'
      print '\n'
      print 'Example: For 2 Layer dense setting in citeulike-a dataset'
      print 'python evaluate_CDL.py -p 10 -l 2 -d a -c 250'
      print '\n'
      print 'Example: For 2 Layer dense setting in citeulike-t dataset'
      print 'python evaluate_CDL.py -p 3 -l 2 -d t -c 250'
      sys.exit(2)
    for opt,arg in opts:
        if opt == '-h':
           print 'python evaluate_CDL.py -p <setting_value> -l <num_layers> -d <dataset_type> -u <userid>'
           print '\n'
           print 'Example: For 2 Layer dense setting in citeulike-a dataset'
           print 'python evaluate_CDL.py -p 10 -l 2 -d a -u 8 -c 250'
           print '\n'
           print 'Example: For 2 Layer dense setting in citeulike-t dataset'
           print 'python evaluate_CDL.py -p 3 -l 2 -d t -u 8 -c 250'
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
        elif opt in ("-c", "-C"):
           cut = int(arg)
        elif opt in ("-L", "-l"):
           if arg not in ("2","3"):
             print 'not valid, allowed is only 2 or 3. Please re-try'
             sys.exit()
           l = arg

    #Preparing the files to be read
    data_dir = 'P%d' % p
    test_file   = 'data/'+dataset+data_dir+'/test_'+data_dir+'_1.dat'
    u_file      = 'experiments/'+dataset+'L'+l+'_'+data_dir+'/final-U.dat'
    v_file      = 'experiments/'+dataset+'L'+l+'_'+data_dir+'/final-V.dat'
    #Preparing the files to be written
    rec_file    = 'experiments/'+dataset+'L'+l+'_'+data_dir+'/rec-list.dat'

    cal_precision(p,cut,u_file,v_file,rec_file,test_file)

if __name__ == "__main__":
   main(sys.argv[1:])
