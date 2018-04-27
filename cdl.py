# pylint: skip-file
import mxnet as mx
import numpy as np
import logging
import data
from math import sqrt
from autoencoder import AutoEncoderModel
import os,sys,getopt

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
             vocab_size = 8000
             nusers = 5551
             nitems = 16980
           elif arg == "t":
             dataset = "citeulike-t/"
             vocab_size = 20000
             nusers = 7947
             nitems = 25975
        elif opt in ("-P", "-p"):
           p = int(arg)
        elif opt in ("-L", "-l"):
           if arg not in ("2","3"):
             print 'not valid, allowed is only 2 or 3. Please re-try'
             sys.exit()
           l = int(arg)

    data_dir = 'P%d' % p
    #Prepare the data files
    dir_save    = 'experiments/'+dataset+'L'+str(l)+'_'+data_dir
    u_file      = dir_save+'/final-U.dat'
    v_file      = dir_save+'/final-V.dat'
    theta_file  = dir_save+'/final-theta.dat'
    log_file    = dir_save+'/cdl.log'
    mult_file   = 'data/'+dataset+'/mult.dat'
    train_file  = 'data/'+dataset+data_dir+'/train_'+data_dir+'_1.dat'
    #Call to main cdl function
    cdl(p,l,nusers,nitems,vocab_size,train_file,mult_file,u_file,v_file,theta_file,log_file,dir_save)

def cdl(p,l,nusers,nitems,vocab_size,train_file,mult_file,u_file,v_file,theta_file,log_file,dir_save):

    #default params
    lambda_u = .1    # lambda_u in CDL
    lambda_v = 10    # lambda_v in CDL
    K = 50           # no of latent vectors in the compact representation
    K1 = 100         # extra layer for L=3
    is_dummy = False # whether to use dummy data
    num_iter = 100
    batch_size = 256
    np.random.seed(1234) # set seed
    lv = 1e-2        # lambda_v/lambda_n in CDL

    if not os.path.isdir(dir_save):
        os.system('mkdir -p %s' % dir_save)
    fp = open(log_file,'w')
    print ('p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d' % (p,lambda_v,lambda_u,lv,K))
    fp.write('p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d\n' % \
            (p,lambda_v,lambda_u,lv,K))
    fp.close()
    if is_dummy:
        X = data.get_dummy_mult()
        R = data.read_dummy_user()
    else:
        X = data.get_mult(mult_file,vocab_size,nitems)
        R = data.read_user(train_file,nusers,nitems)
    # set to INFO to see less information during training
    logging.basicConfig(level=logging.DEBUG)
    #Commenting this because of CUDA9.1 and mxnet python2.7 mismatches. Had to live with CPU
    #for now
    #ae_model = AutoEncoderModel(mx.gpu(0), [784,500,500,2000,10], pt_dropout=0.2,
    #    internal_act='relu', output_act='relu')

    #mx.cpu() no param needed for cpu.
    #Pick layers based on input argument l
    if l == 3 :
        ae_model = AutoEncoderModel(mx.cpu(), [X.shape[1],200,K1,K],
                   pt_dropout=0.2, internal_act='relu', output_act='relu')
    elif l == 2:
        ae_model = AutoEncoderModel(mx.cpu(), [X.shape[1],200,K],
                   pt_dropout=0.2, internal_act='relu', output_act='relu')

    train_X = X

    ae_model.layerwise_pretrain(train_X, 256, 50000, 'sgd', l_rate=0.1, decay=0.0,
                             lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    #V = np.zeros((train_X.shape[0],10))
    V = np.random.rand(train_X.shape[0],K)/10
    lambda_v_rt = np.ones((train_X.shape[0],K))*sqrt(lv)
    U, V, theta, BCD_loss = ae_model.finetune(train_X, R, V, lambda_v_rt, lambda_u,
            lambda_v, dir_save, batch_size,
            num_iter, 'sgd', l_rate=0.1, decay=0.0,
            lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    #ae_model.save('cdl_pt.arg')
    #Save U,V,theta for future calculations
    np.savetxt(u_file,U,fmt='%.5f',comments='')
    np.savetxt(v_file,V,fmt='%.5f',comments='')
    np.savetxt(theta_file,theta,fmt='%.5f',comments='')

    #ae_model.load('cdl_pt.arg')
    Recon_loss = lambda_v/lv*ae_model.eval(train_X,V,lambda_v_rt)
    print ("Training error: %.3f" % (BCD_loss+Recon_loss))
    fp = open(log_file,'a')
    fp.write("Training error: %.3f\n" % (BCD_loss+Recon_loss))
    fp.close()
    #print "Validation error:", ae_model.eval(val_X)


if __name__ == '__main__':
    main(sys.argv[1:])

