
import numpy as np
import time
import CDL


# All parameters
doc_vocab_file_path = 'mult.dat'
user_item_train_file_path = 'cf-train-1-users.dat'
user_item_test_file_path = 'cf-test-1-users.dat'
users_to_test = [10, 20, 30]
m_value = 10
# processing doc vocab file and creating X matrix


def process_mult(filename):
    f_mult = open(filename, 'r')
    lines = f_mult.readlines()

    num_docs = len(lines)
    vocab_size = 8000

    X = np.zeros((num_docs, vocab_size))

    f_mult = open(filename, 'r')

    docid = 0

    for line in f_mult:
        x = str(line).split(' ')
        for i in range(1,int(x[0])):
            s = x[i].split(':')
            X[docid][int(s[0])] = int(s[1])
        docid = docid + 1
    return X, num_docs



start_time = time.time()
X, num_items = process_mult(doc_vocab_file_path)
end_time = time.time()


# processing user item file and creating R matrix

import numpy as np
import csv                                        
#from pandas import read_csv                              
                                                          
def read_rating(file_path, has_header=False, test_data=False, user_to_load=None):             
    rating_mat = list()                                   
    with open(file_path) as fp:                           
        if has_header is True:                            
            fp.readline()                                 
        for line in fp:                                   
            line = line.split(',')                        
            user, item, rating = line[0], line[1], line[2]
            if not test_data:
                rating_mat.append( [user, item, rating] )
            elif test_data and str(user_to_load) == user:
                rating_mat.append( [user, item, rating] )
    return np.array(rating_mat).astype('float32')

def read_artical_titles(file_path):
    article_titles = []
    with open(file_path, 'r', encoding='cp850') as file:
       reader = csv.reader(file, delimiter=',')
       for row in reader:
           article_titles.append(row[3].strip())
    return article_titles
                      
def make_test_mat_for_rec(R, R_test, article_titles):
    user_read = {}
    user_train_read = {}
    rating_mat = list()
    for data in R_test.astype(int):
       user_id = data[0]
       item_id = data[1]
       rating = data[2]
       if user_id not in user_read:
          user_read[user_id] = []
       user_read[user_id].append(item_id)
    print('got here 1')  
    #for data in R.astype(int):
    #   user_id = data[0]
    #   item_id = data[1]
    #   rating = data[2]
    #   if user_id not in user_train_read:
    #      user_train_read[user_id] = []
    #   user_train_read[user_id].append(item_id)
    #print('got here 2')
    already_done = set()
    iterations = 0
    with open('cf--new-test-1-users.dat', 'w') as file: 
       for user_id, read_data in user_read.items():
          for article_id in range(16980):
             if article_id in read_data:
                output = str(user_id) + ',' + str(article_id) + ',' + '1' + '\n'
                file.write(output)
             else:
                output = str(user_id) + ',' + str(article_id) + ',' + '0' + '\n'
                file.write(output)
          iterations += 1
          if iterations % 250 == 0:
             print('Number of users processed: ' + str(iterations))
             
def get_user_rec_pairs(R, R_test, rec_user_ids):
    user_read = {}
    user_train_read = {}
    rating_mat = list()
    for data in R_test.astype(int):
       user_id = data[0]
       item_id = data[1]
       rating = data[2]
       if user_id not in user_read:
          user_read[user_id] = []
       user_read[user_id].append(item_id)
    print('got here 1')
    for data in R.astype(int):
       user_id = data[0]
       item_id = data[1]
       rating = data[2]
       if user_id not in user_train_read:
          user_train_read[user_id] = []
       user_train_read[user_id].append(item_id)
    print('got here 2')
    for rec_user_id in rec_user_ids:
        read_data = user_read[rec_user_id]
        read_train_data = user_train_read[rec_user_id]
        iterations = 0
        for article_id in range(16980):
           if article_id in read_data:
              rating_mat.append([rec_user_id, article_id, 1])
           elif article_id not in read_train_data:
              rating_mat.append([rec_user_id, article_id, 0])
        iterations += 1
        if iterations % 250 == 0:
           print('Number of users processed: ' + str(iterations))
    
    return np.array(rating_mat).astype('float32'), user_read 

#R_test_rec = read_rating('cf--new-test-1-users.dat', test_data=True, user_to_load=3)
R_test = read_rating('cf-test-1-users.dat')               
R = read_rating('cf-train-1-users.dat')
R_test_rec, users_liked = get_user_rec_pairs(R, R_test, users_to_test)           
print('read in data')                                                      
articles_titles = read_artical_titles('raw-data.csv')
#print('read in articles')  
#make_test_mat_for_rec(R_test,articles_titles)    
#print('read in rec test mat.')  

if True:

   from keras.layers import Input, Dense
   from keras.models import Model

   input_size = 8000
   hidden_size = 200
   code_size = 50


   model = CDL.CollaborativeDeepLearning(X, [input_size,hidden_size,code_size])


   model.pretrain(lamda_w=0.001, encoder_noise=0.3, epochs=1)


   model_history = model.fineture(R[:44408], R[44408:55510], lamda_u=0.01, lamda_v=0.1, lamda_n=0.1, lr=0.01, epochs=1)
   #testing_rmse = model.getRMSE(R_test)

   for user_to_test in users_to_test:
      reccommendations = model.get_reccommendations(R_test_rec, user_to_test, m_value)
      model.print_recall(reccommendations, users_liked[user_to_test], user_to_test)

