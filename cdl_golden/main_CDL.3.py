import numpy as np
import time
import CDL


# All parameters
doc_vocab_file_path = 'mult.dat'
user_item_train_file_path = 'cf-train-1-users.dat'
user_item_test_file_path = 'cf-test-1-users.dat'

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
#from pandas import read_csv                              
                                                          
def read_rating(file_path, has_header=False):             
    rating_mat = list()                                   
    with open(file_path) as fp:                           
        if has_header is True:                            
            fp.readline()                                 
        for line in fp:                                   
            line = line.split(',')                        
            user, item, rating = line[0], line[1], line[2]
            rating_mat.append( [user, item, rating] )     
    return np.array(rating_mat).astype('float32')         
                                                          

R_test = read_rating('cf-test-1-users.dat')               
R = read_rating('cf-train-1-users.dat')               
                                                          
from keras.layers import Input, Dense
from keras.models import Model

input_size = 8000
hidden_size = 200
code_size = 50


input_img = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size, activation='relu')(input_img)
code = Dense(code_size, activation='relu')(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(code)
output_img = Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = Model(input_img, output_img)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)







model = CDL.CollaborativeDeepLearning(X, [input_size,hidden_size,code_size])


model.pretrain(lamda_w=0.001, encoder_noise=0.3, epochs=20)


model_history = model.fineture(R, R_test, lamda_u=0.01, lamda_v=0.1, lamda_n=0.1, lr=0.01, epochs=30)
testing_rmse = model.getRMSE(R_test)
print('Testing RMSE = {}'.format(testing_rmse))

