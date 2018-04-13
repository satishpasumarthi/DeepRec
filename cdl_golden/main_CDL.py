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
print (end_time - start_time)


# processing user item file and creating R matrix

def process_rating_data(filename):
    f_train = open(filename, 'r')
    lines = f_train.readlines()
    num_users = len(lines)
    R = np.zeros((num_users, num_items))

    f_train = open(filename, 'r')

    userid = 0

    for line in f_train:
        x = str(line).split(' ')
        for i in range(1,int(x[0])):
            R[userid][int(x[i])] = 1
        userid = userid + 1
    return R



R = process_rating_data(user_item_train_file_path)
R_test = process_rating_data(user_item_test_file_path)



from keras.layers import Input, Dense
from keras.models import Model

input_size = 8000
hidden_size = 100
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



model = CDL.CollaborativeDeepLearning(X, [8000, 200, 50])


model.pretrain(lamda_w=0.001, encoder_noise=0.3, epochs=10)


model_history = model.fineture(R, R_test, lamda_u=0.01, lamda_v=0.1, lamda_n=0.1, lr=0.01, epochs=3)
testing_rmse = model.getRMSE(R_test)
print('Testing RMSE = {}'.format(testing_rmse))

