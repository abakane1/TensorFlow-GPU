# CNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# RNN
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

import tflearn

# data loading and basic transformations
import tflearn.datasets.mnist as mnist
def MnistCNN():
    X, Y, X_test, Y_test = mnist.load_data(one_hot=True)
    # print (X)
    X = X.reshape([-1, 28, 28, 1])
    # print (X)
    X_test = X_test.reshape([-1, 28, 28, 1])

    # Building the Network
    CNN = input_data(shape=[None, 28, 28, 1], name='input')
    CNN = conv_2d(CNN, 32, 5, activation='relu', regularizer="L2")
    CNN = max_pool_2d(CNN, 2)
    CNN = local_response_normalization(CNN)
    CNN = conv_2d(CNN, 64, 5, activation='relu', regularizer="L2")
    CNN = max_pool_2d(CNN, 2)
    CNN = local_response_normalization(CNN)
    CNN = fully_connected(CNN, 1024, activation=None)
    CNN = dropout(CNN, 0.5)
    CNN = fully_connected(CNN, 10, activation='softmax')
    CNN = regression(CNN, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='target')

    # train
    model = tflearn.DNN(CNN,
                        tensorboard_verbose=0,
                        tensorboard_dir='MINST_tflearn_board/',
                        checkpoint_path='MINST_tflearn_checkpoints/checkpoints')
    model.fit({'input': X}, {'target': Y},
        n_epoch=3,
        validation_set=({'input': X_test},{'target':Y_test}),
        show_metric=True,
        snapshot_step=1000,
        run_id='convnet_mnist')

def MNISTRNN():
    train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,valid_portion=0.1)
    X_train, Y_train = train
    X_test, Y_test = test

    X_train = pad_sequences(X_train, maxlen=100,value=0.)
    X_test = pad_sequences(X_test, maxlen=100,value=0.)
    Y_train = to_categorical(Y_train, nb_classes=2)
    Y_test = to_categorical(Y_test, nb_classes=2)

    # LSTM
    RNN = tflearn.input_data([None, 100])
    RNN = tflearn.embedding(RNN, input_dim=10000, output_dim=128)
    RNN = tflearn.lstm(RNN, 128, dropout=0.8)
    RNN = tflearn.fully_connected(RNN, 2, activation='softmax')
    RNN = tflearn.regression(RNN, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    # train
    model = tflearn.DNN(RNN, tensorboard_verbose=0, tensorboard_dir='MINST_tflearn_board_RNN/')
    model.fit(X_train, Y_train, validation_set=(X_test,Y_test),show_metric=True,batch_size=32)

MNISTRNN()
