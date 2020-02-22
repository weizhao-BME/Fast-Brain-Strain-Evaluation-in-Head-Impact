# Fully connected neuron networks to predict 4124 elements of brain impact
# using standardized football data.
# loading CNN packages
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, initializers
from model_fit import modelFit
# util functions
from data_preprocessing import *
from model_evaluate import *
from model_storage import *
# to eliminate random result
import tensorflow as tf
tf.random.set_seed(9001)
import os
os.environ['PYTHONHASHSEED']= str(9001)


# Model declaration iteratively
# hidden_layers is a vector of nodes in each hidden layer. e.g. [300,120,48]
def simplednn(input_nodes, hidden_layers, output_nodes,lr, hid_activation='relu', out_activation='relu'):
    model = Sequential()
    xavier=initializers.glorot_normal(seed=9001)
    # first layer
    model.add(Dense(hidden_layers[0],
                    input_dim=input_nodes,
                    kernel_initializer=xavier,
                    activation=hid_activation))
    # rest hidden layers
    for layer_node in hidden_layers[1:]:
        model.add(Dense(layer_node, kernel_initializer=xavier, activation=hid_activation))
    # output layer
    model.add(Dense(output_nodes,kernel_initializer=xavier,activation=out_activation))
    adam = optimizers.adam(lr=lr)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=adam)
    return model

if __name__ == '__main__':
    # Load data
    AF_X, AF_Y = loadOriginalData()
    X_train_3d, X_test_3d, Y_train, Y_test = dataSplit(AF_X, AF_Y, ratio=0.8)
    X_train = flatten_X(X_train_3d)
    X_test = flatten_X(X_test_3d)
    # standardize Y_train
    Y_std,mean,std = standardizeYData(Y_train)
    # model def
    hidden_layer = [200, 50]
    dnn_model = simplednn(X_train.shape[1], hidden_layer, 4124, lr=0.0001)
    # model fit
    dnn_model_fitted = modelFit(dnn_model, X_train, Y_train, epoch=100)
    # model prediction
    Y_predict = modelPredict(dnn_model_fitted, X_test,standardized=True,mean=mean,std=std)
    # model analysis
    # test set
    print("Test error: ")
    modelAnalysis(Y_test, Y_predict)
    # train set
    print("Train error: ")
    Y_train_predict = modelPredict(dnn_model_fitted, X_train)
    modelAnalysis(Y_train, Y_train_predict)
    # graphs for 5 samples
    # plotting residual histogram with clipped Y_predict (negative values are clipped to 0)
    plot(Y_test, Y_predict, p=True)
