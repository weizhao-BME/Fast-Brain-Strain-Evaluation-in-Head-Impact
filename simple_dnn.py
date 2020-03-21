# Fully connected neuron networks to predict 4124 elements of brain impact
# using standardized football data.
# loading CNN packages
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, initializers
from model_fit import modelFit
import time
# util functions
from data_preprocessing import *
from model_evaluate import *
from model_storage import *
from scipy.io import savemat
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
    AF_X, AF_Y = loadAFData()
    # The first 3 channels have little impact to the result, thus delete
    AF_X = np.transpose(AF_X, [0, 2, 1])
    # get rid of bad data points
    AF_X_reasonable = np.delete(AF_X, id_delete, axis=0)
    AF_Y_reasonable = np.delete(AF_Y, id_delete, axis=0)
    MMA_X, MMA_Y = loadMMAData()
    MMA_X = np.transpose(MMA_X, [2, 1, 0])
    X = np.row_stack((MMA_X, AF_X_reasonable))
    Y = np.row_stack((MMA_Y, AF_Y_reasonable))
    X_train_3d, X_test_3d, Y_train, Y_test = dataSplit(X, Y, ratio=0.8, seed=9001)
    X_train = flatten_X(X_train_3d)
    X_test = flatten_X(X_test_3d)
    # standardize Y_train
    Y_std, mean, std = standardizeYData(Y_train)
    # model def
    hidden_layer = [200, 50]
    dnn_model = simplednn(X_train.shape[1], hidden_layer, 4124, lr=0.0001)
    # model fit
    tik = time.clock()
    dnn_model_fitted, _ = modelFit(dnn_model, X_train, Y_train, epoch=100, analyze=False)
    tok = time.clock()
    train_time = tok - tik
    # model prediction
    tik = time.clock()
    Y_predict, _ = modelPredict(dnn_model_fitted, X_test, standardized=False, mean=mean, std=std)
    tok = time.clock()
    predict_time = tok - tik
    # model analysis
    # test set
    r2_score, rmse, mae, mre = modelAnalysis(Y_test, Y_predict, verbose=False)
