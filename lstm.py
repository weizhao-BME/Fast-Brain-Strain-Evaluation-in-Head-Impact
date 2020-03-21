# LSTM model with PCAed Y_train
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers, initializers, regularizers
from model_fit import *
from model_evaluate import *
from data_preprocessing import *
from data_visualization import *
from keras.losses import *
import keras as K
# to eliminate random result

import os
os.environ['PYTHONHASHSEED']= str(9001)

# define a new loss function to penalize more for failure to model
# the more important PCA components rather than less important ones.
def getPcaLoss(pca_variance=None):
    def pcaLoss(y_true, y_pred, pca_variance=pca_variance):
        if pca_variance is None:
            return mean_squared_error(y_true, y_pred)
        elif len(pca_variance) == 10:
            pca_variance_tensor = K.backend.cast(K.backend.expand_dims(pca_variance, axis=0), dtype='float32')
            abs_residual_tensor = K.backend.abs(y_true - y_pred)
            return K.backend.mean(pca_variance_tensor * abs_residual_tensor, axis=-1)
        else:
            print("pca_variance error")
            print(pca_variance)
            breakpoint()
    return pcaLoss

# define a new loss function to penalize more for failure to model
# the more important PCA components rather than less important ones.
# Also, added the weight of standardization variance.
def getPcaStdLoss(pca_variance=None,std_pca=None):
    if pca_variance is not None and std_pca is not None:
        def pcaStdLoss(y_true, y_pred, pca_variance=pca_variance, std_pca=std_pca):
            weigths = (pca_variance * std_pca) / np.sum(pca_variance * std_pca)
            pca_and_pca_tensor = K.backend.cast(K.backend.expand_dims(weigths, axis=0), dtype='float32')
            abs_residual_tensor = K.backend.abs(y_true - y_pred)
            return K.backend.mean(pca_and_pca_tensor * abs_residual_tensor, axis=-1)
        return pcaStdLoss
    else:
        return mean_squared_error

# model define
# This model contains an lstm layer and a fc layer
def lstmmodel(input_nodes, lstm_node, hidden_layer, output_nodes, lr = 0.001,
              pca_variance = None, std_pca=None, dropout=True):
    # definition
    model = Sequential()
    xavier = initializers.glorot_normal(seed=9001)
    randomUniform = initializers.RandomUniform(0,1,seed=9001)
    randomNormal = initializers.random_normal(stddev=0.01, seed=9001)
    model.add(LSTM(lstm_node, input_shape=input_nodes,
                   kernel_initializer=randomNormal, recurrent_initializer=randomNormal,
                   unit_forget_bias=True,))
    model.add(Dense(hidden_layer, kernel_initializer=xavier
                    , activation='tanh'))
    if dropout:
        model.add(Dropout(0.3))
    model.add(Dense(output_nodes, kernel_initializer=xavier))
    # compile
    adam = optimizers.adam(lr=lr)
    model.compile(loss=getPcaStdLoss(pca_variance, std_pca), optimizer=adam)
    return model

if __name__ == '__main__':
    np.random.seed(9001)
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
    X_train_3d, X_test_3d, Y_train, Y_test = dataSplit(AF_X, AF_Y, ratio=0.8, seed=9001)
    # perform PCA for Y_train
    n_component = 10
    Y_train_std, mean, std = standardizeYData(Y_train)
    Y_train_pca, pca_model = pcaYTrain(Y_train_std, n_component, err_ana=False)
    # perform standardization for Y_train_pca
    Y_train_pca_std, mean_pca, std_pca = standardizeYData(Y_train_pca)
    # model def
    lstm_node = 100
    hidden_layer = 50
    output_node = 10
    lstm_model = lstmmodel(X_train_3d.shape[1:], lstm_node, hidden_layer, output_node,
                           lr=1e-3, pca_variance=pca_model.explained_variance_ratio_,
                           std_pca=std_pca)
    # model fit
    tik = time.clock()
    lstm_model_fitted, history = modelFit(lstm_model, X_train_3d, Y_train_pca_std, epoch=100, analyze=False)
    tok = time.clock()
    train_time = tok - tik
    # model predict
    tik = time.clock()
    Y_test_pca_std_pred = lstm_model_fitted.predict(X_test_3d)
    Y_test_pca_pred = Y_test_pca_std_pred * std_pca + mean_pca
    Y_test_std_pred = pca_model.inverse_transform(Y_test_pca_pred)
    Y_test_pred = Y_test_std_pred * std + mean
    tok = time.clock()
    predict_time = tok - tik
    # model analysis
    # functions in model_evaluation.py can be added here.
