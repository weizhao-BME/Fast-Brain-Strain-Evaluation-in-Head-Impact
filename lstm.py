# LSTM model with PCAed Y_train
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers
from model_fit import *
from model_evaluate import *
from data_preprocessing import *


def lstmmodel(input_nodes, lstm_node, hidden_layer, output_nodes):
    model = Sequential()
    model.add(LSTM(lstm_node, input_shape=input_nodes))
    model.add(Dense(hidden_layer, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_nodes, kernel_initializer='normal', activation='tanh'))
    #compile
    adam = optimizers.adam(lr=0.005,decay=1e-6)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

if __name__ == '__main__':
    np.random.seed(9001)
    # load data
    AF_X, AF_X = loadOriginalData()
    # Load data
    AF_X, AF_Y = loadOriginalData()
    X_train_3d, X_test_3d, Y_train, Y_test = dataSplit(AF_X, AF_Y, ratio=0.8)
    # perform PCA for Y_train
    n_component = 10
    Y_train_std, mean, std = standardizeYData(Y_train)
    Y_train_pca, pca_model = pcaYTrain(Y_train_std, n_component)
    # model def
    lstm_node = 100
    hidden_layer = 50
    output_node = 10
    lstm_model = lstmmodel(X_train_3d.shape[1:], lstm_node, hidden_layer, output_node)
    # model fit
    dnn_model_fitted = modelFit(lstm_model, X_train_3d, Y_train_pca)
    # model prediction
    Y_predict = modelPredict(dnn_model_fitted, X_test_3d, pca=True, standardized=True,
                             mean=mean, std=std, pca_model=pca_model)
    # model analysis
    # test error
    print("Test evaluation: ")
    modelAnalysis(Y_test, Y_predict)
    # train error
    print("Train evaluation: ")
    Y_predict_train = modelPredict(dnn_model_fitted, X_train_3d, pca=True, standardized=True,
                                   mean=mean, std=std, pca_model=pca_model)
    modelAnalysis(Y_train, Y_predict_train)
    # graphs
    plot(Y_test, Y_predict, p=True)
