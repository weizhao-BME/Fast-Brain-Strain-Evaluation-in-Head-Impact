# PCA version of dnn model:
# Y_train (sample, 4124) has its dimension deducted by PCA to (sample, n_component)
# and Y_predict is reconstruct using PCA model information to original dimension
from model_evaluate import *
from model_fit import *
from data_preprocessing import *
from simple_dnn import simplednn

if __name__ == '__main__':
    np.random.seed(9001)
    # Load data
    AF_X, AF_Y = loadOriginalData()
    X_train_3d, X_test_3d, Y_train, Y_test = dataSplit(AF_X, AF_Y, ratio=0.8)
    X_train = flatten_X(X_train_3d)
    X_test = flatten_X(X_test_3d)
    # perform PCA for Y_train
    n_component = 10
    Y_train_std, mean, std = standardizeYData(Y_train)
    Y_train_pca, pca_model = pcaYTrain(Y_train_std, n_component)
    # model def
    hidden_layer = [300, 100]
    dnn_model = simplednn(X_train.shape[1], hidden_layer, n_component, lr=0.00003)
    # model fit
    dnn_model_fitted = modelFit(dnn_model, X_train, Y_train_pca, epoch=100)
    # model prediction
    Y_predict = modelPredict(dnn_model_fitted, X_test, pca=True, standardized=True,
                             mean=mean, std=std, pca_model=pca_model)
    # model analysis
    # test error
    print("Test evaluation: ")
    modelAnalysis(Y_test, Y_predict)
    # train error
    print("Train evaluation: ")
    Y_predict_train = modelPredict(dnn_model_fitted, X_train, pca=True, standardized=True,
                             mean=mean, std=std, pca_model=pca_model)
    modelAnalysis(Y_train, Y_predict_train)
    # graphs
    plot(Y_test, Y_predict,p=False)
