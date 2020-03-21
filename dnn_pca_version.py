# PCA version of dnn model:
# Y_train (sample, 4124) has its dimension deducted by PCA to (sample, n_component)
# and Y_predict is reconstruct using PCA model information to original dimension
from model_evaluate import *
from model_fit import *
from data_preprocessing import *
from simple_dnn import simplednn
from scipy.io import savemat

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
    for repeat in range(10):
        X_train_3d, X_test_3d, Y_train, Y_test = dataSplit(X, Y, ratio=0.8, seed=repeat)
        X_train = flatten_X(X_train_3d)
        X_test = flatten_X(X_test_3d)
        # perform PCA for Y_train
        n_component = 10
        Y_train_std, mean, std = standardizeYData(Y_train)
        Y_train_pca, pca_model = pcaYTrain(Y_train_std, n_component, err_ana=False)
        # model def
        hidden_layer = [100, 30]
        dnn_model = simplednn(X_train.shape[1], hidden_layer, n_component, lr=0.00003)
        # model fit
        tik = time.clock()
        dnn_model_fitted, history = modelFit(dnn_model, X_train, Y_train_pca, epoch=200, analyze=False)
        tok = time.clock()
        train_time = tok - tik
        # model prediction
        tik = time.clock()
        Y_predict,_ = modelPredict(dnn_model_fitted, X_test, pca=True, standardized=True,
                                 mean=mean, std=std, pca_model=pca_model)
        tok = time.clock()
        predict_time = tok - tik
        # model analysis
        r2_score, rmse, mae, mre = modelAnalysis(Y_test, Y_predict, verbose=False)
        savemat("result {}.mat".format(repeat), mdict={'Prediction': Y_predict, 'True': Y_test,
                                                       'Train Time': train_time,
                                                       'Prediction time': predict_time,
                                                       'rmse': rmse, 'mae': mae, 'mre': mre})
