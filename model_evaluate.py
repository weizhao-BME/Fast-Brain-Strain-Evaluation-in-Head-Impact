# metrics
# from sklearn.metrics import r2_score, explained_variance_score
# plotting
import matplotlib.pyplot as plt
from data_preprocessing import *
import seaborn as sns
from keras import layers, backend

# mean squared error
def mse(true, predict):
    return np.mean((true - predict)**2)

# mean absolute error
def mae(true, predict):
    return np.mean(np.abs(true - predict))

# r_2 score
def r2score(true, predict):
    return 1 - np.sum((true - predict)**2)/np.sum((true - np.mean(true))**2)


# get the predicted value of Y using fitted model
# Input: fitted model, X_test, Y_test, PCA(should the output be reconstructed from pca), std(similar to pca)
# Output: Y_predicted
def modelPredict(model, X_test, pca=False, standardized=False,
                 mean=None, std=None, pca_model=None):
    Y_predicted = model.predict(X_test)
    if pca:
        assert type(pca_model)!=type(None), "You should pass pca_model to reconstruct Y"
    elif standardized:
        assert  type(mean)!=type(None) and type(std)!=type(None), "You should pass mean, std to reconstruct Y"
    if pca and standardized :
        Y_reconstruct, Y_predict_std = reconstructY(Y_predicted, mean, std, pca_model)
        cache = {"Y_predicted_pca": Y_predicted,
                 "Y_predicted_std": Y_predict_std}
    elif(pca):
        Y_reconstruct = reconstructYFromPCA(Y_predicted, pca_model)
    elif(standardized):
        Y_reconstruct = reconstructYFromStd(Y_predicted, mean, std)
    else:
        Y_reconstruct = Y_predicted
    cache = {"Y_predicted_pca":Y_predicted}
    return Y_reconstruct, cache

# Analyze the model's accuracy using Y_true and Y_predict,
# when they are of the shape (N, 4124).
# Input: Y_true, Y_predict, verbose=True
# Output: error
def modelAnalysis(Y_true, Y_predict, verbose=True):
    # Assume Y_true.shape = (sample, 4124)
    if Y_true.ndim == 2:
        assert Y_true.shape[1] == 4124, "Y_true dimension error."
        r2_record = np.zeros((Y_true.shape[0]))
        for i in range(Y_true.shape[0]):
            r2_record[i] = r2_score(Y_true[i, :], Y_predict[i, :])
        r2_avg = round(np.mean(r2_record), 4)
    rmse = np.sqrt(mse(Y_true, Y_predict))
    mae1 = mae(Y_true, Y_predict)
    if Y_true.ndim == 2:
        mre = np.mean(np.abs(Y_true - Y_predict) / np.max(Y_true, axis=1, keepdims=True))
    else:
        mre = np.mean(np.abs(Y_true - Y_predict)/np.max(Y_true))
    if verbose:
        if Y_true.ndim == 2:
            print("r2_score_avg is ", r2_avg)
        print("root mean squared error is ", rmse)
        print("mean absolute error is ",mae1)
        print("mean relative error is ",mre)
    if Y_true.ndim == 2:
        return r2_avg,rmse,mae1,mre
    return rmse,mae,mre

# calculate loss manually
def getLossManual(loss):
    y_true = layers.Input(shape=(None,))
    y_pred = layers.Input(shape=(None,))
    loss_func = backend.function([y_true, y_pred], [loss(y_true, y_pred)])
    return loss_func

# analyze the first component
def analyzeFirstComponent(Y_first, Y_first_pred):
    mae = np.mean(mae(Y_first, Y_first_pred))
    print("mae is: ", mae)
    plt.scatter(np.arange(len(Y_first)), Y_first, c='blue', s=0.5)
    plt.scatter(np.arange(len(Y_first_pred)), Y_first_pred, c='red', s=0.5)
    plt.show()
    return mae

# some graphs
# Plot y_predict against y_true (4124 element from 1 sample)
def yHatYPlot(Y_true, Y_predict, title = 'y, y_hat plot',sampleIndex=0):
    plt.scatter(Y_true[sampleIndex,:], Y_predict[sampleIndex,:], s=0.1)
    max_range = np.max([np.max(Y_true[sampleIndex,:]), np.max(Y_predict[sampleIndex,:])])
    x_line = np.linspace(0,max_range,100)
    plt.plot(x_line, x_line, color='r')
    plt.title(title + ' of sample ' + str(sampleIndex))
    plt.ylabel('Predicted MPS')
    plt.xlabel('KTH MPS')
    plt.show()

# plot all y_predict against y_true (for one point predictions)
def wholeYHatY(Y_true, Y_predict, title = 'Y, Y_hat plot'):
    plt.scatter(Y_true, Y_predict, s=0.5)
    max_range = np.max([np.max(Y_true), np.max(Y_predict)])
    x_line = np.linspace(0, max_range, 100)
    plt.plot(x_line, x_line, color='r')
    plt.title(title)
    plt.ylabel('Predicted MPS')
    plt.xlabel('KTH MPS')
    plt.show()

# plot the y values against index
def testPredictPlot(Y_true, Y_predict, title = 'y, y_pred plot',sampleIndex=0):
    plt.scatter(np.arange(Y_true.shape[1]), Y_true[sampleIndex,:], s=0.1, c='blue', label="KTH MPS")
    plt.scatter(np.arange(Y_predict.shape[1]), Y_predict[sampleIndex,:], s=0.1, c='red', label="Predicted MPS")
    plt.title(title + ' of sample ' + str(sampleIndex))
    plt.legend()
    plt.ylabel('MPS')
    plt.xlabel('Index')
    plt.show()

# residual plot
def residualGraph(Y_true, Y_predict, title = 'Residual plot', sampleIndex=0):
    plt.scatter(np.arange(4124),Y_predict[sampleIndex,:]-Y_true[sampleIndex,:], s=0.1)
    plt.title(title+' of sample '+str(sampleIndex))
    plt.ylabel('Residual')
    plt.xlabel('index')
    plt.show()

# residual histogram
def residualHist(Y_true, Y_predict, title = "Residual histogram", abs=True):
    Y_predict_flatten = Y_predict.reshape(Y_predict.shape[0]*Y_predict.shape[1])
    Y_true_flatten = Y_true.reshape(Y_true.shape[0]*Y_true.shape[1])
    if abs:
        x = np.abs(Y_predict_flatten-Y_true_flatten)
    else:
        x = Y_predict_flatten-Y_true_flatten
    hist = sns.distplot(x)
    hist.set_title(title)
    plt.xlabel("absolute error of single element MPS")
    plt.ylabel("fraction of all brain element")
    plt.show()

# relative residual against index
def relativeResidualGraph(Y_true, Y_predict, title = 'Relative Residual plot', sampleIndex=0):
    plt.scatter(np.arange(4124),(Y_predict[sampleIndex,:]-Y_true[sampleIndex,:])/np.max(Y_true[sampleIndex,:]), s=0.1)
    plt.title(title+' of sample '+str(sampleIndex))
    plt.ylabel('Residual')
    plt.xlabel('index')
    plt.show()

# if p=True, then this function would plot 5 sample of residual analysis plots
# for model evaluation
def plot(Y_test, Y_predict, p=True):
    if p:
        residualHist(Y_test, Y_predict)
        for sample in np.arange(0, 70, 14):
            yHatYPlot(Y_test, Y_predict, sampleIndex=sample)
            residualGraph(Y_test, Y_predict, sampleIndex=sample)
            relativeResidualGraph(Y_test, Y_predict, sampleIndex=sample)
