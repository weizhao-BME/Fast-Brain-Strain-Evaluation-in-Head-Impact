# metrics
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error
# plotting
import matplotlib.pyplot as plt
from data_preprocessing import *
import seaborn as sns

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
        Y_reconstruct = reconstructY(Y_predicted, mean, std, pca_model)
    elif(pca):
        Y_reconstruct = reconstructYFromPCA(Y_predicted, pca_model)
    elif(standardized):
        Y_reconstruct = reconstructYFromStd(Y_predicted, mean, std)
    else:
        Y_reconstruct = Y_predicted
    return Y_reconstruct

def clippingYPredict(Y_predict):
    Y_clipped = Y_predict
    Y_clipped[Y_clipped<0]=0
    return Y_clipped


# Analyze the model's accuracy using Y_true and Y_predict, using ???
# Input: Y_true, Y_predict, verbose=True
# Output: error
def modelAnalysis(Y_true, Y_predict, verbose=True):
    r2 = round(r2_score(Y_true, Y_predict),4)
    explained_var = round(explained_variance_score(Y_true, Y_predict),4)
    rmse = np.sqrt(mean_squared_error(Y_true, Y_predict))
    mae = mean_absolute_error(Y_true, Y_predict)
    mre = np.mean(np.abs(Y_true - Y_predict)/np.max(Y_true,axis=1,keepdims=True))
    if verbose:
        print("r2_score is ", r2)
        print("explained variance score is ", explained_var)
        print("root mean squared error is ", rmse)
        print("mean absolute error is ",mae)
        print("mean relative error is ",mre)
    return {"r2":r2,
            "explained_var":explained_var,
            "rmse":rmse,
            "mae":mae,
            "mre":mre}


# some graphs
def testPredictGraph(Y_true, Y_predict, title = 'y, y_hat plot',sampleIndex=0):
    plt.scatter(Y_true[sampleIndex,:], Y_predict[sampleIndex,:], s=0.1)
    max_range = np.max([np.max(Y_true[sampleIndex,:]), np.max(Y_predict[sampleIndex,:])])
    x_line = np.linspace(0,max_range,100)
    plt.plot(x_line, x_line, color='r')
    plt.title(title + ' of sample ' + str(sampleIndex))
    plt.ylabel('Predicted MPS')
    plt.xlabel('KTH MPS')
    plt.show()

def residualGraph(Y_true, Y_predict, title = 'Residual plot', sampleIndex=0):
    plt.scatter(np.arange(4124),Y_predict[sampleIndex,:]-Y_true[sampleIndex,:], s=0.1)
    plt.title(title+' of sample '+str(sampleIndex))
    plt.ylabel('Residual')
    plt.xlabel('index')
    plt.show()

def residualHist(Y_true, Y_predict, title = "Residual histogram", sampleIndex=0, abs=True):
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

def relativeResidualGraph(Y_true, Y_predict, title = 'Relative Residual plot', sampleIndex=0):
    plt.scatter(np.arange(4124),(Y_predict[sampleIndex,:]-Y_true[sampleIndex,:])/np.max(Y_true[sampleIndex,:]), s=0.1)
    plt.title(title+' of sample '+str(sampleIndex))
    plt.ylabel('Residual')
    plt.xlabel('index')
    plt.show()


def plot(Y_test, Y_predict, p=True):
    if p:
        residualHist(Y_test, Y_predict)
        for sample in np.arange(0, 70, 14):
            testPredictGraph(Y_test, Y_predict, sampleIndex=sample)
            residualGraph(Y_test, Y_predict, sampleIndex=sample)
            relativeResidualGraph(Y_test, Y_predict, sampleIndex=sample)
