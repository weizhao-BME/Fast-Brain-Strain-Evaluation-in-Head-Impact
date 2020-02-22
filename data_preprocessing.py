import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


# load original data (not PCAed) of football dataset
def loadOriginalData():
    AF_X_Dict = loadmat('AF_X.mat')
    AF_X = AF_X_Dict["AF_X"]
    AF_Y_Dict = loadmat('football_Y.mat')
    AF_Y = AF_Y_Dict["label_data"]
    return AF_X, AF_Y

# Data index: create split index (random function), get num of samples and trainSet ratio
# returns a partition index
def getIndex(num, ratio):
    np.random.seed(9001)
    index = np.random.permutation(np.arange(num))
    trainIndex = index[:int(num*ratio)]
    testIndex = index[int(num*ratio):]
    return trainIndex, testIndex

# Data split: takes two ndarrays: X, Y and ratio of training set. Returns (X_train, X_test, Y_train, Y_test)
def dataSplit(X, Y, ratio = 0.8):
    # Assume X.shape = (time_span, channel, sample), Y.shape = (sample, 4124)
    # the first dimension of Y is sample size, so as the third dimension of X
    trainIndex, testIndex = getIndex(Y.shape[0], ratio)
    # transpose X.shape to be (sample, channel, time_span)
    X = X.transpose((2,1,0))
    X_train = X[trainIndex,:,:]
    Y_train = Y[trainIndex,:]
    X_test = X[testIndex,:,:]
    Y_test = Y[testIndex,:]
    return X_train, X_test, Y_train, Y_test

# standardize the first dimension(sample) of Y_train, return  Y_train_std, element_mean, element_std
def standardizeYData(Y_train):
    element_mean = np.mean(Y_train, axis=0, keepdims=True)
    element_std = np.std(Y_train, axis=0, keepdims=True)
    Y_train_std = (Y_train - element_mean) / element_std
    return Y_train_std, element_mean, element_std

# Run a PCA model to Y_train_std, inputs number of components to keep, whether to perform error analysis
# return PCA components and model
def pcaYTrain(Y_train_std, nc = 10, err_ana = True):
    pca_model = PCA(n_components=nc)
    pca_model.fit(Y_train_std)
    Y_train_pca = pca_model.transform(Y_train_std)
    if(err_ana):
        Y_reconstruct = pca_model.inverse_transform(Y_train_pca)
        r2 = r2_score(Y_train_std, Y_reconstruct)
        print("The R^2 of PCA reconstruction is", r2)
        print("original({}, {})".format(np.min(Y_train_std), np.max(Y_train_std)),
              ", reconstruct({}, {})".format(np.min(Y_reconstruct), np.max(Y_reconstruct)))
    return Y_train_pca, pca_model


# reconstruct Y elements from standardized data by Y * std + mean
def reconstructYFromStd(Y, mean, std):
    assert Y.shape[1] == 4124, "Y dimension error"
    assert mean.shape[1] == std.shape[1] == 4124, "mean/std dimension error"
    return Y * std + mean

# reconstruct Y from PCA
def reconstructYFromPCA(Y, pca_model):
    assert Y.shape[1] == pca_model.n_components, "component number does not match"
    return pca_model.inverse_transform(Y)

# reconstruct Y from PCA and standardization
def reconstructY(Y, mean, std, pca_model):
    Y_invert_pca = reconstructYFromPCA(Y, pca_model)
    return reconstructYFromStd(Y_invert_pca, mean, std)

# Augmentation of data: by copying data three times and add
# 1.nothing 2. noise ~ N(0,0.01STD)^2) 3. noise ~ N(0,0.02STD)^2)
# to the training predictors, and copy training targets three times
def train_augment(trainX, trainY):
    row = len(trainX[:, 0])
    column = len(trainX[0, :])
    standard_deviation = np.std(trainX, 0)
    add_01 = np.zeros([row, column])
    add_02 = np.zeros([row, column])
    for id in range(0, row):
        add_01[id, :] = 0.01 * standard_deviation * np.random.randn(1, column)
        add_02[id, :] = 0.02 * standard_deviation * np.random.randn(1, column)
    augment_trainX = np.row_stack((trainX, trainX + add_01, trainX + add_02))
    augment_trainY = np.row_stack((trainY, trainY, trainY))
    return augment_trainX, augment_trainY


# flatten X to be the shape (sample, channel * ms) (for cnn model)
def flatten_X(X):
    assert X.shape[1] == 6, "X channel number error"
    assert X.shape[2] == 100, "X timespan length error"
    return X.reshape(X.shape[0],X.shape[1]*X.shape[2])
