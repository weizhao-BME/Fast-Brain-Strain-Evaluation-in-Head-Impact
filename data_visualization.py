# Visualize original data for "outliers"
import matplotlib.pyplot as plt
import numpy as np
import math
# visualize one sample of data as Time Series
# input one sample of data shape (channel, timespan) and a matplotlib.axes._subplots.AxesSubplot object returned by plt.figure.add_subplot(
# returns time series graph that contains # of channels
def visualizeTS(X, subplot,legend=True):
    for channel in range(X.shape[1]):
        if legend: label = "The {} channel".format(channel+1)
        else: label = '_nolegend_'
        subplot.plot(X[:,channel], label = label)

# visualize nine sample of data as Time Series
# input X of shape (sample, channel, timespan), and indexes of sample to display.
# It could be: 1 index(meaning the nine samples starting from this one),
# or a list of length 9, default to be starting from 0
# Display the graph

def visualize9(X, index=0, num=9):
    if type(index) == int:
        index_range = range(index,index+num)
    elif type(index) == list:
        index_range = index
        assert len(index)==num, "Index length error."
    fig = plt.figure(num)
    # plot subplots, only the first subplot has legend (i==0)
    for i in range(num):
        subplot=fig.add_subplot(3,math.ceil(num/3),i+1)
        subplot.title.set_text("The {}th sample".format(str(index_range[i])))
        visualizeTS(X[index_range[i],:,:],subplot,i==0) # control legend
    fig.legend(loc=0,fancybox=True, framealpha=0.5, fontsize='xx-small')
    fig.show()

def visualizeDataProcessing(X_1, X_2, index=0):
    fig = plt.figure(2)


# visualize Y_pred_pca and Y_pca by two kinds of graph
# graphType to control which type of graph to draw, can be 1, 2 or 'both'
def PCATrueVSPred(Y_pca, Y_pred_pca, sampleIndex=0, graphType='both'):
    if graphType=='both' or graphType==1 or graphType=='one':
        plt.scatter(np.arange(Y_pca.shape[1]), Y_pca[sampleIndex,:],c="blue")
        plt.scatter(np.arange(Y_pred_pca.shape[1]), Y_pred_pca[sampleIndex,:],c="red")
        plt.show()
    elif graphType=='both' or graphType==2 or graphType=='two':
        plt.scatter(Y_pca[sampleIndex,:],Y_pred_pca[sampleIndex,:])
        plt.show()
