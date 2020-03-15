
import numpy as np
import scipy.io as io
import os
import matplotlib.pyplot as plt
# fix random seed for reproducibility
os.chdir('C:\\Users\\zhanx\\OneDrive\\Desktop\\Stanford\\Deep Learning\\Project 2 Data\\Project2 Data\\Project')
print(os.getcwd())
np.random.seed(7)
#Data expansion by adding noises
#R2_record = io.loadmat("statistical.mat")["R2_record"]
RMSE_record = io.loadmat("statistical.mat")["RMSE_record"]
MSE_record = io.loadmat("statistical.mat")["MSE_record"]
MAE_record = io.loadmat("statistical.mat")["MAE_record"]

labels = ['DNN Engineered Features', 'DNN','DNN PCA','LSTM PCA']
#plt.boxplot(x=R2_record, labels=labels, sym = "o")
#plt.boxplot(x=MAE_record, labels=labels, sym = "o")
plt.boxplot(x=RMSE_record, labels=labels, sym = "o")
plt.xlabel('Model')
#plt.ylabel('R2')
#plt.ylabel('MAE')
plt.ylabel('RMSE')
plt.title('MPS Prediction Box Plot')
plt.show()

