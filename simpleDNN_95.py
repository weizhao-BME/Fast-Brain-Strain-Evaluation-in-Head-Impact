# MLP for Pima Indians Dataset Serialize to JSON and HDF5
#This model adds truncation
from keras.models import Sequential
from keras import optimizers, initializers, regularizers
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from pandas import Series
import tensorflow
from data_preprocessing import *
import os
import scipy.io as io
import matplotlib.pyplot as plt
# fix random seed for reproducibility
os.chdir('C:\\Users\\zhanx\\OneDrive\\Desktop\\Stanford\\Deep Learning\\Project 2 Data\\Project2 Data\\Project')
print(os.getcwd())
np.random.seed(7)
#Data expansion by adding noises
def train_augment(trainX, trainY):
  row = len(trainX[:,0])
  column = len(trainX[0,:])
  standard_deviation = np.std(trainX,0)
  add_01 = np.zeros([row,column])
  add_02 = np.zeros([row,column])
  add_03 = np.zeros([row,column])
  for id in range(0,row):
      add_01[id,:] = 0.01 * standard_deviation * np.random.randn(1, column)
      add_02[id,:] = 0.02 * standard_deviation * np.random.randn(1, column)
      add_03[id,:] = 0.05 * standard_deviation * np.random.randn(1, column)
  augment_trainX = np.row_stack((trainX,trainX+add_01,trainX+add_02, trainX+add_03))
  augment_trainY = np.row_stack((trainY,trainY,trainY,trainY))
  return augment_trainX, augment_trainY

def yHatYPlot(Y_true, Y_predict, title = 'y, y_hat plot 95 MPS',sampleIndex=0):
    plt.scatter(Y_true, Y_predict, s=0.5)
    max_range = np.max([np.max(Y_true), np.max(Y_predict)])
    x_line = np.linspace(0,max_range,100)
    plt.plot(x_line, x_line, color='r')
    plt.title(title)
    plt.ylabel('Predicted MPS')
    plt.xlabel('KTH MPS')
    plt.show()

# load pima indians dataset
for repeat in range(0,50):
  print("Loading:")
  task = "mix95"
  MMA_X = io.loadmat("MMA_X_hand.mat")["MMA_X"]
  print(np.shape(MMA_X))
  MMA_Y = np.loadtxt("label_95.txt").reshape(-1,1)
  X = io.loadmat("AF_X_hand.mat")["AF_X"]
  X = np.row_stack((MMA_X,X))
  print('X shape: {}'.format(np.shape(X)))
  Y = io.loadmat("AF_Y_95.mat")["AF_Y_95"]
  Y = np.delete(Y, [351], axis=0)
  print(np.shape(MMA_Y))
  Y = np.row_stack((MMA_Y, Y))
  print('Y shape: {}'.format(np.shape(Y)))
  augment = "augment"
  #augment = ""
  #standardization = 'standardization'
  standardization = 'nostandardization'
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=(repeat+1)*10)
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1,random_state=(repeat+1)*10)
  #Y_train_std, mean, std = standardizeYData(Y_train)
  label_num = 1
  print(np.shape(Y_test))
  test_num, feature_num = Y_test.shape[0], 36  #Sample number is the real sample/not by variation.
  input_nodes = 36
  first_hidden_layer = 60 #60/50/45/40/35/30/25
  second_hidden_layer =30 #40/35/30/25
  lr = 0.0003
  #third_hidden_layer = 25 #25
  output_nodes = label_num
  epoch = 600
  regularization = 0.007
  #xavier = initializers.glorot_normal(seed = 2020)
  initialization = "normal"
  loss = "mean_squared_error"

  augment_trainX, augment_trainY = train_augment(X_train, Y_train)
  #augment_trainX, augment_trainY = train_augment(X_train, Y_train_std) #Standardized version
  print("Training X shape:")
  print(augment_trainX.shape)
  print("Training Y shape:")
  print(augment_trainY.shape)
  print("Test X shape:")
  print(X_test.shape)
  print("Test Y shape:")
  print(Y_test.shape)
  print("Start Training:")
# create model
  model= Sequential()
  model.add(Dense(first_hidden_layer, input_dim=input_nodes, kernel_initializer=initialization, kernel_regularizer=regularizers.l2(regularization), activation='relu'))
  model.add(Dense(second_hidden_layer, kernel_initializer=initialization, kernel_regularizer=regularizers.l2(regularization), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(output_nodes, kernel_initializer=initialization))
# Compile model
  Adam = optimizers.adam(learning_rate=lr, decay=1e-6)
  model.compile(loss=loss,optimizer=Adam)
# Fit the model
  import time
  train_start=time.clock()
#model.fit(trainX, trainY, epochs=epoch, batch_size=10, verbose=0) #Fit the model based on current training set, excluding the test sample.
  history = model.fit(augment_trainX, augment_trainY, validation_split=0.2, epochs=epoch, batch_size=128, verbose=0) #Fit the model based on expanded training set, after excluding the test sample.
  train_finished=time.clock()
  #plt.title("learning curve epoch: {}, lr: {}".format(str(epoch),str(lr)))
  #loss, = plt.plot(history.history['loss'])
  #val_loss, = plt.plot(history.history['val_loss'])
  #plt.legend([loss, val_loss], ['loss', 'Val_loss'])
  #plt.show()

  train_time=train_finished-train_start
  print("Training time: %.3f", train_time)
  # evaluate the model
  print("Predicting 95 MPS:")
  predict_start=time.clock()
  print('Predicting:')
  #predicted_Y_val_std = model.predict(X_val)
  #predicted_Y_val = reconstructYFromStd(predicted_Y_val_std, mean, std)
  #predicted_Y_std = model.predict(X_test)
  #predicted_Y = reconstructYFromStd(predicted_Y_std, mean, std)
  predicted_Y = model.predict(X_test)
  predicted_Y_val = model.predict(X_val)
  predict_finished=time.clock()
  predict_time=predict_finished-predict_start
  print("Predicting time: %.3f", predict_time)
  print("Finished predicting and start evaluating: ")
  MAL=mean_absolute_error(Y_train,model.predict(X_train))
  #MAL = mean_absolute_error(Y_train, reconstructYFromStd(model.predict(X_train), mean, std))
  MAE = mean_absolute_error(predicted_Y, Y_test)
  MAV = mean_absolute_error(predicted_Y_val, Y_val)
  MSL=mean_squared_error(Y_train,model.predict(X_train))
  #MSL = mean_squared_error(Y_train,reconstructYFromStd(model.predict(X_train), mean, std))
  RMSL = np.sqrt(MSL)
  MSE=mean_squared_error(predicted_Y, Y_test)
  RMSE = np.sqrt(MSE)
  MSV = mean_squared_error(predicted_Y_val, Y_val)
  RMSV = np.sqrt(MSV)
  R2_train = r2_score(Y_train,model.predict(X_train))
  #R2_train = r2_score(Y_train,reconstructYFromStd(model.predict(X_train), mean, std))
  R2_test = r2_score(Y_test,predicted_Y)
  R2_val = r2_score(Y_val, predicted_Y_val)

  #yHatYPlot(Y_test, predicted_Y)
  print("MPS MSL: %.2f(%.2f)", MSL)
  print("MPS MSE: %.2f(%.2f)", MSE)
  print("MPS MSV: %.2f(%.2f)", MSV)
  print("MPS MAL: %.2f(%.2f)", MAL)
  print("MPS MAE: %.2f(%.2f)", MAE)
  print("MPS MAV: %.2f(%.2f)", MAV)
  print("MPS RMSL: %.2f(%.2f)", RMSL)
  print("MPS RMSE: %.2f(%.2f)", RMSE)
  print("MPS RMSV: %.2f(%.2f)", RMSV)
  print("MPS R2_train: %.2f(%.2f)", R2_train)
  print("MPS R2_test: %.2f(%.2f)", R2_test)
  print("MPS R2_val: %.2f(%.2f)", R2_val)
#predicted_MPS=predicted_Y
  print(type(predicted_Y))
#initialization = "xavier"
#predict_result = pd.DataFrame({"Predicted MPS": predicted_MPS})
  prediction_path = "C:\\Users\\zhanx\\OneDrive\\Desktop\\Stanford\\Deep Learning\\Project 2 Data\\Project2 Data\\Project\\DNN result\\Prediction_ANN_" + 'repeat' + str(repeat)+ '_' + augment + '_' + 'Dropout0.5_'+ str(standardization) + "_"+task + "_" + str(first_hidden_layer)+"_"+str(second_hidden_layer) +"_"+str(output_nodes)+ "_" + \
                  'lr' + str(lr) +"_"+"epoch"+str(epoch) + "_" + str(regularization) + "_" + str(initialization) + "_" + str(loss)
#predict_result.to_csv(prediction_path, index=True, sep=',')

  io.savemat(prediction_path,{'Prediction':predicted_Y,'Prediction Val': predicted_Y_val, 'Y_test': Y_test,'Y_val': Y_val,'Train Time': train_time,
                            'Prediction time': predict_time,
                            'MSE': MSE, 'MSL': MSL, 'MSV': MSV,
                            'RMSE':RMSE, 'MAL': MAL, 'MAV': MAV,
                            'MAE': MAE, 'RMSL': RMSL, 'RMSV':RMSV,
                             'R2_train': R2_train, 'R2_test': R2_test,'R2_val': R2_val})

# serialize model to JSON
  model_json = model.to_json()
  model_path = "DNN model\\DNN_"+ 'repeat' + str(repeat)+ '_' + augment+ '_' + 'Dropout0.5_'+ str(standardization) + "_"+ task +"_"+ str(first_hidden_layer)+"_"+str(second_hidden_layer)+"_"+str(output_nodes)+ "_" +'lr' + str(lr) +"_"+"epoch"+str(epoch) + "_" + str(regularization) + "_" + str(initialization) + "_" + str(loss)+"_model.json"
  with open(model_path, "w") as json_file:  #Save model skeleton
    json_file.write(model_json)
# serialize weights to HDF5
  weight_path = "DNN model\\ANN_"+ 'repeat' + str(repeat)+ '_' + augment+ '_' + 'Dropout0.5_'+ str(standardization) + "_" + task +  "_" + str(first_hidden_layer)+"_"+str(second_hidden_layer)+"_"+str(output_nodes)+ "_" +'lr' + str(lr) +"_"+"epoch"+str(epoch)+ "_" + str(regularization) + "_" + str(initialization)+ "_" + str(loss)+"_weight.h5"
  model.save_weights(weight_path)  #Save model weight
  print("Saved model to disk")

# later...
'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''