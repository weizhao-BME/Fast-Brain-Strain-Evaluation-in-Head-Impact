# time
import time
# plotting
import matplotlib.pyplot as plt
from keras.backend import eval
from numpy.random import seed

# Fit the model: train a model, gives some analysis about the model
# Input: compiled model, X_train, Y_train, epoch, analysis
# Output: fitted model, test error
def modelFit(model, X_train, Y_train, epoch=50, analyze=True):
    seed(9001)
    train_start = time.time()
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=64, verbose=0, shuffle=False)
    train_finish = time.time()
    if analyze:
        print("Spend {} sec training model".format(round(train_finish-train_start,1)))
        plt.title("learning curve epoch: {}, lr: {}".format(str(epoch),str(eval(model.optimizer.lr))))
        loss, = plt.plot(history.history['loss'])
        val_loss, = plt.plot(history.history['val_loss'])
        plt.legend([loss, val_loss],['loss','Val_loss'])
        plt.show()
    return model, history
