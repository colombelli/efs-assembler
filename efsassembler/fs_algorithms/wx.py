import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras import optimizers,applications, callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
import functools
import time


"""

The following class, as well as wx_slp() and naive_SLP_model() functions, were taken and modified 
from DearWXpub GitHub repository, https://github.com/deargen/DearWXpub, which protects the code,
at the time of writing, under MIT license.

The modifications include training verbose mode set to false; and the parameters changed name of Keras 
Model initiation for compatibility purposes ('input' became 'inputs', and 'output' became 'outputs').

"""

class WxHyperParameter(object):
    """
    wx feature selector hyperparameters
    """
    def __init__(self, epochs=25, batch_size=10, learning_ratio=0.01, weight_decay=1e-6, momentum=0.9, num_hidden_layer = 2, num_h_unit = 128, verbose=False):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_ratio = learning_ratio
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.verbose = verbose
        self.num_hidden_layer = num_hidden_layer
        self.num_h_unit = num_h_unit



wx_hyperparam = WxHyperParameter(learning_ratio=0.001)

def wx_slp(x_train, y_train, x_val, y_val, n_selection=100, hyper_param=wx_hyperparam, num_cls=2):
    if num_cls < 2:
        return

    input_dim = len(x_train[0])

    # make model and do train
    model = naive_SLP_model(x_train, y_train, x_val, y_val, hyper_param=hyper_param, num_cls=num_cls)

    #load weights
    weights = model.get_weights()

    #cacul WX scores
    num_data = {}
    running_avg={}
    tot_avg={}
    Wt = weights[0].transpose() #all weights of model
    Wb = weights[1].transpose() #all bias of model
    for i in range(num_cls):
        tot_avg[i] = np.zeros(input_dim) # avg of input data for each output class
        num_data[i] = 0.
    for i in range(len(x_train)):
        c = y_train[i].argmax()
        x = x_train[i]
        tot_avg[c] = tot_avg[c] + x
        num_data[c] = num_data[c] + 1
    for i in range(num_cls):
        tot_avg[i] = tot_avg[i] / num_data[i]

    #for general multi class problems
    wx_mul = []
    for i in range(0,num_cls):
        wx_mul_at_class = []
        for j in range(0,num_cls):
            wx_mul_at_class.append(tot_avg[i] * Wt[j])
        wx_mul.append(wx_mul_at_class)
    wx_mul = np.asarray(wx_mul)

    wx_abs = np.zeros(Wt.shape[1])
    for n in range(0, Wt.shape[1]):
        for i in range(0,num_cls):
            for j in range(0,num_cls):
                if i != j:
                    wx_abs[n] += np.abs(wx_mul[i][i][n] - wx_mul[i][j][n])

    selected_idx = np.argsort(wx_abs)[::-1][0:n_selection]
    selected_weights = wx_abs[selected_idx]

    K.clear_session()

    return selected_idx, selected_weights


def naive_SLP_model(x_train, y_train, x_val, y_val, hyper_param=wx_hyperparam, num_cls=2):
    input_dim = len(x_train[0])
    inputs = Input((input_dim,))
    fc_out = Dense(num_cls,  activation='softmax')(inputs)
    model = Model(inputs=inputs, outputs=fc_out)

    #build a optimizer
    sgd = optimizers.SGD(lr=hyper_param.learning_ratio, decay=hyper_param.weight_decay, momentum=hyper_param.momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])        

    #call backs
    def step_decay(epoch):
        exp_num = int(epoch/10)+1       
        return float(hyper_param.learning_ratio/(10 ** exp_num))

    best_model_path="./slp_wx_weights_best"+".hdf5"
    change_lr = LearningRateScheduler(step_decay)                                

    if len(x_train) + len(x_val) < 10 :
        save_best_model = ModelCheckpoint(best_model_path, monitor="loss", verbose=0, save_best_only=True, mode='min')
        if len(x_val) != 0 :         
            x_train = np.concatenate((x_train, x_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)
        #run train
        history = model.fit(x_train, y_train, verbose=hyper_param.verbose,
                    epochs=hyper_param.epochs, batch_size=hyper_param.batch_size, shuffle=True, callbacks=[save_best_model, change_lr])

    else :
        save_best_model = ModelCheckpoint(best_model_path, monitor="val_loss", verbose=0, save_best_only=True, mode='min')
        history = model.fit(x_train, y_train, validation_data=(x_val,y_val), verbose=hyper_param.verbose,
                    epochs=hyper_param.epochs, batch_size=hyper_param.batch_size, shuffle=True, callbacks=[save_best_model, change_lr])

    #load best model
    model.load_weights(best_model_path)

    return model




def build_rank(df, sel_idx):
    
    print("Processing data...")
    genes = list(df.columns)
    data = {}
    data['gene'] = []
    data['rank'] = []
    for i, gen_id in enumerate(sel_idx):
        data['gene'].append(genes[gen_id])
        data['rank'].append(i+1)

    rank = pd.DataFrame(data, columns=['rank']).set_index(pd.Index(data['gene']))
    return rank


def to_categorical(anno_class):
    
    anno_y = []
    for cls in anno_class:
        new_y = [0, 0]
        new_y[int(cls)] = 1
        anno_y.append(new_y)
        
    return np.array(anno_y, dtype=float)


def select(df):
    
    # Gets one-hot encoded y labels
    anno_class = df['class'].values
    y_true = to_categorical(anno_class)

    # Gets x data (gene expressions)
    x_all = df.iloc[:, 0:-1].values

    print("Ranking features with Wx algorithm...")
    # Note: the validation data used here is the same as the training since we are not
    # interested in validation at this point of the experiments run
    hp = WxHyperParameter(epochs=30, learning_ratio=0.01, batch_size=8, verbose=False)
    sel_idx, sel_weight = wx_slp(x_all, y_true, x_all, y_true, n_selection=len(df.columns)-1, 
                                    hyper_param=hp, num_cls=2)

    return build_rank(df, sel_idx)
    