import sys
import keras
import random
import numpy as np
import sklearn
from h5py._hl import dataset
from numpy import loadtxt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, fbeta_score
import itertools
import matplotlib.pyplot as plt

PAUSE = 1000
PATIENCE = 10


# read the data, and split it into a list of the data(x) and a list of the labels(y)
def create_rows(file):
    x, y = [], []
    row_number = 0
    print("reading " + file)
    # fill lists with the columns
    with open(file, 'r') as csv_f:
        for row in csv_f:
            line = row.split(",")
            if line[0].isnumeric():
                y.append([float(line[0])]) # maybe should be int
            else:
                # in the file "test.csv" there are '?' instead of labels, so it can't be a float
                y.append([(line[0])])
            x.append([])
            for i in range(1, 121):
                x[row_number].append(float(line[i]))
            row_number += 1
            if row_number == PAUSE:
                break
    return x, y


def create_mean_std(tr_samples):
    # create lists for features a,b,c,d to be calculated in
    calc_a, calc_b, calc_c, calc_d = [], [], [], []
    for row in tr_samples:
        # perhaps choosing randomly a seventh of the lines will do the job, didn't implement it here
        for i in range(0, 120, 4):
            calc_a.append(row[i])
            calc_b.append(row[i+1])
            calc_c.append(row[i+2])
            calc_d.append(row[i+3])
    mean_a, std_a = np.mean(calc_a), np.std(calc_a)
    mean_b, std_b = np.mean(calc_b), np.std(calc_b)
    mean_c, std_c = np.mean(calc_c), np.std(calc_c)
    mean_d, std_d = np.mean(calc_d), np.std(calc_d)
    return [(mean_a,std_a),(mean_b,std_b),(mean_c,std_c),(mean_d,std_d)]


# do z-score normalization on each sample
def z_score(samples, meanstd):
    for row in samples:
        for i in range(0, 120, 4):
            row[i] = (row[i] - meanstd[0][0]) / meanstd[0][1]
            row[i + 1] = (row[i + 1] - meanstd[1][0]) / meanstd[1][1]
            row[i + 2] = (row[i + 2] - meanstd[2][0]) / meanstd[2][1]
            row[i + 3] = (row[i + 3] - meanstd[3][0]) / meanstd[3][1]
    return samples


def minmax_for_feature(feature):
    feature_min = np.array(feature).min()
    diff = np.array(feature).max() - feature_min
    for i in range(0, 30):
        feature[i] = (feature[i] - feature_min) / diff

# do min-max normaliaztion by features in rows, meaning we go over every row, and then
# for each feature (a/b/c/d) we do min-max normalization
def minmax(tr_samples):
    for row in tr_samples:
        # for each row collect every feature into a list
        a_in_row, b_in_row, c_in_row, d_in_row = [], [], [], []
        for i in range(0, 120, 4):
            a_in_row.append(row[i])
            b_in_row.append(row[i+1])
            c_in_row.append(row[i+2])
            d_in_row.append(row[i+3])
        # do min max norm for each feature in the row
        minmax_for_feature(a_in_row)
        minmax_for_feature(b_in_row)
        minmax_for_feature(c_in_row)
        minmax_for_feature(d_in_row)
        # update the original list
        for i in range(0, 120, 4):
            row[i] = a_in_row[i%30]
            row[i+1] = b_in_row[i%30]
            row[i+2] = c_in_row[i%30]
            row[i+3] = d_in_row[i%30]
    return tr_samples


def normalize(data, mean_std):
    data = z_score(data, mean_std)
    data = minmax(data)
    return data


def normalize_all_data(tr_samples, v_samples, test_samples):
    print("starting normalization")
    mean_std = create_mean_std(tr_samples) # create mean and stdev to use on all data files
    print("normalizing training data")
    tr_samples = normalize(tr_samples, mean_std)
    print("normalizing validation data")
    v_samples = normalize(v_samples, mean_std)
    print("normalizing test data")
    test_samples = normalize(test_samples, mean_std)
    return tr_samples, v_samples, test_samples

#
# def f_025(tp, fp, fn):
#     if tp + fp == 0 or tp + fn == 0 or tp == 0:
#         return 0.5
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     temp = (precision * recall) / (0.25 ** 2 * precision + recall)
#     f025 = temp * (1 + 0.25 ** 2)
#     return f025
#
#
# def diff_loss(output, target):
#     return target - output
#
#
# def update_vals(nets_population, row_tensor, net_l, actual_res, net_TP, net_FP, net_FN):
#     net = nets_population[0]
#     net_output = net(row_tensor)
#     net_l.append(net_output)
#     net_output = predict(net_output)
#     if net_output == 1 and actual_res == 1:
#         net_TP += 1
#     if net_output == 0 and actual_res == 1:
#         net_FP += 1
#     if net_output == 1 and actual_res == 0:
#         net_FN += 1
#     return net_FN, net_FP, net_TP

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def f_beta(y_true, y_pred):
    return fbeta_score(y_true, y_pred, average='binary', beta=0.25)

def train(tr_samples, tr_anomalies, v_samples, v_anomalies, test_samples, test_anomalies):

    # chroms = []
    # for i in range(0, 4):
    #     chroms.append()

    v_samples = np.array(v_samples)
    v_anomalies = np.array(v_anomalies)
    tr_samples = np.array(tr_samples)
    tr_anomalies = np.array(tr_anomalies)


    # define the keras model
    model = Sequential()
    model.add(Dense(150, input_dim=120, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f_beta])

    # fit_params = {
    #     'x': tr_samples,
    #     'y': tr_anomalies,
    #     'validation_split': 0.1,
    #     'epochs': 300,
    #     'verbose': 1,
    #     'validation_data': (dataset.make_new_dset(v_samples),
    #                         dataset.make_new_dset(v_anomalies)),
    #     'callbacks': [
    #         EarlyStopping(monitor='val_loss',
    #                       patience=PATIENCE,
    #                       verbose=0)
    #     ]
    # }
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    model.fit(x=tr_samples, y=tr_anomalies, epochs=300, verbose=1, validation_data=(v_samples, v_anomalies),
              batch_size=100, callbacks=EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0))
    predictions = model.predict(x=test_samples, batch_size=100, verbose=0)
    rounded_predictions = np.argmax(predictions)
    cm = confusion_matrix(y_true=test_anomalies, y_pred=rounded_predictions)
    cm_plot_labels = ['no_anomaly', 'anomaly']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

def main():
    # create a list of samples and a list of anonmalies for each file, perhaps there is a better way for
    # reading the data. If so, please use it
    tr_samples, tr_anomalies = create_rows('train.csv')
    v_samples, v_anomalies = create_rows('validate.csv')
    test_samples, test_anomalies = create_rows(sys.argv[1]) # test.csv will be given as an argument
    tr_samples, v_samples, test_samples = normalize_all_data(tr_samples, v_samples, test_samples)
    print("normalization done")
    train(tr_samples, tr_anomalies, v_samples, v_anomalies, test_samples, test_anomalies)

if __name__ == "__main__":
    main()