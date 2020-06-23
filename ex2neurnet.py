import numpy as np
from gplearn import genetic, fitness
import os
import pandas as pd
import sklearn
from sys import argv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

REQUIRED_ROW_LENGTH = 10000

class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50]]*120)
        self.error_history = []
        self.epoch_list = []

    # activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        exp = np.exp(-x)
        return 1 / (1 + exp)

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for TRAIN_GENS iterations
    def train(self, epochs):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction


def create_rows(file):
    row_list, anomalies = [], []
    row_number = 0
    # fill lists with the columns
    with open(file, 'r') as csv_f:
        for row in csv_f:
            line = row.split(",")
            if line[0].isnumeric():
                anomalies.append([float(line[0])])
            else:
                anomalies.append([(line[0])])
            row_list.append([])
            for i in range(1, 121):
                row_list[row_number].append(float(line[i]))
            row_number += 1
            if row_number == REQUIRED_ROW_LENGTH:
                break
    return row_list, anomalies


def log(tr_samples, v_samples, test_samples):
    return np.log(tr_samples), np.log(v_samples), np.log(test_samples)


def normalize_data(tr_samples, v_samples, test_samples):
    tr_samples, v_samples, test_samples = log(tr_samples, v_samples, test_samples)
    scaler = StandardScaler()
    scaler.fit(tr_samples)
    tr_samples = scaler.transform(tr_samples)
    v_samples = scaler.transform(v_samples)
    test_samples = scaler.transform(test_samples)
    return tr_samples, v_samples, test_samples


def main():
    tr_samples, tr_anomalies = create_rows('train.csv')
    v_samples, v_anomalies = create_rows('validate.csv')
    test_samples, test_anomalies = create_rows(argv[1])
    tr_samples, v_samples, test_samples = normalize_data(tr_samples, v_samples, test_samples)

    neural_net = NeuralNetwork(tr_samples, tr_anomalies)
    neural_net.train(1000)
    print(neural_net.predict(v_samples))

    # plot the error over the entire training duration
    plt.figure(figsize=(15, 5))
    plt.plot(neural_net.epoch_list, neural_net.error_history)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()


if __name__ == '__main__':
    main()
