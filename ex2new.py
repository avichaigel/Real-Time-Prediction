import numpy as np
from gplearn import genetic, fitness
import os
import pandas as pd
import sklearn
from sys import argv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

REQUIRED_ROW_LENGTH = 10000


def create_columns(file):
    # create list of 120 lists, each one will represent a column
    column_list, anomalies = [], []
    for i in range(0, 120):
        column_list.append([])
    row_number = 0
    # fill lists with the columns
    with open(file, 'r') as csv_f:
        for row in csv_f:
            row_number += 1
            line = row.split(",")
            if line[0].isnumeric():
                anomalies.append(float(line[0]))
            else:
                anomalies.append((line[0]))
            for i in range(1, 121):
                column_list[i - 1].append(float(line[i]))
            if row_number == REQUIRED_ROW_LENGTH:
                break
    return column_list, anomalies

def train(tr_samples, tr_anomalies, v_anomalies):
    gp = genetic.SymbolicRegressor(population_size=6, stopping_criteria=0.1, tournament_size=2,
                                   const_range=(-10, 10), function_set=('add', 'sub', 'mul', 'div'),
                                   p_crossover=0.5, p_subtree_mutation=0, p_hoist_mutation=0,
                                   p_point_mutation=0, p_point_replace=0, low_memory=False)
    tr_samples = np.array(tr_samples).transpose()
    v_anomalies = np.array(v_anomalies).reshape(1,-1)
    gp.fit(tr_samples, tr_anomalies)
    gp.predict(v_anomalies)
    # gp.score()


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
    tr_samples, tr_anomalies = create_columns('train.csv')
    v_samples, v_anomalies = create_columns('validate.csv')
    test_samples, test_anomalies = create_columns(argv[1])
    tr_samples, v_samples, test_samples = normalize_data(tr_samples, v_samples, test_samples)
    train(tr_samples, tr_anomalies, v_anomalies)


if __name__ == '__main__':
    main()
