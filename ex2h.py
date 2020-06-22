from sys import argv
import numpy as np
from gplearn import genetic, fitness
import os
import pandas as pd
from sklearn import preprocessing as pp

REQUIRED_ROW_LENGTH = 10000


def create_feature_list(row):
    features, feat_a, feat_b, feat_c, feat_d = [], [], [], [], []
    length = len(row)
    for i in range(0, length, 4):
        feat_a.append(row[i])
        feat_b.append(row[i + 1])
        feat_c.append(row[i + 2])
        feat_d.append(row[i + 3])
    features.append(feat_a)
    features.append(feat_b)
    features.append(feat_c)
    features.append(feat_d)
    return features


def sigmoid(x, deriv = False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def normalize(tr_samples, v_samples, test_samples):
    tr_length = len(tr_samples)
    debug_counter = 0
    for tr_row in tr_samples:
        if debug_counter == 405:
            print(3)
        tr_features = create_feature_list(tr_row)
        mean_a, std_a = np.mean(tuple(tr_features[0])), np.std(tuple(tr_features[0]))
        mean_b, std_b = np.mean(tr_features[1]), np.std(tr_features[1])
        mean_c, std_c = np.mean(tr_features[2]), np.std(tr_features[2])
        mean_d, std_d = np.mean(tr_features[3]), np.std(tr_features[3])
        for i in range(0, 120, 4):
            tr_row[i] = (tr_row[i] - mean_a) / std_a
            tr_row[i + 1] = (tr_row[i + 1] - mean_b) / std_b
            tr_row[i + 2] = (tr_row[i + 2] - mean_c) / std_c
            tr_row[i + 3] = (tr_row[i + 3] - mean_d) / std_d
        debug_counter += 1
    for v_row in v_samples:
        for i in range(0, 120, 4):
            v_row[i] = (v_row[i] - mean_a) / std_a
            v_row[i + 1] = (v_row[i + 1] - mean_b) / std_b
            v_row[i + 2] = (v_row[i + 2] - mean_c) / std_c
            v_row[i + 3] = (v_row[i + 3] - mean_d) / std_d
    for test_row in test_samples:
        for i in range(0, 120, 4):
            test_row[i] = (test_row[i] - mean_a) / std_a
            test_row[i + 1] = (test_row[i + 1] - mean_b) / std_b
            test_row[i + 2] = (test_row[i + 2] - mean_c) / std_c
            test_row[i + 3] = (test_row[i + 3] - mean_d) / std_d
    print(np.argwhere(np.isnan(tr_samples)))


    # tr_features = create_feature_list(tr_samples)
    # v_features = create_feature_list(v_samples)
    # test_features = create_feature_list(test_samples)
    # scale_a = pp.StandardScaler().fit(np.array(tr_features[0]).reshape(-1,1))
    # scale_b = pp.StandardScaler().fit(np.array(tr_features[1]).reshape(-1,1))
    # scale_c = pp.StandardScaler().fit(np.array(tr_features[2]).reshape(-1,1))
    # scale_d = pp.StandardScaler().fit(np.array(tr_features[3]).reshape(-1,1))


def train(tr_samples, tr_anomalies, v_anomalies):
    gp = genetic.SymbolicRegressor(population_size=6, stopping_criteria=0.1, tournament_size=2,
                                   const_range=(-10, 10), function_set=('add', 'sub', 'mul', 'div'),
                                   p_crossover=0.5, p_subtree_mutation=0, p_hoist_mutation=0,
                                   p_point_mutation=0, p_point_replace=0, low_memory=False)
    gp.fit(tr_samples, tr_anomalies)
    gp.predict(v_anomalies)
    # gp.score()


def read_data(file):
    samples, anomalies = [], []
    row_number = 0
    # fill lists with the columns
    with open(file, 'r') as csv_f:
        for row in csv_f:
            row_number += 1
            line = row.split(",")
            anomalies.append((line.pop(0)))
            samples.append(line)
            if row_number == REQUIRED_ROW_LENGTH:
                break
        samples_len = len(samples)
        for i in range(samples_len):
            samples[i] = pd.to_numeric(samples[i])
        return samples, anomalies
    # return [float(i) for i in samples], anomalies
    # return np.array(samples).astype(np.float), anomalies


def main():
    tr_samples, tr_anomalies = read_data('train.csv')
    v_samples, v_anomalies = read_data('validate.csv')
    test_samples, test_anomalies = read_data(argv[1])
    normalize(tr_samples, v_samples, test_samples)
    train(tr_samples, tr_anomalies, v_anomalies)


if __name__ == '__main__':
    main()
