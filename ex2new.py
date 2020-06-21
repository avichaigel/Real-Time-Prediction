import numpy as np
from gplearn import genetic, fitness
import os
import pandas as pd
import sklearn

REQUIRED_ROW_LENGTH = 10000


class Sample:
    def __init__(self, value1, value2, value3, value4):
        self.value1 = value1
        self.value2 = value2
        self.value3 = value3
        self.value4 = value4


def create_columns(file):
    # create list of 120 lists, each one will represent a column
    column_list, anomalies = [], []
    for i in range(0, 119):
        column_list.append([])
    row_number = 0
    # fill lists with the columns
    with open(file, 'r') as csv_f:
        for row in csv_f:
            row_number += 1
            line = row.split(",")
            anomalies.append((line[0]))
            for i in range(1, 120):
                column_list[i - 1].append(float(line[i]))
            if row_number == REQUIRED_ROW_LENGTH:
                break
    return column_list, anomalies


def normalize(column_list):
    for column in column_list:
        col_mean = np.mean(column)
        col_std = np.std(column)
        length = len(column)
        for i in range(length):
            column[i] = (column[i] - col_mean) / col_std



def normalize_data(file):
    column_list, anomalies = create_columns(file)
    normalize(column_list)
    return column_list, anomalies


def read_data(row_dictionary):
    columns, anomalies = normalize_data()
    # create dictionary of the row of listOfNodes : anomaly
    row_number = 0
    for i in range(REQUIRED_ROW_LENGTH):
        # anomaly = 0/1
        anomaly = anomalies[row_number]
        list_of_samples = []
        for j in range(0, 120, 4):
            sample = Sample(columns[j][i], columns[j + 1][i], columns[j + 2][i], columns[j + 3][i])
            list_of_samples.append(sample)
        row_dictionary[tuple(list_of_samples)] = anomaly
        row_number += 1


def train(columns, anomalies):
    gp = genetic.SymbolicRegressor(population_size=6, stopping_criteria=0.1, tournament_size=2,
                                   const_range=(-10, 10), function_set=('add', 'sub', 'mul', 'div'),
                                   p_crossover=0.5, p_subtree_mutation=0, p_hoist_mutation=0,
                                   p_point_mutation=0, p_point_replace=0, low_memory=False)
    gp.fit(columns, anomalies)
    gp.predict()


def main():
    columns, anomalies = read_data('train.csv')
    train(columns, anomalies)


if __name__ == '__main__':
    main()
