import matplotlib.pyplot as plt
import gplearn as gp
import numpy as np

REQUIRED_ROW_LENGTH = 1000
UNDEFINED = -1


class Sample:
    def __init__(self, value1, value2, value3, value4):
        self.__value1 = value1
        self.__value2 = value2
        self.__value3 = value3
        self.__value4 = value4

    def get_value1(self):
        return self.__value1

    def get_value2(self):
        return self.__value2

    def get_value3(self):
        return self.__value3

    def get_value4(self):
        return self.__value4


class RowData:
    def __init__(self, tuple_list_of_samples, m1, m2):
        self.tuple_list_of_samples = tuple_list_of_samples
        self.m1 = m1
        self.m2 = m2


def create_columns():
    # create list of 120 lists, each one will represent a column
    column_list, anomalies = [], []
    for i in range(0, 120):
        column_list.append([])
    row_number = 0
    # fill lists with the columns
    with open('train.csv', 'r') as csv_f:
        for row in csv_f:
            row_number += 1
            line = row.split(",")
            anomalies.append((line[0]))
            for i in range(1, 121):
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


def normalize_data():
    column_list, anomalies = create_columns()
    normalize(column_list)
    return column_list, anomalies


def read_data(row_dictionary):
    columns, anomalies = normalize_data()

    # create dictionary of the row of listOfNodes : anomaly
    row_number = 0

    for i in range(REQUIRED_ROW_LENGTH ):
        # anomaly = 0/1
        anomaly = anomalies[row_number]
        list_of_samples = []
        for j in range(0, 120, 4):
            sample = Sample(columns[j][i], columns[j+1][i], columns[j+2][i], columns[j+3][i])
            list_of_samples.append(sample)
        row_data = RowData(tuple(list_of_samples),UNDEFINED, UNDEFINED)
        row_dictionary[row_data] = anomaly
        row_number += 1


def generate_slopes(row_dictionary):
    # for value in row_dictionary: generate 4 lines
    a1_list = []
    a2_list = []
    a3_list = []
    a4_list = []
    for row_data in row_dictionary:
        print("new row")
        print("anomaly: " + row_dictionary[row_data])  # 0/1
        for sample in row_data.tuple_list_of_samples:
            a1_list.append(sample.get_value1())
            a2_list.append(sample.get_value2())
            a3_list.append(sample.get_value3())
            a4_list.append(sample.get_value4())
            if len(a1_list) == 15:
                plt.plot(a2_list, a1_list)
                plt.show()
                plt.plot(a4_list, a3_list)
                plt.show()
                plt.close()
        plt.plot(a2_list, a1_list)
        plt.show()
        plt.plot(a4_list, a3_list)
        plt.show()
        plt.close()
        a1_list.clear()
        a2_list.clear()
        a3_list.clear()
        a4_list.clear()


# def train(rows):
    # gp.genetic.SymbolicRegressor(population_size=5, const_range=(-10,10), function_set=('add','sub'),
#                                  )

def main():
    row_dictionary = {}
    read_data(row_dictionary)
    # generates graphs for x1,x2,x3,x4 and determines m1, m2
    generate_slopes(row_dictionary)
    # train(row_dictionary)
    print("done12")


if __name__ == '__main__':
    main()


