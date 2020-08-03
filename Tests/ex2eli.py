from __future__ import print_function
# import torch
# import torch.nn as nn
import keras
import random
import numpy as np

REQUIRED_ROW_LENGTH = 1000
UNDEFINED = -1


class SamplesNet(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(SamplesNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        # self._TP = UNDEFINED
        # self._FP = UNDEFINED
        # self._FN = UNDEFINED
        # self._F025 = UNDEFINED

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


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

    def __init__(self, tuple_list_of_samples):
        self.tuple_list_of_samples = tuple_list_of_samples
        self.__x1_data = UNDEFINED
        self.__x2_data = UNDEFINED
        self.__x3_data = UNDEFINED
        self.__x4_data = UNDEFINED
        self.__x1_normalized_data = UNDEFINED
        self.__x2_normalized_data = UNDEFINED
        self.__x3_normalized_data = UNDEFINED
        self.__x4_normalized_data = UNDEFINED
        self.__all_normalized_data = UNDEFINED

    def get_x1_data(self):
        return self.__x1_data

    def get_x2_data(self):
        return self.__x2_data

    def get_x3_data(self):
        return self.__x3_data

    def get_x4_data(self):
        return self.__x4_data

    def get_x1_normalized_data(self):
        return self.__x1_normalized_data

    def get_x2_normalized_data(self):
        return self.__x2_normalized_data

    def get_x3_normalized_data(self):
        return self.__x3_normalized_data

    def get_x4_normalized_data(self):
        return self.__x4_normalized_data

    def get_all_normalized_data(self):
        return self.__all_normalized_data

    def set_x1_data(self, list_of_data):
        self.__x1_data = list_of_data

    def set_x2_data(self, list_of_data):
        self.__x2_data = list_of_data

    def set_x3_data(self, list_of_data):
        self.__x3_data = list_of_data

    def set_x4_data(self, list_of_data):
        self.__x4_data = list_of_data

    def set_x1_normalized_data(self, list_of_data):
        self.__x1_normalized_data = list_of_data

    def set_x2_normalized_data(self, list_of_data):
        self.__x2_normalized_data = list_of_data

    def set_x3_normalized_data(self, list_of_data):
        self.__x3_normalized_data = list_of_data

    def set_x4_normalized_data(self, list_of_data):
        self.__x4_normalized_data = list_of_data

    def set_all_normalized_data(self):
        self.__all_normalized_data = self.__x1_normalized_data + self.__x2_normalized_data + \
                                     self.__x3_normalized_data + self.__x4_normalized_data


class BestNet:
    def __init__(self):
        self.__best = UNDEFINED
        self.__optimizer = UNDEFINED

    def get_optimizer(self):
        return self.__optimizer

    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer

    def get_best(self):
        return self.__best

    def set_best(self, net):
        self.__best = net


def create_columns():
    # create list of 120 lists, each one will represent a column
    column_list, anomalies = [], []
    for i in range(0, 120):
        column_list.append([])
    row_number = 0
    # fill lists with the columns
    with open('train.csv', 'r') as csv_f:
        for row in csv_f:
            line = row.split(",")
            if line[0].isnumeric():
                anomalies.append([float(line[0])])
            else:
                anomalies.append([(line[0])])
            for i in range(1, 121):
                column_list[i - 1].append(float(line[i]))
            row_number += 1
            if row_number == REQUIRED_ROW_LENGTH:
                break

    return column_list, anomalies


def normalize_data(row_dictionary):
    x1_random_values = []
    x2_random_values = []
    x3_random_values = []
    x4_random_values = []

    for row_data in row_dictionary:
        # get one value of x1,x2,x3,x4 from each row
        x1_to_append = random.sample(row_data.get_x1_data(), 1)[0]
        x1_random_values.append(x1_to_append)
        x2_to_append = random.sample(row_data.get_x2_data(), 1)[0]
        x2_random_values.append(x2_to_append)
        x3_to_append = random.sample(row_data.get_x3_data(), 1)[0]
        x3_random_values.append(x3_to_append)
        x4_to_append = random.sample(row_data.get_x4_data(), 1)[0]
        x4_random_values.append(x4_to_append)

    # evaluate mean, stdev fot x1-x4
    mean_x1 = np.mean(x1_random_values)
    std_x1 = np.std(x1_random_values)
    mean_x2 = np.mean(x2_random_values)
    std_x2 = np.std(x2_random_values)
    mean_x3 = np.mean(x3_random_values)
    std_x3 = np.std(x3_random_values)
    mean_x4 = np.mean(x4_random_values)
    std_x4 = np.std(x4_random_values)

    # normalize by (x-mean)/dev
    counter = 0
    for row_data in row_dictionary:
        # temp lists
        x1_normalized_data_tmp = []
        x2_normalized_data_tmp = []
        x3_normalized_data_tmp = []
        x4_normalized_data_tmp = []

        for x1 in row_data.get_x1_data():
            normalized_x1_value = (x1 - mean_x1) / std_x1
            x1_normalized_data_tmp.append(normalized_x1_value)
        row_data.set_x1_normalized_data(x1_normalized_data_tmp)
        for x2 in row_data.get_x2_data():
            normalized_x2_value = (x2 - mean_x2) / std_x2
            x2_normalized_data_tmp.append(normalized_x2_value)
        row_data.set_x2_normalized_data(x2_normalized_data_tmp)
        for x3 in row_data.get_x3_data():
            normalized_x3_value = (x3 - mean_x3) / std_x3
            x3_normalized_data_tmp.append(normalized_x3_value)
        row_data.set_x3_normalized_data(x3_normalized_data_tmp)
        for x4 in row_data.get_x4_data():
            normalized_x4_value = (x4 - mean_x4) / std_x4
            x4_normalized_data_tmp.append(normalized_x4_value)
        row_data.set_x4_normalized_data(x4_normalized_data_tmp)
        counter += 1
        if counter % 5000 == 0:
            print("line ", counter, "normalized by mean, std")

    # normalize by min-max 0,1
    counter = 0
    for row_data in row_dictionary:
        # temp lists
        x1_normalized_data_tmp = []
        x2_normalized_data_tmp = []
        x3_normalized_data_tmp = []
        x4_normalized_data_tmp = []

        x1_normalized_data = row_data.get_x1_normalized_data()
        x2_normalized_data = row_data.get_x2_normalized_data()
        x3_normalized_data = row_data.get_x3_normalized_data()
        x4_normalized_data = row_data.get_x4_normalized_data()

        for x1_normalized in x1_normalized_data:
            normalized_x1_value = (x1_normalized - min(x1_normalized_data)) / (
                        max(x1_normalized_data) - min(x1_normalized_data))
            x1_normalized_data_tmp.append(normalized_x1_value)
        row_data.set_x1_normalized_data(x1_normalized_data_tmp)
        for x2_normalized in x2_normalized_data:
            normalized_x2_value = (x2_normalized - min(x2_normalized_data)) / (
                        max(x2_normalized_data) - min(x2_normalized_data))
            x2_normalized_data_tmp.append(normalized_x2_value)
        row_data.set_x2_normalized_data(x2_normalized_data_tmp)
        for x3_normalized in x3_normalized_data:
            normalized_x3_value = (x3_normalized - min(x3_normalized_data)) / (
                        max(x3_normalized_data) - min(x3_normalized_data))
            x3_normalized_data_tmp.append(normalized_x3_value)
        row_data.set_x3_normalized_data(x3_normalized_data_tmp)
        for x4_normalized in x4_normalized_data:
            normalized_x4_value = (x4_normalized - min(x4_normalized_data)) / (
                        max(x4_normalized_data) - min(x4_normalized_data))
            x4_normalized_data_tmp.append(normalized_x4_value)
        row_data.set_x4_normalized_data(x4_normalized_data_tmp)

        row_data.set_all_normalized_data()
        counter += 1
        if counter % 5000 == 0:
            print("line ", counter, "normalized")
    print("normalization done")
    # now xi_normalized_data is normalized by min max 0-1, for all the rows in the dictionary


def read_data(row_dictionary):
    columns, anomalies = create_columns()

    # create dictionary of the row of listOfNodes : anomaly
    row_number = 0

    for i in range(REQUIRED_ROW_LENGTH):
        # anomaly = 0/1
        anomaly = anomalies[row_number]
        list_of_samples = []
        for j in range(0, 120, 4):
            sample = Sample(columns[j][i], columns[j + 1][i], columns[j + 2][i], columns[j + 3][i])
            list_of_samples.append(sample)
        row_data = RowData(tuple(list_of_samples))
        row_dictionary[row_data] = anomaly
        if row_number % 5000 == 0:
            print("inserted row data: ", row_number)
        row_number += 1

    # set x1,x2,x3,x4 lists
    for row_data in row_dictionary:
        x1_list = []
        x2_list = []
        x3_list = []
        x4_list = []

        for sample in row_data.tuple_list_of_samples:
            x1_list.append(sample.get_value1())
            x2_list.append(sample.get_value2())
            x3_list.append(sample.get_value3())
            x4_list.append(sample.get_value4())

        row_data.set_x1_data(x1_list)
        row_data.set_x2_data(x2_list)
        row_data.set_x3_data(x3_list)
        row_data.set_x4_data(x4_list)


def predict(output):
    if output > 0.51:
        return 1
    else:
        return 0


def f_025(tp, fp, fn):
    if tp + fp == 0 or tp + fn == 0 or tp == 0:
        return 0.5
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    temp = (precision * recall) / (0.25 ** 2 * precision + recall)
    f025 = temp * (1 + 0.25 ** 2)
    return f025


def diff_loss(output, target):
    return target - output


def main():
    row_dictionary = {}
    read_data(row_dictionary)
    normalize_data(row_dictionary)

    # Our model
    net1 = SamplesNet(120, 100, 70, 1)
    net2 = SamplesNet(120, 100, 70, 1)
    net3 = SamplesNet(120, 100, 70, 1)
    net4 = SamplesNet(120, 100, 70, 1)
    net5 = SamplesNet(120, 100, 70, 1)

    # Out loss function
    loss = nn.MSELoss()
    # Our optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(net1.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0)
    # optimizer2 = torch.optim.SGD(net2.parameters(), lr=learning_rate)
    # optimizer3 = torch.optim.SGD(net3.parameters(), lr=learning_rate)
    # optimizer4 = torch.optim.SGD(net4.parameters(), lr=learning_rate)
    # optimizer5 = torch.optim.SGD(net5.parameters(), lr=learning_rate)

    nets_population = [net1, net2, net3, net4, net5]
    best = BestNet()
    num_epochs = 500  # check if best improves

    file_counter = 0
    epoch_counter = 0

    for epoch in range(num_epochs):

        net1_TP = net1_FP = net1_FN = net_1_F025 = 0
        net2_TP = net2_FP = net2_FN = net_2_F025 = 0
        net3_TP = net3_FP = net3_FN = net_3_F025 = 0
        net4_TP = net4_FP = net4_FN = net_4_F025 = 0
        net5_TP = net5_FP = net5_FN = net_5_F025 = 0
        mutation_1_correct = mutation_2_correct = mutation_3_correct = mutation_4_correct = 0
        mutation_1_loss = mutation_2_loss = mutation_3_loss = mutation_4_loss = 0

        # divide file length to fitness calculation until file_partition, and the rest for 4 mutations
        file_partition = random.randint(REQUIRED_ROW_LENGTH / 2 - REQUIRED_ROW_LENGTH / 10,
                                        REQUIRED_ROW_LENGTH / 2 + REQUIRED_ROW_LENGTH / 10)
        file_mutation_limit = int((REQUIRED_ROW_LENGTH - file_partition) / 4)
        file_mutation_1_limit = file_partition + file_mutation_limit
        file_mutation_2_limit = file_mutation_1_limit + file_mutation_limit
        file_mutation_3_limit = file_mutation_2_limit + file_mutation_limit

        net1_l = []
        net2_l = []
        net3_l = []
        net4_l = []
        net5_l = []
        file_counter = 0
        for row_data in row_dictionary:
            # calculate fitness and determine best
            row_tensor = torch.tensor(row_data.get_all_normalized_data(), dtype=torch.float32)  # 120
            actual_res = int(row_dictionary[row_data])

            if file_counter < file_partition:
                net1 = nets_population[0]
                net1_output = net1(row_tensor)
                net1_l.append(net1_output)
                net1_output = predict(net1_output)
                if net1_output == 1 and actual_res == 1:
                    net1_TP += 1
                if net1_output == 0 and actual_res == 1:
                    net1_FP += 1
                if net1_output == 1 and actual_res == 0:
                    net1_FN += 1
                net2 = nets_population[1]
                net2_output = net2(row_tensor)
                net2_l.append(net2_output)
                net2_output = predict(net2_output)
                if net2_output == 1 and actual_res == 1:
                    net2_TP += 1
                if net2_output == 0 and actual_res == 1:
                    net2_FP += 1
                if net2_output == 1 and actual_res == 0:
                    net2_FN += 1
                net3 = nets_population[2]
                net3_output = net3(row_tensor)
                net3_l.append(net3_output)
                net3_output = predict(net3_output)
                if net3_output == 1 and actual_res == 1:
                    net3_TP += 1
                if net3_output == 0 and actual_res == 1:
                    net3_FP += 1
                if net3_output == 1 and actual_res == 0:
                    net3_FN += 1
                net4 = nets_population[3]
                net4_output = net4(row_tensor)
                net4_l.append(net4_output)
                net4_output = predict(net4_output)
                if net4_output == 1 and actual_res == 1:
                    net4_TP += 1
                if net4_output == 0 and actual_res == 1:
                    net4_FP += 1
                if net4_output == 1 and actual_res == 0:
                    net4_FN += 1
                net5 = nets_population[4]
                net5_output = net5(row_tensor)
                net5_l.append(net5_output)
                net5_output = predict(net5_output)
                if net5_output == 1 and actual_res == 1:
                    net5_TP += 1
                if net5_output == 0 and actual_res == 1:
                    net5_FP += 1
                if net5_output == 1 and actual_res == 0:
                    net5_FN += 1

            elif file_counter == file_partition:
                net_1_F025 = f_025(net1_TP, net1_FP, net1_FN)
                net_2_F025 = f_025(net2_TP, net2_FP, net2_FN)
                net_3_F025 = f_025(net3_TP, net3_FP, net3_FN)
                net_4_F025 = f_025(net4_TP, net4_FP, net4_FN)
                net_5_F025 = f_025(net5_TP, net5_FP, net5_FN)

                max_f025 = max(net_1_F025, net_2_F025, net_3_F025, net_4_F025, net_5_F025)
                print("max f0.25 is ", max_f025)
                if max_f025 == net_1_F025:
                    best.set_best(net1)
                    # best.set_optimizer(optimizer1)
                elif max_f025 == net_2_F025:
                    best.set_best(net2)
                    # best.set_optimizer(optimizer2)
                elif max_f025 == net_3_F025:
                    best.set_best(net3)
                    # best.set_optimizer(optimizer3)
                elif max_f025 == net_4_F025:
                    best.set_best(net4)
                    # best.set_optimizer(optimizer4)
                elif max_f025 == net_5_F025:
                    best.set_best(net5)
                    # best.set_optimizer(optimizer5)
                nets_population.clear()
                nets_population.append(best.get_best())
                # mutations
                orig_best = best
                best_net = best.get_best()
                best_net.train()  # Put the network into training mode - maybe unnecessary

            # 1
            elif file_counter < file_mutation_1_limit and file_counter > file_partition:
                output = best_net(row_tensor)
                actual_res_tensor = torch.tensor([actual_res], dtype=torch.float32)
                mutation_1_loss += loss(output, actual_res_tensor)

                # if (output == 1 and actual_res == 1) or (output == 0 and actual_res == 0):
                #     mutation_1_correct += 1

            elif file_counter == file_mutation_1_limit:
                # best_optimizer = best.get_optimizer()
                # accuracy = mutation_1_correct / (file_mutation_1_limit - file_partition)
                # l = loss(torch.tensor([accuracy]), torch.tensor([1]))  # Calculate the loss,
                # how many correct answers are in the distance of one mutation size
                mutation_1_loss.backward()  # Calculate the gradients with help of back propagation
                optimizer.step()  # Ask the optimizer to adjust the parameters based on the gradients
                optimizer.zero_grad()  # Clear off the gradients from any past operation
                mutation_1 = best.get_best()
                best_net = orig_best.get_best()
                nets_population.append(mutation_1)
            # 2
            elif file_counter < file_mutation_2_limit and file_counter > file_mutation_1_limit:
                output = best_net(row_tensor)
                actual_res_tensor = torch.tensor([actual_res], dtype=torch.float32)
                mutation_2_loss += loss(output, actual_res_tensor)

            elif file_counter == file_mutation_2_limit:
                mutation_2_loss.backward()  # Calculate the gradients with help of back propagation
                optimizer.step()  # Ask the optimizer to adjust the parameters based on the gradients
                optimizer.zero_grad()  # Clear off the gradients from any past operation
                mutation_2 = best.get_best()
                best_net = orig_best.get_best()
                nets_population.append(mutation_2)

            # 3
            elif file_counter < file_mutation_3_limit and file_counter > file_mutation_2_limit:
                output = best_net(row_tensor)
                actual_res_tensor = torch.tensor([actual_res], dtype=torch.float32)
                mutation_3_loss += loss(output, actual_res_tensor)

            elif file_counter == file_mutation_3_limit:
                mutation_3_loss.backward()  # Calculate the gradients with help of back propagation
                optimizer.step()  # Ask the optimizer to adjust the parameters based on the gradients
                optimizer.zero_grad()  # Clear off the gradients from any past operation
                mutation_3 = best.get_best()
                best_net = orig_best.get_best()
                nets_population.append(mutation_3)

            # 4
            elif file_counter < REQUIRED_ROW_LENGTH - 1 and file_counter > file_mutation_3_limit:
                output = best_net(row_tensor)
                actual_res_tensor = torch.tensor([actual_res], dtype=torch.float32)
                mutation_4_loss += loss(output, actual_res_tensor)


            elif file_counter == REQUIRED_ROW_LENGTH - 1:
                mutation_4_loss.backward()  # Calculate the gradients with help of back propagation
                optimizer.step()  # Ask the optimizer to adjust the parameters based on the gradients
                optimizer.zero_grad()  # Clear off the gradients from any past operation
                mutation_4 = best.get_best()
                best_net = orig_best.get_best()
                nets_population.append(mutation_4)

            file_counter += 1

        best.get_best().eval()  # Put the network into evaluation mode - maybe unnecessary
        # after every epoch, check best on validation file and store best and f0.25 score on new scores_dict

        # get very best
        epoch_counter += 1

    # output best to test file

    print("done")


if __name__ == '__main__':
    main()
