from __future__ import print_function

import csv
import math
import os
import neat
from sys import argv
from sklearn.preprocessing import StandardScaler
import visualize
import numpy as np
import random

REQUIRED_ROW_LENGTH = 1000


# def create_rows(file):
#     row_list, anomalies = [], []
#     row_number = 0
#     # fill lists with data
#     with open(file, newline='') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             if row[0].isnumeric():
#                 anomalies.append(int(row.pop(0)))
#             else:
#                 anomalies.append(row.pop(0))
#             row_list.append(pd.to_numeric(row))
#     return row_list, anomalies

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
            if row_number == REQUIRED_ROW_LENGTH and file == 'train.csv':
                break
    return row_list, anomalies


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


def normalize_every_fourth(samples, meanstd):
    for row in samples:
        for i in range(0, 120, 4):
            row[i] = (row[i] - meanstd[0][0]) / meanstd[0][1]
            row[i + 1] = (row[i + 1] - meanstd[1][0]) / meanstd[1][1]
            row[i + 2] = (row[i + 2] - meanstd[2][0]) / meanstd[2][1]
            row[i + 3] = (row[i + 3] - meanstd[3][0]) / meanstd[3][1]


def create_mean_std(tr_samples):
    debug_counter = 0
    for tr_row in tr_samples:
        if debug_counter == 405:
            pass
        tr_features = create_feature_list(tr_row)
        mean_a, std_a = np.mean(tr_features[0]), np.std(tr_features[0])
        mean_b, std_b = np.mean(tr_features[1]), np.std(tr_features[1])
        mean_c, std_c = np.mean(tr_features[2]), np.std(tr_features[2])
        mean_d, std_d = np.mean(tr_features[3]), np.std(tr_features[3])
        debug_counter += 1
    # print(np.argwhere(np.isnan(tr_samples)))
    return [(mean_a,std_a),(mean_b,std_b),(mean_c,std_c),(mean_d,std_d)]


def normalize_data(tr_samples, data_to_normalize):
    meanstd = create_mean_std(tr_samples)
    normalize_every_fourth(data_to_normalize, meanstd)


# read and normalize data
tr_samples, tr_anomalies = create_rows('train.csv')
normalize_data(tr_samples, tr_samples)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for row, anomaly in zip(tr_samples, tr_anomalies):
            output = net.activate(row)
            if output == anomaly:
                genome.fitness += 1


def rand_zero_one(x):
    if random.randint(0,1) == 0:
        return 0
    else:
        return 1


def my_sigmoid(z):
    # z = max(-60.0, min(60.0, 5.0 * z))
    res = 1.0 / (1.0 + math.exp(-z))
    if res >= 0.5:
        return 1
    else:
        return 0


def run(config_file):

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_file)

    # config.genome_config.add_activation('my_sigmoid', rand_zero_one)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # # Show output of the most fit genome against training data.
    # print('\nTrain Output:')
    # for xi, xo in zip(tr_samples, tr_anomalies):
    #     tr_output = winner_net.activate(xi)
    #     print("Expected output {!r}, got {!r}".format(xo, tr_output))
    #
    # # Show output of the most fit genome against validation data.
    # print('\n\nValidation Output:')
    # v_samples, v_anomalies = create_rows('validate.csv')
    # normalize_data(tr_samples, v_samples)
    # for row, anomaly in zip(v_samples, v_anomalies):
    #     v_output = winner_net.activate(row)
    #     print("Expected output {!r}, got {!r}".format(anomaly, v_output))

    # Show output of the most fit genome against test data.
    test_results = []
    print('\n\nTest Output:')
    test_samples, test_anomalies = create_rows(argv[1])
    normalize_data(tr_samples, test_samples)
    for row, anomaly in zip(test_samples, test_anomalies):
        test_output = winner_net.activate(row)
        # print("Output: {!r}".format(test_output))
        test_results.append(test_output)
    out_file = open("316096338_308178136.txt","w")
    for row in test_results:
        out_file.write(str(row[0])+"\n")
    out_file.close()

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

def main():
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)


if __name__ == '__main__':
    main()
