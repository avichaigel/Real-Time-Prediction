"""
2-input XOR example -- this is most likely the simplest possible example.
"""
from __future__ import print_function

import os
import neat
from sys import argv
from sklearn.preprocessing import StandardScaler
import visualize
import numpy as np

REQUIRED_ROW_LENGTH = 100


def eval_genomes(genomes, config):
    tr_samples, tr_anomalies = create_rows('train.csv')
    v_samples, v_anomalies = create_rows('validate.csv')
    test_samples, test_anomalies = create_rows(argv[1])
    tr_samples, v_samples, test_samples = normalize_data(tr_samples, v_samples, test_samples)

    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(tr_samples, tr_anomalies):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def run(config_file):

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    tr_samples, tr_anomalies = create_rows('train.csv')
    for xi, xo in zip(tr_samples, tr_anomalies):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

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
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)


if __name__ == '__main__':
    main()