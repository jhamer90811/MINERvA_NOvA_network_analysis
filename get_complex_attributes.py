"""
Author: Jesse Hamer

Version: 8/2/18

This is a script to read networks from a user-specified directory, compute complex static attributes, build
a pandas dataframe to store this data, and then output the data to a user-specified location. The script
also has methods to scrape output files corresponding to a given network for performance metrics.

Command Line Inputs:

:param input_path: str; The path to the head directory consisting of all genealogies of MINERvA or NOvA networks
:param output_path: str; The path to the output directory where resulting CSVs should be written
:param img_file: str; The path to the hdf5 file with images
:param start_index: int; the index of the genealogy on which data collection should begin
:param end_index: int; the index of the genealogy on which data collection should end
:param mode: str; one of 'minerva' or 'nova'

NOTE: ALL COMMAND LINE INPUTS SHOULD BE SURROUNDED BY QUOTES
"""

import os
import sys
import time
import itertools
# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from Network import Network
# from MINERvA_NOvA_network_analysis.Network import Network

input_path = sys.argv[1]
output_path = sys.argv[2]
img_file = sys.argv[3]
start_index = sys.argv[4].zfill(5)
end_index = sys.argv[5].zfill(5)
mode = sys.argv[6]


def scrape_output(path, network_name, mode):
    """
    Will scrape the output file corresponding to the network with name "input" for initial and final
    accuracies.

    :param path: str; the path to the subdirectory containing the network output file
    :param network_name: str; the identifier of the network whose output file we desire; the name should
        have the form <genealogy>_<network>_<subID>, where each of the angle-bracketed expressions is
        a 5-digit string
    :param mode: str; one of 'minerva' or 'nova'
    :return: (float initial accuracy, float final accuracy)
    """
    if mode == 'minerva':
        output = path + '/' + 'output_' + network_name + '.txt_test.txt'
        acc = []
        with open(output) as op:
            for l in op.readlines():
                try:
                    a = float(l[l.index('=') + 1:])
                except ValueError:
                    a = np.NaN
                acc = acc + [a]
        return acc
    elif mode == 'nova':
        output = path + '/' + 'output_' + network_name + '.txt'
        acc = ()
        with open(output) as op:
            for l in op.readlines()[len(op.readlines())-2:]:
                try:
                    a = float(l[l.index('=') + 1:])
                except ValueError:
                    a = np.NaN
                acc = acc + [a]
            return acc
    else:
        return None


def get_max_paths(ip):
    Net = ip[0]
    ipLayer = ip[1]
    max_paths = Net.get_max_paths(ipLayer, convOnly=True, phases=['ALL', 'TEST'],
                                  include_pooling=True)
    unique_paths = []
    for p in max_paths:
        if p[:-1] not in unique_paths:
            unique_paths.append(p[:-1])

    return unique_paths


if __name__ == '__main__':
    # Perform data generation here

    start_time = time.time()
    df_complex = pd.DataFrame()
    genealogies = [int(g) for g in os.listdir(input_path) if len(g) == 5]
    genealogies.sort()
    genealogies = [str(g).zfill(5) for g in genealogies]
    print('Start index: {}; end index: {}'.format(start_index, end_index))
    missing_op = 0
    num_training_imgs = 50
    g_times = []
    N_times = []
    img_times = []
    pool = Pool()

    img_columns = ['nonzero_activations', 'horiz_spread', 'vert_spread', 'horiz_sd', 'vert_sd', 'min_alpha2',
                   'min_alpha2_betti_0', 'min_alpha2_betti_1', 'num_persistent_components', 'num_persistent_holes',
                   'num_delaunay_edges', 'num_min_alpha_edges', 'inter_layer_bottleneck_avg','first_last_bottleneck',
                   'avg_consecutive_bottleneck', 'total_bottleneck_variation', 'first_layer_bottleneck',
                   'first_layer_min_alpha2']
    feature_names = img_columns[:-5]
    names_flb = [name + '_flb' for name in feature_names]
    names_acb = [name + '_acb' for name in feature_names]
    names_tbv = [name + '_tbv' for name in feature_names]
    names_flb2 = [name + '_flb2' for name in feature_names]
    names_flma = [name + '_flma' for name in feature_names]

    columns = ['initial_accuracy', 'final_accuracy']
    columns = columns + names_acb + names_flb + names_flb2 + names_flma + names_tbv
    columns = columns + ['acb_r2', 'flb_r2', 'flb2_r2', 'flma_r2', 'tbv_r2']

    for g in genealogies:
        if int(end_index) < int(g) or int(g) < int(start_index):
            continue
        else:
            print('Current genealogy: {}'.format(g))
            g_path = input_path + '/' + g
            g_start = time.time()
            for N in os.listdir(g_path):
                N_start = time.time()
                if 'train_test_' not in N or 'caffemodel' in N or 'solverstate' in N:
                    continue
                else:
                    net_id = N[11:-9]
                    print('Working on {}...'.format(net_id))
                    if 'output_' + net_id + '.txt_test.txt' not in os.listdir(g_path) and mode == 'minerva':
                        print('Output file not found for {}. Skipping.'.format(net_id))
                        missing_op += 1
                        continue
                    if 'output_' + net_id + '.txt' not in os.listdir(g_path) and mode == 'nova':
                        print('Output file not found for {}. Skipping.'.format(net_id))
                        missing_op += 1
                        continue
                    accuracy = scrape_output(g_path, net_id, mode)
                    if len(accuracy) < 2:
                        print('Error in output file of {}. Skipping.'.format(net_id))
                        missing_op += 1
                        continue
                    net_path = g_path + '/' + N
                    net = Network(net_path, mode=mode)
                    attributes = accuracy

                    ipLayers = net.inputLayers
                    inputs = [(net, l) for l in ipLayers]

                    df = pd.DataFrame()

                    for i in range(num_training_imgs):
                        img_start = time.time()
                        net.feed_image(hdf5=img_file, mode=mode, rimg=True)
                        print('Extracting Image features on image {}...'.format(net.layers['data0_0'].imgFeatures['id']))

                        #for l in ipLayers:
                        #    net.set_img_alpha_cplx2d(l)
                        #    net.set_img_points2d(l)
                        #    net.set_min_connected_alpha22d(l)
                        points = pool.map(net.get_img_points2d, ipLayers)
                        for l, p in zip(ipLayers, points):
                            net.layers[l].imgFeatures['points_2d'] = p

                        img_features = []

                        paths = pool.map(get_max_paths, inputs)

                        nonzero_activations = pool.map(net.nonzero_activations, ipLayers)
                        horiz_spread = pool.map(net.horiz_spread, ipLayers)
                        vert_spread = pool.map(net.vert_spread, ipLayers)
                        horiz_sd = pool.map(net.horiz_sd, ipLayers)
                        vert_sd = pool.map(net.vert_sd, ipLayers)

                        for l in ipLayers:
                            net.set_img_alpha_cplx2d(l)
                            net.set_min_connected_alpha22d(l)

                        min_alpha2 = [net.layers[l].imgFeatures['min_connected_alpha22d'] for l in ipLayers]
                        betti_nums = [net.get_betti_numbers(l) for l in ipLayers]
                        num_persistent_components = [net.get_num_persistent_components(l) for l in ipLayers]
                        num_persistent_holes = [net.get_num_persistent_holes(l) for l in ipLayers]
                        num_delaunay_edges = [net.get_delaunay_edges(l) for l in ipLayers]
                        num_min_alpha_edges = [net.get_alpha_edges(l) for l in ipLayers]
                        inter_layer_bottleneck_avg = net.inter_layer_bottleneck_avg()

                        scores_bottleneck = [net.bottleneck_scores(p) for p in itertools.chain.from_iterable(paths)]
                        scores_first_layer = [net.first_layer_scores(l) for l in ipLayers]

                        scores1 = [s[0] for s in scores_bottleneck]
                        scores2 = [s[1] for s in scores_bottleneck]
                        scores3 = [s[2] for s in scores_bottleneck]
                        scores4 = [s[0] for s in scores_first_layer]
                        scores5 = [s[1] for s in scores_first_layer]

                        img_features.append(np.mean(nonzero_activations))
                        img_features.append(np.mean(horiz_spread))
                        img_features.append(np.mean(vert_spread))
                        img_features.append(np.mean(horiz_sd))
                        img_features.append(np.mean(vert_sd))
                        img_features.append(np.mean(min_alpha2))
                        img_features.append(np.mean([p[0] for p in betti_nums]))
                        img_features.append(np.mean([p[1] if len(p) > 1 else 0 for p in betti_nums]))
                        img_features.append(np.mean(num_persistent_components))
                        img_features.append(np.mean(num_persistent_holes))
                        img_features.append(np.mean(num_delaunay_edges))
                        img_features.append(np.mean(num_min_alpha_edges))
                        img_features.append(np.mean(inter_layer_bottleneck_avg))
                        img_features = img_features + [np.mean(scores1),
                                                       np.mean(scores2),
                                                       np.mean(scores3),
                                                       np.mean(scores4),
                                                       np.mean(scores5)]

                        if sum([np.isnan(f) for f in img_features]) > 0:
                            print('Found NaN value. Skipping...')
                            net.reset_img_features()
                            continue

                        new_img = pd.DataFrame.from_dict({net.layers['data0_0'].imgFeatures['id']: img_features},
                                                         orient='index',
                                                         columns=img_columns)
                        df = df.append(new_img)
                        img_times.append(time.time()-img_start)
                        net.reset_img_features()

                    # Train regression models on each score and add row to overall dataframe. Can also try other models.

                    X = df.iloc[:, :-5]
                    flb = df.iloc[:, -5:-4]
                    acb = df.iloc[:, -4:-3]
                    tbv = df.iloc[:, -3:-2]
                    flb2 = df.iloc[:, -2:-1]
                    flma = df.iloc[:, -1:]
                    LR = LinearRegression(True, True, True, -1)
                    LR.fit(X, flb)
                    weights_flb = list(LR.coef_[0, :])

                    flb_r2 = LR.score(X, flb)
                    LR.fit(X, acb)
                    weights_acb = list(LR.coef_[0, :])

                    acb_r2 = LR.score(X, acb)
                    LR.fit(X, tbv)
                    weights_tbv = list(LR.coef_[0, :])

                    tbv_r2 = LR.score(X, tbv)
                    LR.fit(X, flb2)
                    weights_flb2 = list(LR.coef_[0, :])

                    flb2_r2 = LR.score(X, flb2)
                    LR.fit(X, flma)
                    weights_flma = list(LR.coef_[0, :])

                    flma_r2 = LR.score(X, flma)

                    attributes = attributes + weights_acb + weights_flb + weights_flb2 + weights_flma + weights_tbv
                    attributes = attributes + [acb_r2, flb_r2, flb2_r2, flma_r2, tbv_r2]

                    new_row = pd.DataFrame.from_dict({net_id: attributes}, orient='index', columns=columns)
                    df_complex = df_complex.append(new_row)
                    N_times.append(time.time()-N_start)
            print("Genealogy time: {}s.".format(time.time()-g_start))
            g_times.append(time.time()-g_start)

    output_name = output_path + '/' + mode + '-complex-' + start_index + '-' + end_index + '.csv'
    df_complex.to_csv(output_name)
    print('Number of missing/corrupted output files: {}'.format(missing_op))
    print('Average image time: {}s.'.format(np.mean(img_times)))
    print('Average network time: {}s.'.format(np.mean(N_times)))
    print('Average genealogy time: {}s.'.format(np.mean(g_times)))
    print('Total time elapsed: {}s.'.format(time.time() - start_time))
