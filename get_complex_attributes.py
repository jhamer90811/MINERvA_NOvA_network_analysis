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

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# from Network import Network
from MINERvA_NOvA_network_analysis.Network import Network

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
        acc = ()
        with open(output) as op:
            for l in op.readlines():
                try:
                    a = float(l[l.index('=') + 1:])
                except ValueError:
                    a = np.NaN
                acc = acc + (a,)
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
                acc = acc + (a,)
            return acc
    else:
        return None


if __name__ == '__main__':
    # Perform data generation here

    start_time = time.time()
    df_complex = pd.DataFrame()
    genealogies = [int(g) for g in os.listdir(input_path) if len(g) == 5]
    genealogies.sort()
    genealogies = [str(g).zfill(5) for g in genealogies]
    print('Start index: {}; end index: {}'.format(start_index, end_index))
    missing_op = 0
    num_training_imgs = 200
    for g in genealogies:
        print('Current genealogy: {}'.format(g))
        if int(end_index) < int(g) or int(g) < int(start_index):
            continue
        else:
            g_path = input_path + '/' + g
            for N in os.listdir(g_path):
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
                    attributes = []
                    columns = []
                    attributes.append(accuracy[0])
                    columns.append('initial_accuracy')
                    attributes.append(accuracy[1])
                    columns.append('final_accuracy')

                    df = pd.DataFrame()

                    for i in range(num_training_imgs):
                        net.feed_image(hdf5=img_file, mode=mode, rimg=True)
                        for l in net.inputLayers:
                            net.set_img_alpha_cplx2d(l)
                            net.set_img_points2d(l)
                            net.set_min_connected_alpha22d(l)
                        img_features = []
                        img_columns = []

                        # GET IMAGE FEATURES

                        # nonzero_activations
                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.nonzero_activations(l))
                        img_features.append(np.mean(feature))
                        img_columns.append('nonzero_activations')

                        # horiz_spread
                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.horiz_spread(l))
                        img_features.append(np.mean(feature))
                        img_columns.append('horiz_spread')

                        # vert_spread
                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.vert_spread(l))
                        img_features.append(np.mean(feature))
                        img_columns.append('vert_spread')

                        # horiz_sd
                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.horiz_sd(l))
                        img_features.append(np.mean(feature))
                        img_columns.append('horiz_sd')

                        # vert_sd
                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.vert_sd(l))
                        img_features.append(np.mean(feature))
                        img_columns.append('vert_sd')

                        # min_alpha2
                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.layers[l].imgFeatures['min_connected_alpha22d'])
                        img_features.append(np.mean(feature))
                        img_columns.append('min_alpha2')

                        # min_alpha2_betti_0 and min_alpha2_betti_1

                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.get_betti_numbers(l))
                        img_features.append(np.mean([f[0] for f in feature]))
                        img_features.append(np.mean([f[1] for f in feature]))
                        img_columns.append('min_alpha2_betti_0')
                        img_columns.append('min_alpha2_betti_1')

                        # num_persistent_components

                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.get_num_persistent_components(l, pers_scaler=0.25))
                        img_features.append(np.mean(feature))
                        img_columns.append('num_persistent_components')

                        # num_persistent_holes

                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.get_num_persistent_holes(l, pers_scaler=0.25))
                        img_features.append(np.mean(feature))
                        img_columns.append('num_persistent_holes')

                        # num_delaunay_edges

                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.get_delaunay_edges(l))
                        img_features.append(np.mean(feature))
                        img_columns.append('num_delaunay_edges')

                        # num_min_alpha_edges

                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.get_alpha_edges(l))
                        img_features.append(np.mean(feature))
                        img_columns.append('num_min_alpha_edges')

                        # min_alpha_ec

                        feature = []
                        for l in net.inputLayers:
                            feature.append(net.get_EC(l))
                        img_features.append(np.mean(feature))
                        img_columns.append('min_alpha_ec')

                        # inter_layer_bottleneck_avg

                        img_features.append(net.inter_layer_bottleneck_avg())
                        img_columns.append('inter_layer_bottleneck_avg')

                        # NOW GET IMAGE SCORES

                        scores1 = []
                        scores2 = []
                        scores3 = []
                        scores4 = []
                        scores5 = []

                        for l in net.inputLayers:
                            max_paths = net.get_max_paths(l, convOnly=False, phases=['ALL', 'TEST'], include_pooling=True)
                            unique_paths = []
                            for p in max_paths:
                                if p[:-1] not in unique_paths:
                                    unique_paths.append(p[:-1])
                            for p in unique_paths:
                                scores1.append(net.first_last_bottleneck(p))
                                scores2.append(net.avg_consecutive_bottleneck(p))
                                scores3.append(net.total_bottleneck_variation(p))
                            scores4.append(net.first_layer_bottleneck(l))
                            scores5.append(net.first_layer_min_alpha2(l))

                        img_features = img_features + [np.mean(scores1),
                                                       np.mean(scores2),
                                                       np.mean(scores3),
                                                       np.mean(scores4),
                                                       np.mean(scores5)]
                        img_columns = img_columns + ['first_last_bottleneck',
                                                     'avg_consecutive_bottleneck',
                                                     'total_bottleneck_variation',
                                                     'first_layer_bottleneck',
                                                     'first_layer_min_alpha2']
                        new_img = pd.DataFrame.from_dict({net.layers['data0_0'].imgFeatures['id']: img_features},
                                                         orient='index',
                                                         columns=img_columns)
                        df.append(new_img)

                    # Train regression models on each score and add row to overall dataframe.
                    feature_names = img_columns[:-5]
                    X = df.iloc[:, :-5]
                    flb = df.iloc[:, -5:-4]
                    acb = df.iloc[:, -4:-3]
                    tbv = df.iloc[:, -3:-2]
                    flb2 = df.iloc[:, -2:-1]
                    flma = df.iloc[:, -1:]
                    LR = LinearRegression(True, True, True, -1)
                    LR.fit(X, flb)
                    weights_flb = LR.coef_
                    names_flb = [name + '_flb' for name in feature_names]
                    flb_r2 = LR.score(X, flb)
                    LR.fit(X, acb)
                    weights_acb = LR.coef_
                    names_acb = [name + '_acb' for name in feature_names]
                    acb_r2 = LR.score(X, acb)
                    LR.fit(X, tbv)
                    weights_tbv = LR.coef_
                    names_tbv = [name + '_tbv' for name in feature_names]
                    tbv_r2 = LR.score(X, tbv)
                    LR.fit(X, flb2)
                    weights_flb2 = LR.coef_
                    names_flb2 = [name + '_flb2' for name in feature_names]
                    flb2_r2 = LR.score(X, flb2)
                    LR.fit(X, flma)
                    weights_flma = LR.coef_
                    names_flma = [name + '_flma' for name in feature_names]
                    flma_r2 = LR.fit(X, flma)

                    attributes = weights_acb + weights_flb + weights_flb2 + weights_flma + weights_tbv
                    attributes = attributes + [acb_r2, flb_r2, flb2_r2, flma_r2, tbv_r2]
                    names = names_acb + names_flb + names_flb2 + names_flma + names_tbv
                    names = names + ['acb_r2', 'flb_r2', 'flb2_r2', 'flma_r2', 'tbv_r2']

                    new_row = pd.DataFrame.from_dict({net_id: attributes}, orient='index', columns=names)
                    df_complex.append(new_row)

    output_name = output_path + '/' + mode + '-complex-' + start_index + '-' + end_index + '.csv'
    df_complex.to_csv(output_name)
    print('Number of missing/corrupted output files: {}'.format(missing_op))
    print('Total time elapsed: {} s'.format(time.time() - start_time))


