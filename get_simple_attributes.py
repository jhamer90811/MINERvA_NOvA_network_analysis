"""
Author: Jesse Hamer

Version: 7/27/18

This is a script to read networks from a user-specified directory, computes simple static attributes, build
a pandas dataframe to store this data, and then output the data to a user-specified location. The script
also has methods to scrape output files corresponding to a given network for performance metrics.

Command Line Inputs:

:param input_path: str; The path to the head directory consisting of all genealogies of MINERvA or NOvA networks
:param output_path: str; The path to the output directory where resulting CSVs should be written
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

from Network import Network
# from MINERvA_NOvA_network_analysis.Network import Network

input_path = sys.argv[1]
output_path = sys.argv[2]
start_index = sys.argv[3].zfill(5)
end_index = sys.argv[4].zfill(5)
mode = sys.argv[5]


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
                a = float(l[l.index('=') + 1:])
                acc = acc + (a,)
        return acc
    elif mode == 'nova':
        output = path + '/' + 'output_' + network_name + '.txt'
        acc = ()
        with open(output) as op:
            for l in op.readlines()[len(op.readlines())-2:]:
                a = float(l[l.index('=') + 1:])
                acc = acc + (a,)
            return acc
    else:
        return None


if __name__ == '__main__':
    # Perform data generation here
    start_time = time.time()
    df = pd.DataFrame()
    genealogies = [int(g) for g in os.listdir(input_path) if len(g) == 5]
    genealogies.sort()
    genealogies = [str(g).zfill(5) for g in genealogies]
    print('Start index: {}; end index: {}'.format(start_index,end_index))
    for g in genealogies:
        print('Current genealogy: {}'.format(g))
        if int(end_index) < int(g) < int(start_index):
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
                        continue
                    if 'output_' + net_id + '.txt' not in os.listdir(g_path) and mode == 'nova':
                        print('Output file not found for {}. Skipping.'.format(net_id))
                        continue
                    accuracy = scrape_output(g_path, net_id, mode)
                    if len(accuracy) < 2:
                        print('Error in output file of {}. Skipping.'.format(net_id))
                        continue
                    net_path = g_path + '/' + N
                    net = Network(net_path, mode = mode)
                    attributes = []
                    columns = []
                    attributes.append(accuracy[0])
                    columns.append('initial_accuracy')
                    attributes.append(accuracy[1])
                    columns.append('final_accuracy')
                    ## EXTRACT SIMPLE ATTRIBUTES HERE (all use 'TEST' phase only):
                    # Genealogy
                    attributes.append(g)
                    columns.append('genealogy')
                    # NOTE: THE NET_DEPTH ATTRIBUTES DO NOT CONSIDER INCEPTION UNITS AS SINGLE ENTITIES
                    # AND IP LAYERS ARE INCLUDED
                    # net_depth_min
                    d = []
                    for ip in net.inputLayers:
                        d.append(net.get_net_depth(ip, weightsOnly=True, include_pooling=True,
                                                   phases=['ALL', 'TEST'], key='MIN')[0])
                    attributes.append(min(d))
                    columns.append('net_depth_min')
                    # net_depth_max
                    d = []
                    for ip in net.inputLayers:
                        d.append(net.get_net_depth(ip, weightsOnly=True, include_pooling=True,
                                                   phases=['ALL', 'TEST'], key='MAX')[0])
                    attributes.append(max(d))
                    columns.append('net_depth_max')
                    # net_depth_avg
                    d = []
                    for ip in net.inputLayers:
                        d.append(net.get_net_depth(ip, weightsOnly=True, include_pooling=True,
                                                   phases=['ALL', 'TEST'], key='AVG'))
                    attributes.append(np.mean(d))
                    columns.append('net_depth_avg')
                    # num_conv_layers; inception_unit = False; 1x1 considered
                    attributes.append(net.num_conv_layers(phases=['ALL', 'TEST']))
                    columns.append('num_conv_layers')
                    # num_inception_modules
                    attributes.append(net.num_inception_module(phases=['ALL', 'TEST']))
                    columns.append('num_inception_modules')
                    # num_pooling_layers; inception_unit = False
                    attributes.append(net.num_pooling_layers(phases=['ALL', 'TEST']))
                    columns.append('num_pooling_layers')
                    # num_IP_layers
                    attributes.append(net.num_IP_layers(phases=['All', 'TEST']))
                    columns.append('num_IP_layers')
                    # min_IP_neurons
                    attributes.append(net.num_IP_neurons(key='MIN', phases=['ALL', 'TEST'])[0])
                    columns.append('min_IP_neurons')
                    # max_IP_neurons
                    attributes.append(net.num_IP_neurons(key='MAX', phases=['ALL', 'TEST'])[0])
                    columns.append('max_IP_neurons')
                    # avg_IP_neurons
                    attributes.append(net.num_IP_neurons(key='AVG', phases=['ALL', 'TEST']))
                    columns.append('avg_IP_neurons')
                    # min_IP_weights
                    attributes.append(net.num_IP_weights(key='MIN', phases=['ALL', 'TEST'])[0])
                    columns.append('min_IP_weights')
                    # max_IP_weights
                    attributes.append(net.num_IP_weights(key='MAX', phases=['ALL', 'TEST'])[0])
                    columns.append('max_IP_weights')
                    # avg_IP_neurons
                    attributes.append(net.num_IP_weights(key='AVG', phases=['ALL', 'TEST']))
                    columns.append('avg_IP_weights')
                    # num_splits
                    attributes.append(net.num_splits(phases=['ALL', 'TEST']))
                    columns.append('num_splits')
                    # min_split_width
                    attributes.append(net.split_widths(key='MIN', phases=['ALL', 'TEST'])[0])
                    columns.append('min_split_width')
                    # max_split_width
                    attributes.append(net.split_widths(key='MAX', phases=['ALL', 'TEST'])[0])
                    columns.append('max_split_width')
                    # avg_split_width
                    attributes.append(net.split_widths(key='AVG', phases=['ALL', 'TEST']))
                    columns.append('avg_split_width')
                    # num_concats
                    attributes.append(net.num_concats(phases=['ALL', 'TEST']))
                    columns.append('num_concats')
                    # min_concat_width
                    attributes.append(net.concat_widths(key='MIN', phases=['ALL', 'TEST'])[0])
                    columns.append('min_concat_width')
                    # max_concat_width
                    attributes.append(net.concat_widths(key='MAX', phases=['ALL', 'TEST'])[0])
                    columns.append('max_concat_width')
                    # avg_concat_width
                    attributes.append(net.concat_widths(key='AVG', phases=['ALL', 'TEST']))
                    columns.append('avg_concat_width')
                    # min_conv_ker_area; inception included
                    attributes.append(net.conv_ker_area(key='MIN', phases=['ALL', 'TEST'])[0])
                    columns.append('min_conv_ker_area')
                    # max_conv_ker_area; inception included
                    attributes.append(net.conv_ker_area(key='MAX', phases=['ALL', 'TEST'])[0])
                    columns.append('max_conv_ker_area')
                    # avg_conv_ker_area; inception included
                    attributes.append(net.num_conv_features(key='AVG', phases=['ALL', 'TEST']))
                    columns.append('avg_conv_ker_area')
                    # min_num_conv_features; inception included
                    attributes.append(net.num_conv_features(key='MIN', phases=['ALL', 'TEST'])[0])
                    columns.append('num_conv_features')
                    # max_num_conv_features; inception included
                    attributes.append(net.num_conv_features(key='MAX', phases=['ALL', 'TEST'])[0])
                    columns.append('num_conv_features')
                    # avg_num_conv_features; inception included
                    attributes.append(net.num_conv_features(key='AVG', phases=['ALL', 'TEST']))
                    columns.append('avg_num_conv_features')
                    # prop_conv_into_pool; include inception
                    attributes.append(net.prop_conv_into_pool(phases=['ALL', 'TEST'])[0])
                    columns.append('prop_conv_into_pool')
                    # prop_pool_into_pool
                    attributes.append(net.prop_pool_into_pool(phases=['ALL', 'TEST'])[0])
                    columns.append('prop_pool_into_pool')
                    # prop_padded_conv; inception included
                    attributes.append(net.prop_padded_conv(phases=['ALL', 'TEST'])[0])
                    columns.append('prop_padded_conv')
                    # prop_same_padded_conv; inception included
                    attributes.append(net.prop_same_padded_conv(phases=['ALL', 'TEST'])[0])
                    columns.append('prop_same_padded_conv')
                    # prop_1x1_conv; inception included
                    attributes.append(net.prop_1x1_conv(phases=['ALL', 'TEST'])[0])
                    columns.append('prop_1x1_conv')
                    # prop_square_kernels; tol: 0.01; include pooling and inception
                    attributes.append(net.prop_square_kernels(phases=['ALL', 'TEST'])[0])
                    columns.append('prop_square_kernels')
                    # prop_horiz_kernels; tol = 16/9; pooling and inception included
                    attributes.append(net.prop_horiz_kernels(phases=['ALL', 'TEST'])[0])
                    columns.append('prop_horiz_kernels')
                    # prop_vert_kernels; tol = 16/9; pooling and inception included
                    attributes.append(net.prop_vert_kernels(phases=['ALL', 'TEST'])[0])
                    columns.append('prop_vert_kernels')
                    # num_relu; pooling and inception included
                    attributes.append(net.num_nonlinearities(nl='ReLU', phases=['ALL', 'TEST'])[0])
                    columns.append('num_relu')
                    # num_sigmoid; pooling and inception included
                    attributes.append(net.num_nonlinearities(nl='Sigmoid', phases=['ALL', 'TEST'])[0])
                    columns.append('num_sigmoid')
                    # num_tanh; pooling and inception included
                    attributes.append(net.num_nonlinearities(nl='TanH', phases=['ALL', 'TEST'])[0])
                    columns.append('num_tanh')
                    # num_max_pool; inception included
                    attributes.append(net.num_pool_type(pool_type='MAX', phases=['ALL', 'TEST'])[0])
                    columns.append('num_max_pool')
                    # num_avg_pool; inception included
                    attributes.append(net.num_pool_type(pool_type='AVG', phases=['ALL', 'TEST'])[0])
                    columns.append('num_avg_pool')
                    # num_stochastic_pool; inception included
                    attributes.append(net.num_pool_type(pool_type='STOCHASTIC', phases=['ALL', 'TEST'])[0])
                    columns.append('num_stochastic_pool')
                    # min_grid_reduction_area_consecutive; inception not unit; pooling included
                    attributes.append(net.grid_reduction_consecutive(key='MIN', phases=['ALL', 'TEST'], dim='a',
                                                                     include_pooling=True)[0])
                    columns.append('min_grid_reduction_area_consecutive')
                    # max_grid_reduction_area_consecutive; inception included; pooling included
                    attributes.append(net.grid_reduction_consecutive(key='MAX', phases=['ALL', 'TEST'], dim='a',
                                                                     include_pooling=True)[0])
                    columns.append('max_grid_reduction_area_consecutive')
                    # avg_grid_reduction_area_consecutive; inception included; pooling included
                    attributes.append(net.grid_reduction_consecutive(key='AVG', phases=['ALL', 'TEST'], dim='a',
                                                                     include_pooling=True))
                    columns.append('avg_grid_reduction_area_consecutive')
                    # min_grid_reduction_height_consecutive; inception not unit; pooling included
                    attributes.append(net.grid_reduction_consecutive(key='MIN', phases=['ALL', 'TEST'], dim='h',
                                                                     include_pooling=True)[0])
                    columns.append('min_grid_reduction_height_consecutive')
                    # max_grid_reduction_height_consecutive; inception included; pooling included
                    attributes.append(net.grid_reduction_consecutive(key='MAX', phases=['ALL', 'TEST'], dim='h',
                                                                     include_pooling=True)[0])
                    columns.append('max_grid_reduction_height_consecutive')
                    # avg_grid_reduction_height_consecutive; inception included; pooling included
                    attributes.append(net.grid_reduction_consecutive(key='AVG', phases=['ALL', 'TEST'], dim='h',
                                                                     include_pooling=True))
                    columns.append('avg_grid_reduction_height_consecutive')
                    # min_grid_reduction_width_consecutive; inception not unit; pooling included
                    attributes.append(net.grid_reduction_consecutive(key='MIN', phases=['ALL', 'TEST'], dim='w',
                                                                     include_pooling=True)[0])
                    columns.append('min_grid_reduction_width_consecutive')
                    # max_grid_reduction_width_consecutive; inception included; pooling included
                    attributes.append(net.grid_reduction_consecutive(key='MAX', phases=['ALL', 'TEST'], dim='w',
                                                                     include_pooling=True)[0])
                    columns.append('max_grid_reduction_width_consecutive')
                    # avg_grid_reduction_width_consecutive; inception included; pooling included
                    attributes.append(net.grid_reduction_consecutive(key='AVG', phases=['ALL', 'TEST'], dim='w',
                                                                     include_pooling=True))
                    columns.append('avg_grid_reduction_width_consecutive')
                    # min_grid_reduction_area_total; inception not unit; pooling not included
                    attributes.append(net.grid_reduction_total(key='MIN', phases=['ALL', 'TEST'], dim='a')[0])
                    columns.append('min_grid_reduction_area_total')
                    # max_grid_reduction_area_total; inception included; pooling not included
                    attributes.append(net.grid_reduction_total(key='MAX', phases=['ALL', 'TEST'], dim='a')[0])
                    columns.append('max_grid_reduction_area_total')
                    # avg_grid_reduction_area_total; inception included; pooling not included
                    attributes.append(net.grid_reduction_total(key='AVG', phases=['ALL', 'TEST'], dim='a'))
                    columns.append('avg_grid_reduction_area_total')
                    # min_grid_reduction_height_total; inception not unit; pooling not included
                    attributes.append(net.grid_reduction_total(key='MIN', phases=['ALL', 'TEST'], dim='h')[0])
                    columns.append('min_grid_reduction_height_total')
                    # max_grid_reduction_height_total; inception included; pooling not included
                    attributes.append(net.grid_reduction_total(key='MAX', phases=['ALL', 'TEST'], dim='h')[0])
                    columns.append('max_grid_reduction_height_total')
                    # avg_grid_reduction_height_total; inception included; pooling not included
                    attributes.append(net.grid_reduction_total(key='AVG', phases=['ALL', 'TEST'], dim='h'))
                    columns.append('avg_grid_reduction_height_total')
                    # min_grid_reduction_width_total; inception not unit; pooling not included
                    attributes.append(net.grid_reduction_total(key='MIN', phases=['ALL', 'TEST'], dim='w')[0])
                    columns.append('min_grid_reduction_width_total')
                    # max_grid_reduction_width_total; inception included; pooling not included
                    attributes.append(net.grid_reduction_total(key='MAX', phases=['ALL', 'TEST'], dim='w')[0])
                    columns.append('max_grid_reduction_width_total')
                    # avg_grid_reduction_width_total; inception included; pooling not included
                    attributes.append(net.grid_reduction_total(key='AVG', phases=['ALL', 'TEST'], dim='w'))
                    columns.append('avg_grid_reduction_width_total')
                    # prop_nonoverlapping; not including 1x1; including pooling and inception
                    attributes.append(net.prop_nonoverlapping(phases=['ALL', 'TEST']))
                    columns.append('prop_nonoverlapping')
                    # min_stride_h; pooling and inception included; 1x1 not included
                    attributes.append(net.stride_dims(key='MIN', phases=['ALL', 'TEST'], dim='h')[0])
                    columns.append('min_stride_h')
                    # max_stride_h; pooling and inception included; 1x1 not included
                    attributes.append(net.stride_dims(key='MAX', phases=['ALL', 'TEST'], dim='h')[0])
                    columns.append('max_stride_h')
                    # avg_stride_h; pooling and inception included; 1x1 not included
                    attributes.append(net.stride_dims(key='AVG', phases=['ALL', 'TEST'], dim='h'))
                    columns.append('avg_stride_h')
                    # min_stride_w; pooling and inception included; 1x1 not included
                    attributes.append(net.stride_dims(key='MIN', phases=['ALL', 'TEST'], dim='w')[0])
                    columns.append('min_stride_w')
                    # max_stride_w; pooling and inception included; 1x1 not included
                    attributes.append(net.stride_dims(key='MAX', phases=['ALL', 'TEST'], dim='w')[0])
                    columns.append('max_stride_w')
                    # avg_stride_w; pooling and inception included; 1x1 not included
                    attributes.append(net.stride_dims(key='AVG', phases=['ALL', 'TEST'], dim='w'))
                    columns.append('avg_stride_w')
                    # min_ratio_features_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_depth(key='MIN', phases=['ALL', 'TEST'])[0])
                    columns.append('min_ratio_features_to_depth')
                    # max_ratio_features_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_depth(key='MAX', phases=['ALL', 'TEST'])[0])
                    columns.append('max_ratio_features_to_depth')
                    # avg_ratio_features_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_depth(key='AVG', phases=['ALL', 'TEST']))
                    columns.append('avg_ratio_features_to_depth')
                    # min_ratio_features_to_kerArea; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_kerDim(key='MIN', phases=['ALL', 'TEST'], dim='a')[0])
                    columns.append('min_ratio_features_to_kerArea')
                    # max_ratio_features_to_kerArea; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_kerDim(key='MAX', phases=['ALL', 'TEST'], dim='a')[0])
                    columns.append('max_ratio_features_to_kerArea')
                    # avg_ratio_features_to_kerArea; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_kerDim(key='AVG', phases=['ALL', 'TEST'], dim='a'))
                    columns.append('avg_ratio_features_to_kerArea')
                    # min_ratio_features_to_kerWidth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_kerDim(key='MIN', phases=['ALL', 'TEST'], dim='w')[0])
                    columns.append('min_ratio_features_to_kerWidth')
                    # max_ratio_features_to_kerWidth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_kerDim(key='MAX', phases=['ALL', 'TEST'], dim='w')[0])
                    columns.append('max_ratio_features_to_kerWidth')
                    # avg_ratio_features_to_kerWidth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_kerDim(key='AVG', phases=['ALL', 'TEST'], dim='w'))
                    columns.append('avg_ratio_features_to_kerWidth')
                    # min_ratio_features_to_kerHeight; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_kerDim(key='MIN', phases=['ALL', 'TEST'], dim='h')[0])
                    columns.append('min_ratio_features_to_kerHeight')
                    # max_ratio_features_to_kerHeight; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_kerDim(key='MAX', phases=['ALL', 'TEST'], dim='h')[0])
                    columns.append('max_ratio_features_to_kerHeight')
                    # avg_ratio_features_to_kerHeight; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_features_to_kerDim(key='AVG', phases=['ALL', 'TEST'], dim='h'))
                    columns.append('avg_ratio_features_to_kerHeight')
                    # min_ratio_kerArea_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_kerDim_to_depth(key='MIN', phases=['ALL', 'TEST'], dim='a')[0])
                    columns.append('min_ratio_kerArea_to_depth')
                    # max_ratio_kerArea_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_kerDim_to_depth(key='MAX', phases=['ALL', 'TEST'], dim='a')[0])
                    columns.append('max_ratio_kerArea_to_depth')
                    # avg_ratio_kerArea_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_kerDim_to_depth(key='AVG', phases=['ALL', 'TEST'], dim='a'))
                    columns.append('avg_ratio_kerArea_to_depth')
                    # min_ratio_kerWidth_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_kerDim_to_depth(key='MIN', phases=['ALL', 'TEST'], dim='w')[0])
                    columns.append('min_ratio_kerWidth_to_depth')
                    # max_ratio_kerWidth_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_kerDim_to_depth(key='MAX', phases=['ALL', 'TEST'], dim='w')[0])
                    columns.append('max_ratio_kerWidth_to_depth')
                    # avg_ratio_kerWidth_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_kerDim_to_depth(key='AVG', phases=['ALL', 'TEST'], dim='w'))
                    columns.append('avg_ratio_kerWidth_to_depth')
                    # min_ratio_kerHeight_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_kerDim_to_depth(key='MIN', phases=['ALL', 'TEST'], dim='h')[0])
                    columns.append('min_ratio_kerHeight_to_depth')
                    # max_ratio_kerHeight_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_kerDim_to_depth(key='MAX', phases=['ALL', 'TEST'], dim='h')[0])
                    columns.append('max_ratio_kerHeight_to_depth')
                    # avg_ratio_kerHeight_to_depth; pooling not factored into depth; inception modules are units; 1x1, IP included
                    attributes.append(net.ratio_kerDim_to_depth(key='AVG', phases=['ALL', 'TEST'], dim='h'))
                    columns.append('avg_ratio_kerHeight_to_depth')
                    #####

                    new_row = pd.DataFrame.from_dict({net_id: attributes}, orient='index', columns=columns)
                    df = df.append(new_row)
    output_name = output_path + '/' + mode + '-simple-' + start_index + '-' + end_index + '.csv'
    df.to_csv(output_name)
    print('Total time elapsed: {} s'.format(time.time()-start_time))
