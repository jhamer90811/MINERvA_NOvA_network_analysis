"""
Author: Jesse Hamer

Version: 7/25/18

This is a test script to ensure that all relevant libraries load properly.
"""
import sys

print('Running {}...'.format(sys.argv[0]))

print('System path is: {}'.format(sys.path))

try:
    from Network import Network
    print('Loaded Network!')
except:
    print('Failed to load Network')
try:
    import gudhi as gd
    print('Loaded GUDHI!')
except:
    print('Failed to load GUDHI.')
try:
    import matplotlib.pyplot as plt
    print('Loaded plt!')
except:
    print('Failed to load plt.')
try:
    import simplicial as sc
    print('Loaded Simplicial!')
except:
    print('Failed to load Simplicial.')
try:
    import numpy as np
    print('Loaded numpy!')
except:
    print('Failed to load numpy.')
try:
    import itertools
    print('Loaded itertools!')
except:
    print('Failed to load itertools.')
try:
    import h5py
    print('Loaded h5py!')
except:
    print('Failed to load h5py.')
try:
    from google.protobuf import text_format
    print('Loaded google.protobuf!')
except:
    print('Failed to load google.protobuf.')
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    print('Loaded mpl_toolkits!')
except:
    print('Failed to load mpl_toolkits.')
try:
    import caffe_pb2
    print('Loaded caffe_pb2!')
except:
    print('Failed to load caffe_pb2')

print('The arguments you passed were: ')
for arg in sys.argv[1:]:
    print(arg)

