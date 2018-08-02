"""
Author: Jesse Hamer

This is a collection of classes meant for handling caffe layer objects for
the purposes of extracting network architecture information.

The classes are designed specifically to parse Caffe networks used to analyze
data from the MINERvA and NOvA experiments. They are not intended as general
Caffe network containers.
"""

import itertools

import gudhi as gd
import h5py
import numpy as np
import simplicial as simp
from google.protobuf import text_format
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Include one of the following; depends on current working directory
from MINERvA_NOvA_network_analysis import caffe_pb2


# import caffe_pb2


# %%

# NETWORK CLASS INITIALIZER AND BASE METHODS

class Network:
    """
    This class is the main container for Caffe network objects. It will
    consist of several layer objects and methods for those objects.
    
    Currently, many methods are specific to networks designed for MINERvA 
    and NOvA data, and will need to be modified so as to accommodate more generic
    network structures. Specifically, an all-purpose method to build input
    layers from a given Caffe network specification is needed.
    
    Parameters:
        caffeNet:
            str; path to Caffe layer .prototxt file
        
        mode:
            one of 'minerva' or 'nova'; specifies how certain aspects of
            network initialization are handled, e.g. input-dimensions
    """

    # caffeNet is a path to a file containing a Caffe network protobuf
    def __init__(self, caffeNet, mode='minerva'):
        # List used to add nonlinearity information to neuron layers
        nonlinearity_list = ['ReLU', 'PReLU', 'ELU', 'Sigmoid', 'TanH',
                             'Power', 'Exp', 'Log', 'BNLL', 'Threshold',
                             'Bias', 'Scale']
        # List used to handle MINERvA-specific Caffe layers whose "top" is not
        # the same as the layer's name.
        minerva_dummy_list = ['target_label', 'target_label_planecode',
                              'source_data', 'label_planecode', 'data',
                              'Slice NodeX1', 'Slice NodeX2', 'bottleneck',
                              'discard_features', 'keep_features', 'dc_labels']
        nova_dummy_list = ['label']

        tmp_net = caffe_pb2.NetParameter()
        with open(caffeNet) as f:
            text_format.Merge(f.read(), tmp_net)
        self.name = tmp_net.name
        self._img_counter = 0
        self.layers = {}
        self.dataset = mode
        # Maintain special list of input layers for quick lookup
        self.inputLayers = []
        trash_list = []
        for layer in tmp_net.layer:
            skip = False
            # Need special handler for inception concatenators in NOVA networks
            if layer.type == 'Concat' and 'inception' in layer.name:
                layer_name = layer.top[0]
            elif layer.name == 'finalip':
                layer_name = layer.top[0]
            else:
                layer_name = layer.name
            if layer.type == 'Pooling':
                if layer.pooling_param.kernel_size == 1 or (
                        layer.pooling_param.kernel_h == 1 and layer.pooling_param.kernel_w == 1):
                    trash_list.append((layer_name, layer.bottom))
                    continue
            self.layers[layer_name] = Layer(caffeLayer=layer)
            for b in layer.bottom:
                if b in [t[0] for t in trash_list]:
                    if self.layers[layer_name].bottom == layer.bottom:
                        self.layers[layer_name].bottom = []
                    if layer.bottom == layer.top:
                        trash = self.layers.pop(layer_name)
                        if trash:
                            pass
                        skip = True
                        break
                    exists = False
                    new_bottom = [nb for p, nb in trash_list if p == b][0]
                    while not exists:
                        if new_bottom[0] in [t[0] for t in trash_list]:
                            new_bottom = [nb for p, nb in trash_list if p == new_bottom[0]][0]
                        else:
                            exists = True
                    self.layers[layer_name].bottom.append(new_bottom[0])
                    # trash_list.remove((b, new_bottom))
            if skip:
                continue
            if layer.name == 'finalip':
                self.layers[layer_name].name = layer_name
            # Build dummy layers, if necessary:
            if layer.top:
                for top in layer.top:
                    if (top in minerva_dummy_list) or (top in nova_dummy_list):
                        if top not in self.layers.keys():
                            temp_phase = ''
                            if layer.include:
                                if layer.include[0].phase == 0:
                                    temp_phase = 'TRAIN'
                                elif layer.include[0].phase == 1:
                                    temp_phase = 'TEST'
                            else:
                                temp_phase = 'ALL'
                            dummyInfo = {'name': top, 'type': 'Placeholder', 'top': [],
                                         'bottom': [layer_name], 'phase': temp_phase}
                            self.layers[top] = Layer(inputLayer=dummyInfo,
                                                     dummy=True)
                            self.layers[layer_name].top.append(top)
                        else:
                            if layer.include:
                                if layer.include[0].phase != self.layers[top].phase:
                                    self.layers[top].phase = 'ALL'
                            else:
                                self.layers[top].phase = 'ALL'
                            self.layers[layer_name].top.append(top)
                            self.layers[top].bottom.append(layer_name)
            # Build input layers, if necessary.
            if layer.top:
                for top in layer.top:
                    if 'data0' in top:
                        inputInfo = self._getInputInfo(layer, top, mode)
                        self.layers[inputInfo['name']] = Layer(inputLayer=inputInfo)
                        self.inputLayers.append(top)
            # Update the tops of other layers.
            for bottom in self.layers[layer_name].bottom:
                self.layers[bottom].top.append(layer_name)
            # Handle concatenation, if necessary:
            if layer.type == 'Concat':
                self._concat_handler(layer_name)
            # Update num_outputs of pooling or LRN layers.
            if layer.type in ['Pooling', 'LRN']:
                self.layers[layer_name].layerParams['num_output'] = \
                    self.layers[self.layers[layer_name].bottom[0]].layerParams['num_output']
            # Handle Flatten and IP layers.
            if layer.type == 'Flatten':
                self.layers[layer_name].layerParams['input_grid'] = \
                    self.layers[self.layers[layer_name].bottom[0]].layerParams['input_grid']
                self.layers[layer_name].layerParams['num_output'] = \
                    np.prod(self.layers[layer_name].layerParams['input_grid'].shape)

            if layer.type == 'InnerProduct':
                self.layers[layer_name].layerParams['num_input'] = \
                    self.layers[self.layers[layer_name].bottom[0]].layerParams['num_output']

            # Add outputs to special layers.

            if layer_name in ['alias_to_bottleneck', 'slice_features',
                              'bottleneck_alias', 'grl']:
                self.layers[layer_name].layerParams['num_output'] = \
                    self.layers[self.layers[layer_name].bottom[0]].layerParams['num_output']
                for top in layer.top:
                    if top in ['bottleneck', 'keep_features']:
                        self.layers[top].layerParams['num_output'] = \
                            self.layers[layer_name].layerParams['num_output']

            # Now check to see if we have layers whose grid attributes need
            # updating.
            if layer.type in ['Pooling', 'Convolution', 'LRN']:
                # add input grid, and output grid
                self.layers[layer_name].layerParams['input_grid'], \
                self.layers[layer_name].layerParams['output_grid'] = \
                    self.get_grids(self.layers[self.layers[layer_name].bottom[0]],
                                   self.layers[layer_name])
            if layer.type in nonlinearity_list:
                self.layers[self.layers[layer_name].bottom[0]].layerParams['nonlinearity'] = layer.type

    @staticmethod
    def _getInputInfo(caffeLayer, name, mode):
        """
        Handler for MINERvA and NOvA input layers.

        Parameters:
            caffeLayer:
                A Caffe layer protobuf message; see caffe_pb2 for details
            name:
                str; name of the input layer
            mode:
                str; one of 'minerva' or 'nova'; necessary for dataset-specific parsing

        Returns:
            a dict for use in the Layer class constructor below
        """
        if mode == 'minerva':
            input_c = 2
            input_h = 127
            input_w = 47
            if name in ['data0_1', 'data0_2']:
                return {
                    'name': name,
                    'type': 'Input',
                    'bottom': [caffeLayer.name],
                    'top': [],
                    'channels': input_c,
                    'height': input_h,
                    'width': input_w,
                    'include': caffeLayer.include
                }
            else:
                return {
                    'name': name,
                    'type': 'Input',
                    'bottom': [caffeLayer.name],
                    'top': [],
                    'channels': input_c,
                    'height': input_h,
                    'width': 2 * input_w,
                    'include': caffeLayer.include
                }
        elif mode == 'nova':
            return {
                'name': name,
                'type': 'Input',
                'bottom': [caffeLayer.name],
                'top': [],
                'channels': 1,
                'height': 100,
                'width': 80,
                'include': caffeLayer.include
            }
        else:
            print("Unexpected dataset: " + mode)
            return {}

    def _concat_handler(self, layer_name):
        """
        Special handler for concatenation layers.

        Parameters:
            layer_name:
                str; name of concatenation layer to handle
        """
        # In NOVA networks, need to change name of inception concatenators
        if 'inception' in self.layers[layer_name].name:
            self.layers[layer_name].name = layer_name

        axis = self.layers[layer_name].layerParams['axis']

        self.layers[layer_name].layerParams['input_grid'] = []

        arrays = []

        lengths = []

        for bottom in self.layers[layer_name].bottom:
            bottom_layer = self.layers[bottom]
            if bottom_layer.type in ['Convolution', 'Pooling', 'LRN', 'Concat']:
                input_grid = bottom_layer.layerParams['output_grid']
                arrays = arrays + [input_grid]
                self.layers[layer_name].layerParams['input_grid'].append(input_grid)
            if bottom_layer.type in ['Flatten']:
                lengths.append(bottom_layer.layerParams['num_output'])
        if arrays:
            self.layers[layer_name].layerParams['output_grid'] = np.concatenate(arrays, axis=axis)
            self.layers[layer_name].layerParams['num_output'] = \
                self.layers[layer_name].layerParams['output_grid'].shape[0]
        if lengths:
            self.layers[layer_name].layerParams['num_output'] = sum(lengths)

    @staticmethod
    def get_grids(input_layer, output_layer):

        """
        Computes the (padded) input activation grid and output activation grid
        of output_layer on input_layer.
        
        Parameters:
            input_layer: A Layer object.
            
            output_layer: A layer object.
            
        Returns:
            ndarray input_grid, ndarray output_grid
        """
        if input_layer.type == 'Input':
            ip_channels = input_layer.layerParams['channels']
        else:
            ip_channels = input_layer.layerParams['num_output']

        ip_layer_grid = input_layer.layerParams['output_grid'][0]
        ip_layer_grid_h = ip_layer_grid.shape[0]
        ip_layer_grid_w = ip_layer_grid.shape[1]

        op_channels = output_layer.layerParams['num_output']

        kernel_h = output_layer.layerParams['kernel_h']
        kernel_w = output_layer.layerParams['kernel_w']

        stride_h = output_layer.layerParams['stride_h']
        stride_w = output_layer.layerParams['stride_w']

        pad_h = output_layer.layerParams['pad_h']
        pad_w = output_layer.layerParams['pad_w']

        input_grid = np.zeros((ip_channels, ip_layer_grid_h + 2 * pad_h,
                               ip_layer_grid_w + 2 * pad_w))

        op_grid_h = max([1 + (ip_layer_grid_h + 2 * pad_h - kernel_h) / stride_h, 1])
        op_grid_w = max([1 + (ip_layer_grid_w + 2 * pad_w - kernel_w) / stride_w, 1])

        # NOTE: ROUNDING MAY OCCUR-NEED TO IMPLEMENT CHECK TO ENSURE THAT THE
        # KERNEL FITS EVENLY INTO THE INPUT GRID.

        # if not op_grid_h - int(op_grid_h) == 0:
        # print("WARNING: KERNEL_H DOES NOT EVENLY DIVIDE INPUT_H in " +
        #     output_layer.name)
        # print(op_grid_h)
        # if not op_grid_w - int(op_grid_w) == 0:
        # print("WARNING: KERNEL_W DOES NOT EVENLY DIVIDE INPUT_W in " +
        #      output_layer.name)
        # print(op_grid_w)
        output_grid = np.zeros((op_channels, int(op_grid_h),
                                int(op_grid_w)))
        return input_grid, output_grid

    # noinspection PyTypeChecker
    def feed_image(self, img_arr=np.array(None), mode='minerva', hdf5=None, img_num=-1, rimg=False, normalize=True):
        """
        Takes an input image and feeds it into network's internal image
        containers. For MINERvA data, these are the layers data0_0, data0_1, 
        and data0_2. For NOvA data, the layers are data0_0 and data0_1. If
        hdf5 is specified, will look for img within given hdf5 database.
        
        Parameters:
            img_arr: 
                an image from either of the MINERvA or NOVA datasets in
                ndarray form
            
            mode: 
                either 'minerva' or 'nova'; default 'minerva'. Indicates
                how images should be preprocessed
                
            hdf5: 
                str; name of hdf5 file
            
            img_num: 
                int; index into hdf5 image database indicating which image to get
            
            rimg: 
                bool; if True and img_num not specified, will choose a random
                image from the hdf5 image database

            normalize:
                bool; if True, will divide the image by the maximal absolute value of the array
                
        Returns:
            img_index:
                if hdf5 mode was used, will return the index of the image
                in the dataset which was fed to the network
        """
        if img_arr.all() is not None:
            if mode == 'minerva':
                # Preprocess image into desired shape, if necessary
                # Assume image has shape (8, 127, 47)
                img0_X1 = img_arr[0:2]
                img0_X2 = img_arr[2:4]
                img0 = np.concatenate((img0_X1, img0_X2), axis=2)
                img1 = img_arr[4:6]
                img2 = img_arr[6:8]
                if normalize:
                    max0_0 = np.max(np.abs(img0[0]))
                    max0_1 = np.max(np.abs(img0[1]))
                    max1_0 = np.max(np.abs(img1[0]))
                    max1_1 = np.max(np.abs(img1[1]))
                    max2_0 = np.max(np.abs(img2[0]))
                    max2_1 = np.max(np.abs(img2[1]))
                    img0[0] = img0[0] / max0_0
                    img0[1] = img0[1] / max0_1
                    img1[0] = img1[0] / max1_0
                    img1[1] = img1[1] / max1_1
                    img2[0] = img2[0] / max2_0
                    img2[1] = img2[1] / max2_1
                self.layers['data0_0'].layerParams['output_grid'] = img0
                self.layers['data0_0'].imgFeatures['id'] = 'custom_id'
                self.layers['data0_1'].layerParams['output_grid'] = img1
                self.layers['data0_1'].imgFeatures['id'] = 'custom_id'
                self.layers['data0_2'].layerParams['output_grid'] = img2
                self.layers['data0_2'].imgFeatures['id'] = 'custom_id'
            elif mode == 'nova':
                img0 = img_arr[0].reshape((1,) + img_arr[0].shape)
                img1 = img_arr[1].reshape((1,) + img_arr[0].shape)
                if normalize:
                    max0_0 = np.max(np.abs(img0[0]))
                    max1_0 = np.max(np.abs(img1[0]))
                    img0[0] = img0[0] / max0_0
                    img1[0] = img1[0] / max1_0
                self.layers['data0_0'].layerParams['output_grid'] = img0
                self.layers['data0_0'].imgFeatures['id'] = 'custom_id'
                self.layers['data0_1'].layerParams['output_grid'] = img1
                self.layers['data0_1'].imgFeatures['id'] = 'custom_id'
            else:
                print('Cannot yet handle images from the dataset' + mode)
                return None
        elif hdf5:
            if mode == 'minerva':
                data = h5py.File(hdf5)
                imgs = data.get('/img_data')
                u_view = imgs.get('hitimes-u')
                v_view = imgs.get('hitimes-v')
                x_view = imgs.get('hitimes-x')

                if img_num >= 0:
                    if img_num >= u_view.shape[0]:
                        print('Given index is too large: ', img_num)
                        return None
                    index = img_num
                elif rimg:
                    index = np.random.randint(0, u_view.shape[0] - 1)
                else:
                    print('No way to index images specified.')
                    data.close()
                    return None
                x_img = x_view[index]
                u_img = u_view[index]
                v_img = v_view[index]

                if normalize:
                    max0_0 = np.max(np.abs(x_img[0]))
                    max0_1 = np.max(np.abs(x_img[1]))
                    max1_0 = np.max(np.abs(u_img[0]))
                    max1_1 = np.max(np.abs(u_img[1]))
                    max2_0 = np.max(np.abs(v_img[0]))
                    max2_1 = np.max(np.abs(v_img[1]))
                    x_img[0] = x_img[0] / max0_0
                    x_img[1] = x_img[1] / max0_1
                    u_img[0] = u_img[0] / max1_0
                    u_img[1] = u_img[1] / max1_1
                    v_img[0] = v_img[0] / max2_0
                    v_img[1] = v_img[1] / max2_1
                self.layers['data0_0'].layerParams['output_grid'] = x_img
                self.layers['data0_0'].imgFeatures['id'] = index
                self.layers['data0_1'].layerParams['output_grid'] = u_img
                self.layers['data0_1'].imgFeatures['id'] = index
                self.layers['data0_2'].layerParams['output_grid'] = v_img
                self.layers['data0_2'].imgFeatures['id'] = index
                data.close()
            elif mode == 'nova':
                data = h5py.File(hdf5)
                imgs = data.get('/data')

                if img_num >= 0:
                    if img_num >= imgs.shape[0]:
                        print('Given index is too large: ', img_num)
                        return None
                    index = img_num
                elif rimg:
                    index = np.random.randint(0, imgs.shape[0] - 1)
                else:
                    print('No way to index images specified.')
                    data.close()
                    return None
                img = imgs[index]
                if normalize:
                    max0 = np.max(np.abs(img[0]))
                    max1 = np.max(np.abs(img[1]))
                    img[0] = img[0] / max0
                    img[1] = img[1] / max1
                shape = img[0].shape
                self.layers['data0_0'].layerParams['output_grid'] = img[0].reshape((1,) + shape)
                self.layers['data0_0'].imgFeatures['id'] = index
                self.layers['data0_1'].layerParams['output_grid'] = img[1].reshape((1,) + shape)
                self.layers['data0_1'].imgFeatures['id'] = index
                data.close()

            else:
                print('Cannot yet handle images from the dataset' + mode)
                return None
        else:
            print('Invalid input type.')
            return None

    def reset_img_features(self):
        """
        Resets the features of each input layer.
        """
        for layer in self.layers:
            self.layers[layer].imgFeatures = {}

    def get_simple_img_features(self, feature_list, avg_over_channels=False):
        """
        Populates the imgFeatures field of all Input layers with the features
        listed in 'feature_list'.
        **DO NOT USE: NEED TO IMPLEMENT ABILITY TO PASS PARAMETERS TO ATTRIBUTE FUNCTIONS
        CURRENTLY ONLY DEFAULT OPTIONS ARE USABLE**
        
        Parameters:
            
            feature_list:
                list of str; one of the following: ['prop_nonzero_activations',
                'prop_cropped_nonzero', 'horiz_spread', 'vert_spread']
                
            avg_over_channels:
                bool; if True, will store the average of a feature over all
                image channels; otherwise, stores separate entries in the 
                feature dict for each channel
                    
        """
        feature_dict = {
            ### KEYS SHOULD REFERENCE FEATURE EXTRACTORS DEFINED BELOW ###
            'prop_nonzero_activations': self._prop_nonzero_activations,
            'prop_cropped_nonzero': self._prop_cropped_nonzero,
            'horiz_spread': self._horiz_spread,
            'vert_spread': self._vert_spread
        }
        for l in self.inputLayers:
            for feat in feature_list:
                channel_features = []
                for c in range(self.layers[l].layerParams['channels']):
                    channel_features.append((c, feature_dict[feat](self.layers[l].layerParams['output_grid'][c])))
                if avg_over_channels:
                    self.layers[l].imgFeatures[feat] = np.mean([f[1] for f in channel_features])
                else:
                    for cf in channel_features:
                        self.layers[l].imgFeatures[feat + str(cf[0])] = cf[1]

    def get_layer(self, layer_name):
        """
        Returns layer with given name, if it exists.
        
        Parameters:
            layer_name: a string giving the name of the layer to retrieve
            
        Returns:
            a Layer object
        """
        if layer_name not in self.layers.keys():
            print('Layer ' + layer_name + ' does not exist in ' + self.name)
            return 0
        else:
            return self.layers[layer_name]

    @staticmethod
    def intersect_fields(fields):
        """
        Returns the upper-left and lower-right corner points of the
        intersection of a set of rectangular fields.
        
        Parameters:
        
            fields:
                list of tuples of length 4 (i_1, j_1, i_2, j_2), where (i_1, j_1)
                is the upper-left corner of a field and (i_2,j_2) is
                the lower-right corner. (Note: in the degenerate case of a singleton
                set all of i_1, j_1, i_2, and j_2 are equal.)
        
        Returns:
            a 4-tuple (i_1,j_1, i_2, j_2) denoting the upper-left and
            lower-right corner points of a rectangular subgrid.
        """

        max_ulr = max([field[0] for field in fields])
        max_ulc = max([field[1] for field in fields])
        min_lrr = min([field[2] for field in fields])
        min_lrc = min([field[3] for field in fields])

        if (max_ulr <= min_lrr) and (max_ulc <= min_lrc):
            return max_ulr, max_ulc, min_lrr, min_lrc
        else:
            return None

    @staticmethod
    def union_fields(fields=None, points=None):
        """
        Returns an iterator object representing the union of rectangular
        fields or points.
        
        Parameters:
        
            fields:
                list of tuples of length 4 (i_1, j_1, i_2, j_2),
                where (i_1, j_1) is the upper-left corner of a field and (i_2,j_2) is
                the lower-right corner. (Note: in the degenerate case of a singleton
                set all of i_1, j_1, i_2, and j_2 are equal.)
            
            points:
                a list of ordered pairs
        
        Returns:
            a _Union_iter iterator object
        """
        if points is None:
            points = []
        if fields is None:
            fields = []
        if points:
            union_input = [point + point for point in points]
        else:
            union_input = fields
        return UnionIter(union_input)

    def get_flow(self, A, inputLayer, layerPath=None, D=0):
        """
        Returns a list of activations of all layers of layerPath
        into which the activations of A flow. If D is specified, will
        return the flow of A into all child layers of inputLayer of depth <= D.
        
        :param A: A list or tuple of ordered pairs denoting the activations whose
            flow is desired.
        :param inputLayer: The layer whose output grid contains A.
        :param layerPath: A list of layer names. The first element is taken to be
            inputLayer, and any consecutive layers should have a bottom->
            top relationship.
        :param D: The depth relative to layer1 of layers for which we would like
            to know the flow of A.
                
        :return:
            If layerPath is nonempty and D = 0:
                A list of tuples of the form (layer_name, ListFields),
                where ListFields is a list of tuples (i1,j1,i2,j2), where 
                (i1,j1) and (i2,j2) are the upper-left and lower-right corners of
                a subgrid. Any element of ListFields should fit into the output_
                grid member of layer 'layer_name'.
            If D is nonzero:
                Will return a list of tuples of the form (layerPath, get_flow_
                output), where get_flow_output is the result of applying 
                get_flow on layerPath.
        """
        if layerPath is None:
            layerPath = []
        if layerPath:
            for i in range(len(layerPath) - 1):
                if self.layers[layerPath[i]].name not in self.layers[layerPath[i + 1]].bottom:
                    print("Invalid path: " + layerPath[i] + "is not bottom of " +
                          layerPath[i + 1])
                    return [[None, None]]
            return_list = []
            activations = A
            for i in range(len(layerPath) - 1):
                bottom_layer = self.layers[layerPath[i]]
                top_layer = self.layers[layerPath[i + 1]]
                op_shape = top_layer.layerParams['output_grid'][0].shape

                if not activations:
                    print('List of activations in network {}, layer {}, image {} is empty.'.format(self.name,
                                                                                                   bottom_layer.name,
                                                                                                   bottom_layer.imgFeatures[
                                                                                                       'id']))
                    return [[None, None]]

                flow = self._flow_once(activations, bottom_layer, top_layer)
                if not flow:
                    print(
                        'Obtained empty flow from {} to {} in {}.'.format(bottom_layer.name, top_layer.name, self.name))
                    return [[None, None]]
                activations = [p for p in self.union_fields(flow) if
                               0 <= p[0] < op_shape[0] and 0 <= p[1] < op_shape[1]]
                return_list.append((top_layer.name, activations))

            return return_list

        elif D:
            return_list = []
            activations = A

            layer_path = self._get_layer_paths(inputLayer, D)

            for path in layer_path:
                return_list.append((path, self.get_flow(activations, inputLayer, path)))
            return return_list

        else:
            return None

    def _get_layer_paths(self, inputLayer, D):

        """
        Auxiliary recursive function to get the list of all depth D layer
        paths starting at inputLayer.
        
        Inputs:
            inputLayer:
                string name of Layer
            
            D:
                int depth
            
        Returns:
            a list of lists of layer names
        """

        tops = [t for t in self.layers[inputLayer].top if self.layers[t].type in ['Convolution', 'Pooling']]

        if D == 1:
            return [[inputLayer, t] for t in tops]
        else:
            return_list = []
            for t in tops:
                return_list = return_list + [[inputLayer] + path for path in
                                             self._get_layer_paths(t, D - 1)]
            return return_list

    @staticmethod
    def _flow_once(A, bottom, top):
        """
        Returns a list of fields, expressed as tuples (i1, j1, i2, j2), where
        (i1,j1) and (i2,j2) are the upper-left and lower-right corners of the
        rectangular field. Each field is the flow of a particular activation
        in A. A must be contained in the layer bottom, and top must be in the
        list of "top" layers for bottom.
        
        Parameters:
            A: a list of ordered pairs
            bottom: string
            top: string
            
        Returns:
            List of tuples of length 4.
        """

        bottom_layer = bottom

        top_layer = top

        if top_layer.name not in bottom_layer.top:
            print(bottom_layer.name + " does not flow into " + top_layer.name)

        kernel_h = top_layer.layerParams['kernel_h']
        kernel_w = top_layer.layerParams['kernel_w']

        stride_h = top_layer.layerParams['stride_h']
        stride_w = top_layer.layerParams['stride_w']

        pad_h = top_layer.layerParams['pad_h']
        pad_w = top_layer.layerParams['pad_w']

        return_list = []

        for point in A:
            i = point[0]
            j = point[1]

            i1 = int(np.ceil(max([0, (i + pad_h - kernel_h) / stride_h])))
            j1 = int(np.ceil(max([0, (j + pad_w - kernel_w) / stride_w])))
            i2 = int(np.floor((i + pad_h) / stride_h))
            j2 = int(np.floor((j + pad_w) / stride_w))

            return_list.append([i1, j1, i2, j2])

        return return_list

    def get_ERF(self, A, layer_path):
        """
        Returns a list of tuples (layer_name, points), where points
        is a list of ordered pairs constituting the ERF of A in the layer
        layer_name, and A is a set of activations in the last element of
        layer_path.
        
        :param A: a list of ordered pairs
            
        :param layer_path: a list of layer names, each of which feeds into the next
            
        :returns: list of tuples of length 2
        """

        D = len(layer_path)

        points = A

        return_list = []

        for d in range(1, D):

            top = self.layers[layer_path[D - d]]
            bottom = self.layers[layer_path[D - d - 1]]

            fields = []

            kernel_h = top.layerParams['kernel_h']
            kernel_w = top.layerParams['kernel_w']

            stride_h = top.layerParams['stride_h']
            stride_w = top.layerParams['stride_w']

            pad_h = top.layerParams['pad_h']
            pad_w = top.layerParams['pad_w']

            r, c = bottom.layerParams['output_grid'][0].shape

            for point in points:
                i, j = point

                m, n = kernel_h, kernel_w

                i1, j1 = i * stride_h, j * stride_w

                if i1 < pad_h:
                    m = m - (pad_h - i1)
                    i1 = pad_h
                if j1 < pad_w:
                    n = n - (pad_w - j1)
                    j1 = pad_w

                i2 = i1 + m - 1
                j2 = j1 + n - 1

                if i2 > pad_h + r:
                    i2 = pad_h + r
                if j2 > pad_w + c:
                    j2 = pad_w + c

                fields.append([i1, j1, i2, j2])
            union = self.union_fields(fields)

            points = [point for point in union]

            return_list.append((bottom.name, points))

        return return_list

    def get_max_paths(self, inputLayer, weightsOnly=False, convOnly=False,
                      phases=None, inception_unit=False,
                      include_pooling=False):
        """
        Returns the maximal length path starting at inputLayer.
        
        Parameters:
            inputLayer:
                str; name of layer at which all desired paths begin
            
            weightsOnly:
                bool; whether or not to consider only layers which
                have weights/biases. Effectively reduces to considering only
                convolutional and inner product layers. Default: False
            
            convOnly:
                bool; whether or not to consider only convolutional
                layers. If True, the path search will stop when a "Flatten" layer
                is met, as this signals the end of the convolutional segment of
                the network. Default: False
            
            phases:
                list; sublist of ['ALL', 'TEST', 'TRAIN']; indicates the
                phases for which layers should be considered. Default is ['ALL'].
            
            inception_unit:
                bool; whether or not to treat an inception module
                as a single 'layer' (thus contributing only 1 to depth). Default: False
            
            include_pooling:
                bool; whether or not to add pooling layers to
                paths when weightsOnly or convOnly is set to True. Default: False
            
        Returns:
            a list of lists representing all paths from inputLayer to an
            output. The first entry in every path is always inputLayer, and
            the last entry is always a leaf node in the network.
        """
        if phases is None:
            phases = ['ALL']
        ignore_list = ['ReLU', 'PReLU', 'ELU', 'Sigmoid', 'TanH',
                       'Power', 'Exp', 'Log', 'BNLL', 'Threshold',
                       'Bias', 'Scale', 'Dropout']
        # For use when weightsOnly == True or convOnly == True
        silence_list = ['Concat', 'Flatten', 'Pooling', 'Split', 'Slice',
                        'Placeholder', 'LRN']
        silence = False
        if self.layers[inputLayer].type == 'Pooling':
            h = self.layers[inputLayer].layerParams['kernel_h']
            w = self.layers[inputLayer].layerParams['kernel_w']
            if h == 1 and w == 1:
                silence = True

        check_inception = 'inception' in inputLayer \
                          and self.layers[inputLayer].type == 'Concat'

        if weightsOnly or convOnly:
            if include_pooling:
                silence_list.remove('Pooling')
            if self.layers[inputLayer].type in silence_list and not check_inception:
                silence = True

        if convOnly:
            if self.layers[inputLayer].type == 'InnerProduct':
                silence = True

        if not self.layers[inputLayer].phase in phases:
            return []

        if silence:
            inputList = []
        else:
            inputList = [inputLayer]

        tops = self.layers[inputLayer].top

        if not tops:
            return [inputList]
        else:
            if inception_unit:
                current = None
                for top in tops:
                    if 'inception' in top:
                        current = top
                        while self.layers[current].type != 'Concat':
                            look_ahead = self.layers[current].top
                            for t in look_ahead:
                                if self.layers[t].type in ['Convolution',
                                                           'Pooling', 'Concat']:
                                    current = t
                                    break
                        break
                if current:
                    tops = [current]
            return_list = []
            for t in tops:
                if self.layers[t].type in ignore_list:
                    continue

                if self.layers[t].phase in phases:
                    return_list = return_list + \
                                  [inputList + path
                                   for path in self.get_max_paths(t, weightsOnly,
                                                                  convOnly, phases, inception_unit,
                                                                  include_pooling)]

            return return_list

    def path_between_layers(self, shallow_layer, deep_layer, weightsOnly=False,
                            convOnly=False, phases=None,
                            inception_unit=False, include_pooling=False):
        """
        Returns all paths starting at shallow_layer and ending at deep_layer.
        
        Parameters:
            shallow_layer:
                str; the layer at which the path begins

            deep_layer:
                str; the layer at which the path ends
        
            weightsOnly:
                bool; whether or not to consider only layers which
                have weights/biases. Effectively reduces to considering only
                convolutional and inner product layers. Default: False
            
            convOnly:
                bool; whether or not to consider only convolutional
                layers. If True, the path search will stop when a "Flatten" layer
                is met, as this signals the end of the convolutional segment of
                the network. Default: False
            
            phases:
                list; sublist of ['ALL', 'TEST', 'TRAIN']; indicates the
                phases for which layers should be considered. Default is ['ALL'].
            
            inception_unit:
                bool; whether or not to treat an inception module
                as a single 'layer' (thus contributing only 1 to depth). Default: False
            
            include_pooling:
                bool; whether or not to add pooling layers to
                paths when weightsOnly or convOnly is set to True. Default: False
            
        Returns:
            a list of lists representing all paths from shallow_layer to 
            deep_layer. The first entry in every path is always inputLayer, and
            the last entry is always a leaf node in the network.
        """
        if phases is None:
            phases = ['ALL']
        return_list = []
        all_paths = self.get_max_paths(shallow_layer, weightsOnly, convOnly, phases, inception_unit, include_pooling)

        for path in all_paths:
            if deep_layer in path:
                index = path.index(deep_layer)
                return_list.append(path[:index + 1])

        return return_list

    def get_net_depth(self, start_layer, key='MAX', weightsOnly=False,
                      convOnly=False, phases=None,
                      inception_unit=False, include_pooling=False):
        """
        Uses key to return a summary statistic of network depth relative to start_layer.
        Network depth is measured as the length of a path starting at start_layer
        and ending at a network leaf-node (e.g. a loss layer).
        
        Parameters:
            start_layer:
                str; the first layer in every considered path.
            
            key:
                str, one of 'MAX', 'MIN', or 'AVG': specifies whether the
                maximum, minimum, or average depth should be returned. Default: MAX
            
            weightsOnly:
                bool; whether or not to consider only layers which
                have weights/biases. Effectively reduces to considering only
                convolutional and inner product layers. Default: False
            
            convOnly:
                bool; whether or not to consider only convolutional
                layers. If True, the path search will stop when a "Flatten" layer
                is met, as this signals the end of the convolutional segment of
                the network. Default: False
            
            phases:
                list; sublist of ['ALL', 'TEST', 'TRAIN']; indicates the
                phases for which layers should be considered. Default is ['ALL'].
            
            inception_unit:
                bool; whether or not to treat an inception module
                as a single 'layer' (thus contributing only 1 to depth). Default: False
            
            include_pooling:
                bool; whether or not to add pooling layers to
                paths when weightsOnly or convOnly is set to True. Default: False
            
        Returns:
            int/float summary_stat, str paths; paths is a list of all paths achieving
            the 'MAX' or 'MIN', if key is set to either of these
            
        """

        if phases is None:
            phases = ['ALL']
        all_paths = self.get_max_paths(start_layer, weightsOnly, convOnly,
                                       phases, inception_unit, include_pooling)
        all_paths = [p[:-1] for p in all_paths]
        unique_paths = []
        for path in all_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        all_paths = unique_paths
        if key == 'MAX':
            if not all_paths:
                return np.NaN, None
            return_value = max([len(path) for path in all_paths])
            return_list = [path for path in all_paths if len(path) == return_value]
            return return_value, return_list

        elif key == 'MIN':
            if not all_paths:
                return np.NaN, None
            return_value = min([len(path) for path in all_paths])
            return_list = [path for path in all_paths if len(path) == return_value]
            return return_value, return_list

        elif key == 'AVG':
            if not all_paths:
                return np.NaN
            return_value = np.mean([len(path) for path in all_paths])
            return return_value
        else:
            print('Invalid key: ', key)
            return None

    def get_layer_depth(self, layer_name, weightsOnly=False, phases=None,
                        inception_unit=False, include_pooling=False):
        """
        Returns the length of a shortest path from layer_name to an input
        layer which feeds into it. Only considers layers which participate in
        the phases of 'phases'.
        
        Parameters:
            layer_name:
                str; the layer whose depth is desired
            
            weightsOnly:
                bool; whether or not to consider only layers which
                have weights/biases. Effectively reduces to considering only
                convolutional and inner product layers. Default: False
            
            phases:
                list; sublist of ['ALL', 'TEST', 'TRAIN']; indicates the
                phases for which layers should be considered. Default is ['ALL'].
            
            inception_unit:
                bool; whether or not to treat an inception module
                as a single 'layer' (thus contributing only 1 to depth). Default: False
            
            include_pooling:
                bool; whether or not to add pooling layers to
                paths when weightsOnly or convOnly is set to True. Default: False
            
        Returns:
            int len_path, list of list of str shortest_paths
        """

        if phases is None:
            phases = ['ALL']
        input_found = False

        paths = [[layer_name]]

        if self.layers[layer_name].type == 'Input':
            input_found = True

        if inception_unit and 'inception' in layer_name:
            if self.layers[layer_name].type != 'Concat':
                print('Inception layers treated as units, but specified layer',
                      'comes from interior of inception layer.')
                return 0

        while not input_found:
            new_paths = []
            for path in paths:
                for bottom in self.layers[path[0]].bottom:
                    new_paths.append([bottom] + path)

            paths = new_paths

            for path in paths:
                if self.layers[path[0]].type == 'Input':
                    input_found = True

        paths_to_input = [path for path in paths if self.layers[path[0]].type == 'Input']

        remove_list = []

        # Prune paths for phase
        for path in paths_to_input:
            for layer in path:
                if self.layers[layer].phase not in phases:
                    remove_list.append(path)
                    break
        paths_to_input = [path for path in paths_to_input if path not in remove_list]

        # Prune paths for weightsOnly

        if weightsOnly:
            keep_types = ['Convolution', 'InnerProduct', 'Input']
            if include_pooling:
                keep_types = keep_types.append('Pooling')
            for i in range(len(paths_to_input)):
                remove_list = []
                for layer in paths_to_input[i]:
                    if self.layers[layer].type not in keep_types:
                        if self.layers[layer].type == 'Concat' and 'inception' in layer:
                            continue
                        remove_list.append(layer)
                paths_to_input[i] = [layer for layer in
                                     paths_to_input[i] if layer not in remove_list]

        if inception_unit:
            for i in range(len(paths_to_input)):
                remove_list = []
                for layer in paths_to_input[i]:
                    if 'inception' in layer:
                        if self.layers[layer].type == 'Concat':
                            continue
                        remove_list.append(layer)
                paths_to_input[i] = [layer for layer in paths_to_input[i] if
                                     layer not in remove_list]

        len_path = min([len(path) for path in paths_to_input])

        depth = len_path - 1

        shortest_paths = [path for path in paths_to_input if len(path) == len_path]

        return depth, shortest_paths

    def reset_grids(self, keep_inputs=False, reset=None):
        """
        Resets the activation grids of all layers according the key 'reset'.
        
        Parameters:
            keep_inputs:
                bool; if True, input grids will not be reset
            
            reset:
                one of ['zero'], ['ones'], ['gauss', gauss_params];
                specifies whether to reset grids with zeros, ones, or Gaussian noise.
                If Gaussian noise is chosen, the user may enter a tuple
                gauss_params = (mean, sd) to specify parameters of the distribution
                from which the noise is drawn.
        """

        if reset is None:
            reset = ['zero']
        for layer in self.layers.keys():
            if not self.layers[layer].layerParams:
                continue
            if 'input_grid' in self.layers[layer].layerParams.keys():
                bottom_types = [self.layers[b].type for b in self.layers[layer].bottom]

                if keep_inputs and 'Input' in bottom_types:
                    continue

                if not self.layers[layer].layerParams['input_grid']:
                    continue
                ip_shape = self.layers[layer].layerParams['input_grid'].shape

                if reset[0] == 'zeros':
                    self.layers[layer].layerParams['input_grid'] = np.zeros(ip_shape)
                elif reset[0] == 'ones':
                    self.layers[layer].layerParams['input_grid'] = np.ones(ip_shape)
                elif reset[0] == 'gauss':
                    if len(reset) > 1:
                        mean = reset[1][0]
                        sd = reset[1][1]
                    else:
                        mean = 0
                        sd = 1
                    self.layers[layer].layerParams['input_grid'] = \
                        sd * np.random.randn(ip_shape[0], ip_shape[1], ip_shape[2]) + mean
            if 'output_grid' in self.layers[layer].layerParams.keys():
                if keep_inputs and self.layers[layer].type == 'Input':
                    continue

                if not self.layers[layer].layerParams['output_grid']:
                    continue

                op_shape = self.layers[layer].layerParams['output_grid'].shape

                if reset[0] == 'zeros':
                    self.layers[layer].layerParams['output_grid'] = np.zeros(op_shape)
                elif reset[0] == 'ones':
                    self.layers[layer].layerParams['output_grid'] = np.ones(op_shape)
                elif reset[0] == 'gauss':
                    if len(reset) > 1:
                        mean = reset[1][0]
                        sd = reset[1][1]
                    else:
                        mean = 0
                        sd = 1
                    self.layers[layer].layerParams['output_grid'] = \
                        sd * np.random.randn(op_shape[0], op_shape[1], op_shape[2]) + mean

    # %%

    # ############SIMPLE STATIC ATTRIBUTES###########

    def num_conv_layers(self, no_1x1=False, phases=None, inception_unit=False):
        """
        Returns the number of convolutional layers in the network. If no_1x1 is
        set to True, will ignore convolutional layers with 1x1 kernels. Will
        only count layers that participate in phases in 'phases'
        
        Parameters:
            no_1x1:
                bool; Default: False
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            inception_unit:
                bool; indicates whether or not convolutional layers
                within inception modules should be counted. Default: False.
            
        Returns:
            int num_conv_layers
        """

        if phases is None:
            phases = ['ALL']
        counter = 0

        for key in self.layers.keys():
            if self.layers[key].type == 'Convolution' and self.layers[key].phase in phases:
                if no_1x1:
                    h = self.layers[key].layerParams['kernel_h']
                    w = self.layers[key].layerParams['kernel_w']
                    if h == 1 and w == 1:
                        continue
                if inception_unit and 'inception' in self.layers[key].name:
                    continue
                counter += 1

        return counter

    def num_inception_module(self, phases=None):
        """
        Returns the number of inception module in the network which participate
        in the phases of phases.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int num_inception_modules
        """

        if phases is None:
            phases = ['ALL']
        counter = 0

        for key in self.layers.keys():
            if self.layers[key].type == 'Concat' and \
                    self.layers[key].phase in phases and 'inception' in key:
                counter += 1

        return counter

    def num_pooling_layers(self, phases=None, inception_unit=False):
        """
        Returns the number of pooling layers in the network. Only considers
        layers which participate in the phases of 'phases'.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            inception_unit:
                bool; indicates whether or not pooling layers
                within inception modules should be counted. Default: False.
            
        Returns:
            int num_pooling_layers
        """

        if phases is None:
            phases = ['ALL']
        counter = 0

        for key in self.layers.keys():
            if self.layers[key].type == 'Pooling' and self.layers[key].phase in phases:
                h = self.layers[key].layerParams['kernel_h']
                w = self.layers[key].layerParams['kernel_w']
                if h == 1 and w == 1:
                    continue
                if inception_unit and 'inception' in key:
                    continue
                counter += 1

        return counter

    def num_IP_layers(self, phases=None):
        """
        Returns the number of inner product (fully-connected) layers in the network.
        Only considers layers which participate in the phases of 'phases'.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int num_IP_layers
        """

        if phases is None:
            phases = ['ALL']
        counter = 0

        for key in self.layers.keys():
            if self.layers[key].type == 'InnerProduct' and self.layers[key].phase in phases:
                counter += 1

        return counter

    def num_skip_connections(self, phases=None):
        """
        Returns the number of skip connections in phases of 'phases'.
        ***NEEDS IMPLEMENTATION***
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int num_skip_connections
        """
        if phases is None:
            phases = ['ALL']
        # Placeholder statement to avoid warnings; remove on implementation.
        if phases or self:
            pass
        return None

    def len_skip_connections(self, key='MAX', phases=None):
        """
        Uses 'key' to return a summary statistic for the length of skip connections
        in the network which participate in the phases of 'phases'.
        ****NEEDS IMPLEMENTATION******
        
        Parameters:
            
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int/float summary_stat, str layer_name; layer_name is the layer achieving
            the 'MAX' or 'MIN', if key is set to either of these
        """

        if phases is None:
            phases = ['ALL']

        # Placeholder statement to avoid warnings; remove on implementation.
        if phases or key or self:
            pass
        return None

    def num_IP_neurons(self, key='MAX', phases=None):
        """
        Uses 'key' to return a summary statistic for the number of neurons
        per inner product layer.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        if phases is None:
            phases = ['ALL']
        IP_layers = [layer for layer in self.layers.keys() if
                     self.layers[layer].type == 'InnerProduct' and
                     self.layers[layer].phase in phases]

        if key == 'MAX':
            if not IP_layers:
                return np.NaN, None
            return_value = max([self.layers[layer].layerParams['num_output'] for
                                layer in IP_layers])
            return_list = [layer for layer in IP_layers
                           if self.layers[layer].layerParams['num_output'] == return_value]
            return return_value, return_list

        elif key == 'MIN':
            if not IP_layers:
                return np.NaN, None
            return_value = min([self.layers[layer].layerParams['num_output'] for
                                layer in IP_layers])
            return_list = [layer for layer in IP_layers
                           if self.layers[layer].layerParams['num_output'] == return_value]
            return return_value, return_list

        elif key == 'AVG':
            if not IP_layers:
                return np.NaN
            return_value = np.mean([self.layers[layer].layerParams['num_output'] for
                                    layer in IP_layers])
            return return_value
        else:
            print('Invalid key: ', key)
            return None

    def num_IP_weights(self, key='MAX', phases=None):
        """
        Uses 'key' to return a summary statistic for the number of weights in
        a given IP layer. The number of weights is computed as num_input*num_output.
        Only layers participating in the phases of 'phases' are considered.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
            
        """
        if phases is None:
            phases = ['ALL']
        IP_layers = [layer for layer in self.layers.keys() if
                     self.layers[layer].type == 'InnerProduct' and
                     self.layers[layer].phase in phases]
        if key == 'MAX':
            if not IP_layers:
                return np.NaN, None
            return_value = max([self.layers[layer].layerParams['num_output'] *
                                self.layers[layer].layerParams['num_input'] for
                                layer in IP_layers])
            return_list = [layer for layer in IP_layers
                           if self.layers[layer].layerParams['num_output'] *
                           self.layers[layer].layerParams['num_input'] == return_value]
            return return_value, return_list

        elif key == 'MIN':
            if not IP_layers:
                return np.NaN, None
            return_value = min([self.layers[layer].layerParams['num_output'] *
                                self.layers[layer].layerParams['num_input'] for
                                layer in IP_layers])
            return_list = [layer for layer in IP_layers
                           if self.layers[layer].layerParams['num_output'] *
                           self.layers[layer].layerParams['num_input'] == return_value]
            return return_value, return_list

        elif key == 'AVG':
            if not IP_layers:
                return np.NaN
            return_value = np.mean([self.layers[layer].layerParams['num_output'] *
                                    self.layers[layer].layerParams['num_input'] for
                                    layer in IP_layers])
            return return_value
        else:
            print('Invalid key: ', key)
            return None

    def num_splits(self, phases=None, return_splits=False):
        """
        Returns the number of splits in the network. A 'split' is said to
        occur if a layer has multiple tops. Only layers participating in phases
        of 'phases' are considered. Nonlinearity layers are not counted towards
        the count to determine if a layer splits or not.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            return_splits:
                bool; determines whether a list of split layers
                should be returned. Default: False
            
        Returns:
            int num_splits, [list of str] layer_names
        """

        if phases is None:
            phases = ['ALL']
        ignore_types = ['ReLU', 'PReLU', 'ELU', 'Sigmoid', 'TanH',
                        'Power', 'Exp', 'Log', 'BNLL', 'Threshold',
                        'Bias', 'Scale', 'Dropout']
        split_counter = 0
        splits = []

        for layer in self.layers.keys():
            if self.layers[layer].phase not in phases:
                continue
            counter = 0
            for top in self.layers[layer].top:
                if self.layers[top].type not in ignore_types:
                    counter += 1

            if counter > 1:
                split_counter += 1
                if return_splits:
                    splits.append(layer)

        if return_splits:
            return split_counter, splits
        else:
            return split_counter

    def split_widths(self, key='MAX', phases=None):
        """
        Uses 'key' to return a summary statistic for the widths of splits
        in the network. The width of a split is defined as the number of
        layers to which it points (not including nonlinearity layers). Only
        layers participating in the phases of 'phases' are considered.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        if phases is None:
            phases = ['ALL']
        ignore_types = ['ReLU', 'PReLU', 'ELU', 'Sigmoid', 'TanH',
                        'Power', 'Exp', 'Log', 'BNLL', 'Threshold',
                        'Bias', 'Scale', 'Dropout']

        splits = self.num_splits(phases=phases, return_splits=True)[1]

        widths = []
        for split in splits:
            tops = [top for top in self.layers[split].top if
                    top not in ignore_types]
            widths.append((len(tops), split))
        if key == 'MAX':
            if not widths:
                return np.NaN, None
            return_value = max([width[0] for width in widths])
            return_list = [width[1] for width in widths if width[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not widths:
                return np.NaN, None
            return_value = min([width[0] for width in widths])
            return_list = [width[1] for width in widths if width[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not widths:
                return np.NaN
            return_value = np.mean([width[0] for width in widths])
            return return_value
        else:
            print("Invalid key: ", key)
            return None

    def num_concats(self, phases=None, return_concats=False):
        """
        Returns the number of concatenations in the network. A concatenation
        is defined as any layer which receives input from multiple layers; that
        is, any layer with multiple entries in its 'bottom' list. Only layers 
        participating in the phases of 'phases' are considered.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            return_concats:
                bool; if True will return a list of all layers
                where a concatenation occurs
            
        Returns:
            int num_concats, [list of str] layer_names
        """

        if phases is None:
            phases = ['ALL']
        concat_counter = 0
        concats = []

        for layer in self.layers.keys():
            if len(self.layers[layer].bottom) > 1 and self.layers[layer].phase in phases:
                concat_counter += 1
                if return_concats:
                    concats.append(layer)
        if return_concats:
            return concat_counter, concats
        else:
            return concat_counter

    def concat_widths(self, key='MAX', phases=None):
        """
        Uses 'key' to return a summary statistic for the widths of concats
        in the network. The width of a concat is defined as the number of
        layers which feed into it. Only layers participating in the phases 
        of 'phases' are considered.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        if phases is None:
            phases = ['ALL']
        concats = self.num_concats(phases=phases, return_concats=True)[1]

        widths = []
        for concat in concats:
            bottoms = [bottom for bottom in self.layers[concat].bottom]
            widths.append((len(bottoms), concat))
        if key == 'MAX':
            if not widths:
                return np.NaN, None
            return_value = max([width[0] for width in widths])
            return_list = [width[1] for width in widths if width[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not widths:
                return np.NaN, None
            return_value = min([width[0] for width in widths])
            return_list = [width[1] for width in widths if width[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not widths:
                return np.NaN
            return_value = np.mean([width[0] for width in widths])
            return return_value
        else:
            print("Invalid key: ", key)
            return None

    def conv_ker_area(self, key='MAX', phases=None, include_inception=True):
        """
        Uses 'key' to return a summary statistic for the areas of convolutional 
        kernels in the network. Only layers participating in the phases 
        of 'phases' are considered.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        if phases is None:
            phases = ['ALL']
        conv_layers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Convolution']
        conv_layers = [l for l in conv_layers if self.layers[l].phase in phases]
        if not include_inception:
            conv_layers = [layer for layer in conv_layers if
                           'inception' not in layer]
        areas = []
        for layer in conv_layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']
            area = h * w
            areas.append((area, layer))

        if key == 'MAX':
            if not areas:
                return np.NaN, None
            return_value = max([area[0] for area in areas])
            return_list = [area[1] for area in areas if area[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not areas:
                return np.NaN, None
            return_value = min([area[0] for area in areas])
            return_list = [area[1] for area in areas if area[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not areas:
                return np.NaN
            return_value = np.mean([area[0] for area in areas])
            return return_value
        else:
            print("Invalid key: ", key)
            return None

    def num_conv_features(self, key='MAX', phases=None, include_inception=True):
        """
        Uses 'key' to return a summary statistic for the number of features in 
        convolutional kernels in the network. Only layers participating in the 
        phases of 'phases' are considered.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        if phases is None:
            phases = ['ALL']
        conv_layers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Convolution']

        conv_layers = [l for l in conv_layers if self.layers[l].phase in phases]

        if not include_inception:
            conv_layers = [layer for layer in conv_layers if
                           'inception' not in layer]
        features = []
        for layer in conv_layers:
            feat = self.layers[layer].layerParams['num_output']
            features.append((feat, layer))

        if key == 'MAX':
            if not features:
                return np.NaN, None
            return_value = max([feat[0] for feat in features])
            return_list = [feat[1] for feat in features if feat[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not features:
                return np.NaN, None
            return_value = min([feat[0] for feat in features])
            return_list = [feat[1] for feat in features if feat[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not features:
                return np.NaN
            return_value = np.mean([feat[0] for feat in features])
            return return_value
        else:
            print("Invalid key: ", key)
            return None

    def prop_conv_into_pool(self, phases=None, include_inception=True):
        """
        Returns the proportion of convolutional layers which are followed by 
        a pooling layer. Only layers participating in phases of 'phases' are
        considered. Also returns the list of layers which are followed by
        pooling layers.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            float proportion, list of str pooled_conv_layers
            
        """
        if phases is None:
            phases = ['ALL']
        pool_layers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Pooling']
        pool_layers = [l for l in pool_layers if self.layers[l].phase in phases]
        follows_pooling = []
        for layer in pool_layers:
            for bottom in self.layers[layer].bottom:
                if self.layers[bottom].type == 'Pooling':
                    h = self.layers[bottom].layerParams['kernel_h']
                    w = self.layers[bottom].layerParams['kernel_w']
                    if h == 1 and w == 1:
                        continue
                    else:
                        follows_pooling.append(layer)
                        break
        pool_layers = [layer for layer in pool_layers if layer not in follows_pooling]

        if not include_inception:
            pool_layers = [layer for layer in pool_layers if
                           'inception' not in layer]

        conv_layers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Convolution']
        if not include_inception:
            conv_layers = [layer for layer in conv_layers if
                           'inception' not in layer]

        num_conv_layers = len(conv_layers)

        if num_conv_layers == 0 or len(pool_layers) == 0:
            return 0, []

        pooled_conv_layers = []

        for layer in pool_layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']
            if h == 1 and w == 1:
                continue
            for bottom in self.layers[layer].bottom:
                if self.layers[bottom].type in ['LRN', 'Concat', 'Convolution']:
                    pooled_conv_layers.append(bottom)
                    break

        proportion = len(pooled_conv_layers) / num_conv_layers
        return proportion, pooled_conv_layers

    def prop_pool_into_pool(self, phases=None, include_inception=True):
        """
        Returns the proportion of pooling layers which are followed by 
        a pooling layer. Only layers participating in phases of 'phases' are
        considered. Also returns the list of layers which are followed by
        pooling layers.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            float proportion, list of str pooled_pool_layers
            
        """
        if phases is None:
            phases = ['ALL']
        pool_layers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Pooling']
        pool_layers = [l for l in pool_layers if self.layers[l].phase in phases]
        fake_pooling = []
        for layer in pool_layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']
            if h == 1 and w == 1:
                fake_pooling.append(layer)
        pool_layers = [layer for layer in pool_layers if layer not in fake_pooling]

        if not include_inception:
            pool_layers = [layer for layer in pool_layers if
                           'inception' not in layer]

        num_pool_layers = len(pool_layers)

        if num_pool_layers == 0:
            return 0, []

        pooled_pool_layers = []

        for layer in pool_layers:
            for bottom in self.layers[layer].bottom:
                if bottom in pool_layers:
                    pooled_pool_layers.append(bottom)
                    break

        proportion = len(pooled_pool_layers) / num_pool_layers
        return proportion, pooled_pool_layers

    def prop_padded_conv(self, phases=None, include_inception=True):
        """
        Returns the proportion of convolutional layers which are padded. 
        Only layers participating in phases of 'phases' are considered. 
        Also returns the list of convolutional layers which are padded.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            float proportion, list of str padded_conv_layers
            
        """
        if phases is None:
            phases = ['ALL']
        conv_layers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Convolution']
        conv_layers = [l for l in conv_layers if self.layers[l].phase in phases]

        if not include_inception:
            conv_layers = [layer for layer in conv_layers if
                           'inception' not in layer]
        padded_layers = []

        num_conv_layers = len(conv_layers)

        if num_conv_layers == 0:
            return 0, []

        for layer in conv_layers:
            h = self.layers[layer].layerParams['pad_h']
            w = self.layers[layer].layerParams['pad_w']

            if h == 0 and w == 0:
                continue
            else:
                padded_layers.append(layer)

        proportion = len(padded_layers) / num_conv_layers
        return proportion, padded_layers

    def prop_same_padded_conv(self, phases=None, include_inception=True):
        """
        Returns the proportion of convolutional layers which are same-padded. 
        Only layers participating in phases of 'phases' are considered. 
        Also returns the list of convolutional layers which are same-padded.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            float proportion, list of str same_padded_conv_layers
            
        """
        if phases is None:
            phases = ['ALL']
        conv_layers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Convolution']
        conv_layers = [l for l in conv_layers if self.layers[l].phase in phases]
        if not include_inception:
            conv_layers = [layer for layer in conv_layers if
                           'inception' not in layer]
        same_padded_layers = []

        num_conv_layers = len(conv_layers)

        if num_conv_layers == 0:
            return 0, []

        for layer in conv_layers:
            h = self.layers[layer].layerParams['pad_h']
            w = self.layers[layer].layerParams['pad_w']

            ip_grid = self.layers[layer].layerParams['input_grid'][0]
            r = ip_grid.shape[0]
            c = ip_grid.shape[1]
            ip_grid = ip_grid[h:r - h, w:c - w]
            op_grid = self.layers[layer].layerParams['input_grid'][0]

            if h == 0 and w == 0:
                continue
            elif ip_grid.shape != op_grid.shape:
                continue
            else:
                same_padded_layers.append(layer)

        proportion = len(same_padded_layers) / num_conv_layers
        return proportion, same_padded_layers

    def prop_1x1_conv(self, phases=None, include_inception=True):
        """
        Returns the proportion of convolutional layers wwith 1x1 kernels. 
        Only layers participating in phases of 'phases' are considered. 
        Also returns the list of convolutional layers with 1x1 kernels.
        
        Parameters:
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            float proportion, list of str same_padded_conv_layers
            
        """
        if phases is None:
            phases = ['ALL']
        conv_layers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Convolution']

        conv_layers = [l for l in conv_layers if self.layers[l].phase in phases]

        if not include_inception:
            conv_layers = [layer for layer in conv_layers if
                           'inception' not in layer]

        num_conv_layers = len(conv_layers)

        if num_conv_layers == 0:
            return 0, []

        conv_1x1 = []

        for layer in conv_layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']

            if h == 1 and w == 1:
                conv_1x1.append(layer)

        proportion = len(conv_1x1) / num_conv_layers
        return proportion, conv_1x1

    def prop_square_kernels(self, phases=None, tol=0.01, convOnly=False,
                            include_inception=True):
        """
        Returns the proportion of convolutional and pooling layers whose kernels
        have a height/width ratio which is within tol of 1. Setting tol to 0
        will yield layers with perfectly square kernels.
        
        Parameters:
            tol:
                float; specifies the maximum distance from 1 that a h/w ratio
                may be for the corresponding kernel to be considered square. Default: 0.01
            
            convOnly:
                bool; if True, will ignore pooling layers
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            float proportion, list of str square_kernels
        
        """
        if phases is None:
            phases = ['ALL']
        layers = [layer for layer in self.layers.keys() if
                  self.layers[layer].type in ['Convolution', 'Pooling']]

        layers = [l for l in layers if self.layers[l].phase in phases]

        remove_list = []
        for layer in layers:
            if self.layers[layer].type == 'Pooling':
                h = self.layers[layer].layerParams['kernel_h']
                w = self.layers[layer].layerParams['kernel_w']

                if h == 1 and w == 1:
                    remove_list.append(layer)
        layers = [layer for layer in layers if layer not in remove_list]

        if convOnly:
            layers = [layer for layer in layers if
                      self.layers[layer].type == 'Convolution']
        if not include_inception:
            layers = [layer for layer in layers if
                      'inception' not in layer]

        num_layers = len(layers)

        if num_layers == 0:
            return 0, []

        square_kernels = []

        for layer in layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']

            if abs(h / w - 1) <= tol:
                square_kernels.append(layer)

        proportion = len(square_kernels) / num_layers
        return proportion, square_kernels

    def prop_horiz_kernels(self, phases=None, tol=16 / 9, convOnly=False,
                           include_inception=True):
        """
        Returns the proportion of convolutional and pooling layers whose kernels
        have a width-to-height ratio which is greater than tol.
        
        Parameters:
            tol:
                float; specifies the minimum size that a w/h ratio
                must be for the corresponding kernel to be considered horizontal. Default:16/9
            
            convOnly:
                bool; if True, will ignore pooling layers
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            float proportion, list of str horiz_kernels
        
        """
        if phases is None:
            phases = ['ALL']
        layers = [layer for layer in self.layers.keys() if
                  self.layers[layer].type in ['Convolution', 'Pooling']]

        layers = [l for l in layers if self.layers[l].phase in phases]

        remove_list = []
        for layer in layers:
            if self.layers[layer].type == 'Pooling':
                h = self.layers[layer].layerParams['kernel_h']
                w = self.layers[layer].layerParams['kernel_w']

                if h == 1 and w == 1:
                    remove_list.append(layer)
        layers = [layer for layer in layers if layer not in remove_list]

        if convOnly:
            layers = [layer for layer in layers if
                      self.layers[layer].type == 'Convolution']
        if not include_inception:
            layers = [layer for layer in layers if
                      'inception' not in layer]

        num_layers = len(layers)

        if num_layers == 0:
            return 0, []

        horiz_kernels = []

        for layer in layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']

            if abs(w / h) > tol:
                horiz_kernels.append(layer)

        proportion = len(horiz_kernels) / num_layers
        return proportion, horiz_kernels

    def prop_vert_kernels(self, phases=None, tol=16 / 9, convOnly=False,
                          include_inception=True):
        """
        Returns the proportion of convolutional and pooling layers whose kernels
        have a height-to-width ratio which is greater than tol.
        
        Parameters:
            tol:
                float; specifies the minimum size that a h/w ratio
                must be for the corresponding kernel to be considered vertical. Default:16/9
            
            convOnly:
                bool; if True, will ignore pooling layers
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            float proportion, list of str vert_kernels
        
        """
        if phases is None:
            phases = ['ALL']
        layers = [layer for layer in self.layers.keys() if
                  self.layers[layer].type in ['Convolution', 'Pooling']]

        layers = [l for l in layers if self.layers[l].phase in phases]

        remove_list = []
        for layer in layers:
            if self.layers[layer].type == 'Pooling':
                h = self.layers[layer].layerParams['kernel_h']
                w = self.layers[layer].layerParams['kernel_w']

                if h == 1 and w == 1:
                    remove_list.append(layer)
        layers = [layer for layer in layers if layer not in remove_list]

        if convOnly:
            layers = [layer for layer in layers if
                      self.layers[layer].type == 'Convolution']
        if not include_inception:
            layers = [layer for layer in layers if
                      'inception' not in layer]

        num_layers = len(layers)

        if num_layers == 0:
            return 0, []

        vert_kernels = []

        for layer in layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']

            if abs(h / w) > tol:
                vert_kernels.append(layer)

        proportion = len(vert_kernels) / num_layers
        return proportion, vert_kernels

    def num_nonlinearities(self, nl='ReLU', phases=None, convOnly=False,
                           include_inception=True):
        """
        Returns the number of nonlinearities of type 'nl' in the network.
        
        Parameters:
            nl:
                one of the following:['ReLU', 'PReLU', 'ELU', 'Sigmoid', 'TanH',
                'Power', 'Exp', 'Log', 'BNLL', 'Threshold',
                'Bias', 'Scale']. Default: 'ReLU'
            
            convOnly:
                bool; if True, will ignore pooling layers
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            int num_nl, list of str layers_with_nl
        """
        if phases is None:
            phases = ['ALL']
        layers = [layer for layer in self.layers.keys() if
                  self.layers[layer].type in ['Convolution', 'InnerProduct']]

        layers = [l for l in layers if self.layers[l].phase in phases]

        if convOnly:
            layers = [layer for layer in layers if
                      self.layers[layer].type == 'Convolution']
        if not include_inception:
            layers = [layer for layer in layers if
                      'inception' not in layer]

        layers_with_nl = []

        for layer in layers:
            if self.layers[layer].layerParams['nonlinearity'] == nl:
                layers_with_nl.append(layer)

        return len(layers_with_nl), layers_with_nl

    def num_pool_type(self, pool_type='MAX', phases=None,
                      include_inception=True):
        """
        Returns the number of pooling layers of type 'pool_type' in the network.
        
        Parameters:
            pool_type:
                one of the following:['MAX', 'AVG', 'STOCHASTIC']. Default: 'ReLU'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception:
                If false, convolutional layers appearing in
                inception modules will not be considered.
        
        Returns:
            int num_pool_type, list of str layers_with_pool_type
        """
        if phases is None:
            phases = ['ALL']
        layers = [layer for layer in self.layers.keys() if
                  self.layers[layer].type in ['Pooling']]

        layers = [l for l in layers if self.layers[l].phase in phases]

        remove_list = []
        for layer in layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']

            if h == 1 and w == 1:
                remove_list.append(layer)
        layers = [layer for layer in layers if layer not in remove_list]

        if not include_inception:
            layers = [layer for layer in layers if
                      'inception' not in layer]
        if pool_type == 'MAX':
            code = 0
        elif pool_type == 'AVG':
            code = 1
        elif pool_type == 'STOCHASTIC':
            code = 2
        else:
            print('Invalid pooling type: ', pool_type)
            return None

        layers_with_pool_type = []

        for layer in layers:
            if self.layers[layer].layerParams['pool'] == code:
                layers_with_pool_type.append(layer)

        return len(layers_with_pool_type), layers_with_pool_type

    def grid_reduction_consecutive(self, key='MAX', inception_unit=False,
                                   include_pooling=False, phases=None,
                                   include_1x1=False, dim='a'):
        """
        Uses 'key' to return a summary statistic of the percent reduction in
        activation grid 'dim' between consecutive convolutional layers. Percent
        reduction is computed as 1 - (output_dim/input_dim) and 
        represents the amound of area/height/width of the input_grid which is
        lost in passing through the convolution, pooling, or inception.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            inception_unit:
                bool; determines whether inception modules are considered
                as a single layer. Default: False
            
            include_pooling:
                bool; determines whether pooling layers should
                be explicitly included in the computation. Default: False
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_1x1:
                bool; Determines whether or not a 1x1 convolutional layer
                should be included. Default: False
            
            dim:
                one of 'a', 'h', or 'w'--indicates that reduction in area,
                height, or width should be returned, respectively. Default: 'a'
            
        Returns:
            float percent_reduction,[list of str optimal_paths (for 'MAX' or 'MIN')]
            
        """

        if phases is None:
            phases = ['ALL']
        inputLayers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Input']
        paths = []

        for layer in inputLayers:
            paths = paths + self.get_max_paths(layer, convOnly=True,
                                               include_pooling=include_pooling,
                                               phases=phases,
                                               inception_unit=inception_unit)

        reductions = []

        for i in range(len(paths)):
            remove_list = []
            for layer in paths[i]:
                if self.layers[layer].type in ['Pooling', 'Convolution']:
                    h = self.layers[layer].layerParams['kernel_h']
                    w = self.layers[layer].layerParams['kernel_w']
                    check_1x1 = (not include_1x1) and self.layers[layer].type == 'Convolution' and h == 1 and w == 1
                    if (self.layers[layer].type == 'Pooling' and h == 1 and w == 1) \
                            or check_1x1:
                        remove_list.append(layer)
            paths[i] = [layer for layer in paths[i] if layer not in remove_list]
        # paths = np.unique(np.ravel(paths))
        unique_paths = []
        for path in paths:
            for p in path:
                if p not in unique_paths:
                    unique_paths.append(p)
        paths = unique_paths
        for layer in paths:
            if self.layers[layer].type not in ['Convolution', 'Pooling']:
                continue
            pad_h = self.layers[layer].layerParams['pad_h']
            pad_w = self.layers[layer].layerParams['pad_w']

            ip_h, ip_w = self.layers[layer].layerParams['input_grid'][0].shape
            ip_h, ip_w = ip_h - pad_h, ip_w - pad_w

            op_h, op_w = self.layers[layer].layerParams['output_grid'][0].shape

            ip_area = ip_h * ip_w
            op_area = op_h * op_w

            if dim == 'a':
                pr = 1 - op_area / ip_area
            elif dim == 'h':
                pr = 1 - op_h / ip_h
            elif dim == 'w':
                pr = 1 - op_w / ip_w
            else:
                print('Invalid dimension: ', dim)
                return 0

            reductions.append((pr, layer))

        if key == 'MAX':
            if not reductions:
                return np.NaN, None
            return_value = max([reduction[0] for reduction in reductions])
            return_list = [reduction[1] for reduction in
                           reductions if reduction[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not reductions:
                return np.NaN, None
            return_value = min([reduction[0] for reduction in reductions])
            return_list = [reduction[1] for reduction in
                           reductions if reduction[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not reductions:
                return np.NaN
            return_value = np.mean([reduction[0] for reduction in reductions])
            return return_value

    def grid_reduction_total(self, key='MAX', phases=None, dim='a'):
        """
        Uses 'key' to return a summary statistic of the percent reduction in
        activation grid area between input layers and final convolutional layers.
        Percent reduction is computed as 1 - (output_dim/input_dim) and 
        represents the amound of area/height/width of the input_grid which is
        lost in passing through the convolution, pooling, or inception.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            dim:
                one of 'a', 'h', or 'w'--indicates that reduction in area,
                height, or width should be returned, respectively. Default: 'a'
            
        Returns:
            float percent_reduction,[list of str optimal_paths (for 'MAX' or 'MIN')]
            
        """

        if phases is None:
            phases = ['ALL']
        inputLayers = [layer for layer in self.layers.keys() if
                       self.layers[layer].type == 'Input']
        paths = []

        for layer in inputLayers:
            paths = paths + self.get_max_paths(layer, convOnly=True, phases=phases, inception_unit=True)

        pairs = []
        for path in paths:
            end_found = False
            counter = len(path) - 1
            while not end_found and counter >= 0:
                tail = path[counter]
                check_inception = 'inception' in tail and self.layers[tail].type == 'Concat'
                if self.layers[tail].type in ['Pooling', 'Convolution'] or check_inception:
                    end_found = True
                    if (path[0], tail) not in pairs:
                        pairs.append((path[0], tail))
                counter -= 1

        reductions = []
        for pair in pairs:
            ip_layer = pair[0]
            op_layer = pair[1]

            ip_h, ip_w = self.layers[ip_layer].layerParams['output_grid'][0].shape
            op_h, op_w = self.layers[op_layer].layerParams['output_grid'][0].shape

            ip_area = ip_h * ip_w
            op_area = op_h * op_w

            if dim == 'a':
                pr = 1 - op_area / ip_area
            elif dim == 'h':
                pr = 1 - op_h / ip_h
            elif dim == 'w':
                pr = 1 - op_w / ip_w
            else:
                print("Invalid dimension: ", dim)
                return 0

            reductions.append((pr, pair))
        if key == 'MAX':
            if not reductions:
                return np.NaN, None
            return_value = max([reduction[0] for reduction in reductions])
            return_list = [reduction[1] for reduction in
                           reductions if reduction[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not reductions:
                return np.NaN, None
            return_value = min([reduction[0] for reduction in reductions])
            return_list = [reduction[1] for reduction in
                           reductions if reduction[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not reductions:
                return np.NaN
            return_value = np.mean([reduction[0] for reduction in reductions])
            return return_value

    def prop_nonoverlapping(self, layer_type='ALL', phases=None,
                            include_1x1=False, ignore_inception=False):
        """
        Returns the proportion of convolutional or pooling layers (specified by
        'layer_type') which have non-overlapping strides.
        
        Parameters:
        
            layer_type:
                one of 'ALL', 'CONV', or 'POOL'; indicates which layers
                should be considered
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_1x1:
                bool; whether or not to consider convolutional layers
                with 1x1 kernels. Default: False
            
            ignore_inception:
                bool; whether or not to ignore layers within
                inception modules. Default: False
            
        Returns:
            float prop, list of str nonoverlapping_layers
        """
        if phases is None:
            phases = ['ALL']
        if layer_type == 'ALL':
            keep_types = ['Convolution', 'Pooling']
        elif layer_type == 'CONV':
            keep_types = ['Convolution']
        elif layer_type == 'POOL':
            keep_types = ['Pooling']
        else:
            print('Invalid layer type: ', layer_type)
            return 0

        layers = [l for l in self.layers.keys() if self.layers[l].type in keep_types]

        layers = [l for l in layers if self.layers[l].phase in phases]

        non_ol = []

        num_layers = len(layers)

        if num_layers == 0:
            return 0, []

        for layer in layers:
            if ('inception' in layer) and ignore_inception:
                num_layers -= 1
                continue
            if not include_1x1 or self.layers[layer].type == 'Pooling':
                h = self.layers[layer].layerParams['kernel_h']
                w = self.layers[layer].layerParams['kernel_w']
                if h == 1 and w == 1:
                    num_layers -= 1
                    continue

            kh = self.layers[layer].layerParams['kernel_h']
            kw = self.layers[layer].layerParams['kernel_w']

            sh = self.layers[layer].layerParams['stride_h']
            sw = self.layers[layer].layerParams['stride_w']

            if kh <= sh and kw <= sw:
                non_ol.append(layer)

        prop = len(non_ol) / num_layers

        return prop, non_ol

    def stride_dims(self, key='MAX', layer_type='ALL', phases=None,
                    include_1x1=False, ignore_inception=False,
                    dim='h'):
        """
        Uses 'key' to return a summary statistic for the strides of layers in
        the network. 
        
        Parameters:
            
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
        
            layer_type:
                one of 'ALL', 'CONV', or 'POOL'; indicates which layers
                should be considered
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_1x1:
                bool; whether or not to consider convolutional layers
                with 1x1 kernels. Default: False
            
            ignore_inception:
                bool; whether or not to ignore layers within
                inception modules. Default: False
            
            dim:
                one of 'h' or 'w'; determines which stride dimension should be
                considered; default is 'h'
            
        Returns:
            float summary_stat, [list of str optimal_layers; used with 'key' 
            = 'MAX' or 'MIN']
        """

        if phases is None:
            phases = ['ALL']
        if layer_type == 'ALL':
            keep_types = ['Convolution', 'Pooling']
        elif layer_type == 'CONV':
            keep_types = ['Convolution']
        elif layer_type == 'POOL':
            keep_types = ['Pooling']
        else:
            print('Invalid layer type: ', layer_type)
            return 0

        layers = [l for l in self.layers.keys() if self.layers[l].type in keep_types]

        layers = [l for l in layers if self.layers[l].phase in phases]

        layer_stats = []

        for layer in layers:
            if ('inception' in layer) and ignore_inception:
                continue
            if not include_1x1 or self.layers[layer].type == 'Pooling':
                h = self.layers[layer].layerParams['kernel_h']
                w = self.layers[layer].layerParams['kernel_w']
                if h == 1 and w == 1:
                    continue

            sh = self.layers[layer].layerParams['stride_h']
            sw = self.layers[layer].layerParams['stride_w']

            if dim == 'h':
                stat = sh
            elif dim == 'w':
                stat = sw
            else:
                print('Invalid stride dimension: ', dim)
                return 0

            layer_stats.append((stat, layer))

        if key == 'MAX':
            if not layer_stats:
                return np.NaN, None
            return_value = max([l[0] for l in layer_stats])
            return_list = [l[1] for l in layer_stats if l[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not layer_stats:
                return np.NaN, None
            return_value = min([l[0] for l in layer_stats])
            return_list = [l[1] for l in layer_stats if l[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not layer_stats:
                return np.NaN
            return_value = np.mean([l[0] for l in layer_stats])
            return return_value
        else:
            print('Invalid statistic key: ', key)
            return 0

    def ratio_features_to_depth(self, key='MAX', phases=None,
                                include_IP=True, inception_unit=True,
                                include_pooling=False, include_1x1=True):
        """
        Uses key to return a statistical summary of layer's ratios of features
        to depth. Only layers with weights are considered. By default, 
        inception modules are treated as single layers and inner product layers
        are considered.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_1x1:
                bool; whether or not to consider convolutional layers
                with 1x1 kernels. Default: True
            
            include_IP:
                bool; whether or not to consider inner product layers; Default: True
            
            inception_unit:
                bool; whether or not to treat inception modules
                as single layers; Default: True
            
            include_pooling:
                bool; whether or not to include pooling layers for
                computation of depth; Default: False
            
        Returns:
            float stat, [list of str optimal_layers; used when key = 'MAX' or 'MIN']
            
        """

        # Get all layers with features, subject to the constraints of 'phases'
        # 'include_IP', 'inception_unit', and 'include_1x1'

        if phases is None:
            phases = ['ALL']
        keep_types = ['Convolution', 'Concat']

        if include_IP:
            keep_types.append('InnerProduct')

        layers = [l for l in self.layers.keys() if self.layers[l].type in keep_types]
        layers = [l for l in layers if self.layers[l].phase in phases]

        remove_list = []

        for layer in layers:
            if self.layers[layer].type == 'Concat' and 'inception' not in layer:
                remove_list.append(layer)
                continue
            if not include_1x1 and self.layers[layer].type == 'Convolution':
                h = self.layers[layer].layerParams['kernel_h']
                w = self.layers[layer].layerParams['kernel_w']
                if h == 1 and w == 1:
                    remove_list.append(layer)
            if inception_unit and 'inception' in layer and self.layers[layer].type != 'Concat':
                remove_list.append(layer)

        layers = [l for l in layers if l not in remove_list]

        stats = []

        for l in layers:
            d = self.get_layer_depth(l, weightsOnly=True, phases=phases,
                                     inception_unit=inception_unit,
                                     include_pooling=include_pooling)[0]
            features = self.layers[l].layerParams['num_output']
            if d == 0:
                stats.append((0, l))
            else:
                stats.append((features / d, l))

        if key == 'MAX':
            if not stats:
                return np.NaN, None
            return_value = max([stat[0] for stat in stats])
            return_list = [stat[1] for stat in stats if stat[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not stats:
                return np.NaN, None
            return_value = min([stat[0] for stat in stats])
            return_list = [stat[1] for stat in stats if stat[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not stats:
                return np.NaN
            return_value = np.mean([stat[0] for stat in stats])
            return return_value
        else:
            print('Invalid statistic key: ', key)
            return 0

    def ratio_features_to_kerDim(self, key='MAX', phases=None,
                                 include_1x1=True, dim='a'):
        """
        Uses key to return a statistical summary of layer's ratios of features
        to kernel dim 'dim'. Only convolutional layers are considered. By default, 
        inception modules are treated as single layers.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_1x1:
                bool; whether or not to consider convolutional layers
                with 1x1 kernels. Default: True
            
            dim:
                one of 'a', 'h', or 'w'; specifies whether to compare features
                to kernel area, height, or width; Default: 'a'
            
        Returns:
            float stat, [list of str optimal_layers; used when key = 'MAX' or 'MIN']
            
        """

        # Get all layers with features, subject to the constraints of 'phases'
        # 'include_IP', 'inception_unit', and 'include_1x1'

        if phases is None:
            phases = ['ALL']
        keep_types = ['Convolution']

        layers = [l for l in self.layers.keys() if self.layers[l].type in keep_types]
        layers = [l for l in layers if self.layers[l].phase in phases]

        remove_list = []

        for layer in layers:
            if not include_1x1:
                h = self.layers[layer].layerParams['kernel_h']
                w = self.layers[layer].layerParams['kernel_w']
                if h == 1 and w == 1:
                    remove_list.append(layer)

        layers = [l for l in layers if l not in remove_list]

        stats = []

        for l in layers:
            kh = self.layers[l].layerParams['kernel_h']
            kw = self.layers[l].layerParams['kernel_w']
            ka = kh * kw

            if dim == 'a':
                kernel_dim = ka
            elif dim == 'h':
                kernel_dim = kh
            elif dim == 'w':
                kernel_dim = kw
            else:
                print('Invalid kernel dimension: ', dim)
                return 0

            features = self.layers[l].layerParams['num_output']
            if kernel_dim == 0:
                stats.append((0, l))
            else:
                stats.append((features / kernel_dim, l))

        if key == 'MAX':
            if not stats:
                return np.NaN, None
            return_value = max([stat[0] for stat in stats])
            return_list = [stat[1] for stat in stats if stat[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not stats:
                return np.NaN, None
            return_value = min([stat[0] for stat in stats])
            return_list = [stat[1] for stat in stats if stat[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not stats:
                return np.NaN
            return_value = np.mean([stat[0] for stat in stats])
            return return_value
        else:
            print('Invalid statistic key: ', key)
            return 0

    def ratio_kerDim_to_depth(self, key='MAX', phases=None,
                              include_pooling=False, include_1x1=True,
                              dim='a'):
        """
        Uses key to return a statistical summary of layer's ratios of features
        to depth. Only layers with weights are considered. By default, 
        inception modules are treated as single layers and inner product layers
        are considered.
        
        Parameters:
            key:
                str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
                statistic should be computed. Default: 'MAX'
            
            phases:
                list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_1x1:
                bool; whether or not to consider convolutional layers
                with 1x1 kernels. Default: True
            
            include_pooling:
                bool; whether or not to include pooling layers for
                computation of depth; Default: False
            
            dim:
                one of 'a', 'h', 'w': Determines whether to compare depth with
                kernel area, height, or width. Default: 'a'
            
        Returns:
            float stat, [list of str optimal_layers; used when key = 'MAX' or 'MIN']
            
        """

        # Get all layers with features, subject to the constraints of 'phases'
        # 'include_IP', 'inception_unit', and 'include_1x1'

        if phases is None:
            phases = ['ALL']
        keep_types = ['Convolution']

        layers = [l for l in self.layers.keys() if self.layers[l].type in keep_types]
        layers = [l for l in layers if self.layers[l].phase in phases]

        remove_list = []

        for layer in layers:
            if not include_1x1 and self.layers[layer].type == 'Convolution':
                h = self.layers[layer].layerParams['kernel_h']
                w = self.layers[layer].layerParams['kernel_w']
                if h == 1 and w == 1:
                    remove_list.append(layer)

        layers = [l for l in layers if l not in remove_list]

        stats = []

        for l in layers:
            d = self.get_layer_depth(l, weightsOnly=True, phases=phases,
                                     inception_unit=False,
                                     include_pooling=include_pooling)[0]

            kh = self.layers[l].layerParams['kernel_h']
            kw = self.layers[l].layerParams['kernel_w']
            ka = kh * kw

            if dim == 'a':
                kernel_dim = ka
            elif dim == 'h':
                kernel_dim = kh
            elif dim == 'w':
                kernel_dim = kw
            else:
                print('Invalid kernel dimension: ', dim)
                return 0

            if d == 0:
                stats.append((0, l))
            else:
                stats.append((kernel_dim / d, l))

        if key == 'MAX':
            if not stats:
                return np.NaN, None
            return_value = max([stat[0] for stat in stats])
            return_list = [stat[1] for stat in stats if stat[0] == return_value]
            return return_value, return_list
        elif key == 'MIN':
            if not stats:
                return np.NaN, None
            return_value = min([stat[0] for stat in stats])
            return_list = [stat[1] for stat in stats if stat[0] == return_value]
            return return_value, return_list
        elif key == 'AVG':
            if not stats:
                return np.NaN
            return_value = np.mean([stat[0] for stat in stats])
            return return_value
        else:
            print('Invalid statistic key: ', key)
            return 0

    # %%

    ########### IMAGE HANDLER FUNCTIONS ##################

    def nonzero_activations(self, ip_layer):
        """
        Returns the number of pixels in the image which are greater than tol
        in absolute value.
        
        Parameters:
            
            ip_layer:
                str; name of input layer
                
        Returns:
            float prop
        """
        if 'points_2d' in self.layers[ip_layer].imgFeatures.keys():
            return len(self.layers[ip_layer].imgFeatures['points_2d'])
        else:
            return len(self.get_img_points2d(ip_layer))

    def horiz_spread(self, ip_layer):
        """
        Returns the column distance between right-most and left-most activations
        which are larger than tol.
        
        Parameters:
            ip_layer:
                str; name of input layer
                
        Returns:
            int horiz_spread
        """

        if 'points_2d' in self.layers[ip_layer].imgFeatures.keys():
            return max([p[1] for p in self.layers[ip_layer].imgFeatures['points_2d']]) - \
                   min([p[1] for p in self.layers[ip_layer].imgFeatures['points_2d']])
        else:
            pts = self.get_img_points2d(ip_layer)
            return max([p[1] for p in pts]) - min([p[1] for p in pts])

    def _vert_spread(self, ip_layer):
        """
        Returns the column distance between right-most and left-most activations
        which are larger than tol.

        Parameters:
            ip_layer:
                str; name of input layer

        Returns:
            int horiz_spread
        """

        if 'points_2d' in self.layers[ip_layer].imgFeatures.keys():
            return max([p[0] for p in self.layers[ip_layer].imgFeatures['points_2d']]) - \
                   min([p[0] for p in self.layers[ip_layer].imgFeatures['points_2d']])
        else:
            pts = self.get_img_points2d(ip_layer)
            return max([p[0] for p in pts]) - min([p[1] for p in pts])

    def horiz_sd(self, ip_layer):
        """
        Gives the standard deviation of column coordinates of nonzero activations.
        :param ip_layer: str; name of input layer
        :return: float; horiz_sd
        """
        if 'points_2d' in self.layers[ip_layer].imgFeatures.keys():
            points = self.layers[ip_layer].imgFeatures['points_2d']
        else:
            points = self.get_img_points2d(ip_layer)

        horiz_coords = [p[1] for p in points]

        return np.std(horiz_coords)

    def vert_sd(self, ip_layer):
        """
        Gives the standard deviation of row coordinates of nonzero activations.
        :param ip_layer: str; name of input layer
        :return: float; vert_sd
        """

        if 'points_2d' in self.layers[ip_layer].imgFeatures.keys():
            points = self.layers[ip_layer].imgFeatures['points_2d']
        else:
            points = self.get_img_points2d(ip_layer)

        vert_coords = [p[0] for p in points]

        return np.std(vert_coords)

    def get_img_distance(self, inputLayer, combine_channels=True, df='L2'):
        """
        Constructs distance matrix for use in building Rips complex from the image stored in inputLayer. The distance
        function is computed as D((i1,j1,a1),(i2,j2,a2)) = (1/(a1^2 + a2^2))*d((i1,j1),(i2,j2)),
        where d is the Euclidean distance in the plane.


        :param inputLayer: str; name of inputLayer
        :param combine_channels: bool; If True, distances will be computed using averaged channel activations, otherwise
            separate distance matrices will be returned, one for each channel
        :param df: the distance function used in computing pixel distances. Choose from one of the following:

            1) 'L1': the L1 norm in R^2
            2) 'L2': the L2 norm in R^2
            3) 'L22': the L2 norm in R^2, squared
            4) 'diffL2': the L2 norm in R^2, weighted by (difference of activations)
            5) 'diffL22': the L2 norm in R^2, squared, weighted by (difference of activations)
            6) 'L23d': the L2 norm in R^3
            7) 'sumL2': the L2 norm in R^2, weighted by 1/(sum of abs(activation)s)
            8) 'sumL22':the L2 norm in R^2, squared, weighted by 1/(sum of abs(activations)s)
        :return: Populates the imgFeatures attribute of inputLayer with a lower-sub-triangular matrix
            (list of lists of floats) for each input channel.
        """
        df_dict = {
            'L1': self._L1,
            'L2': self._L2,
            'L22': self._L22,
            'diffL2': self._diffL2,
            'diffL22': self._diffL22,
            'L23d': self._L23d,
            'sumL2': self._sumL2,
            'sumL22': self._sumL22
        }

        distances = []

        points_list = self.get_img_points(inputLayer, combine_channels)

        for points in points_list:
            new_distances = []
            for r in range(len(points)):
                current_row = []
                for c in range(r):
                    p = np.array(points[r])
                    q = np.array(points[c])
                    D = df_dict[df](p, q)
                    current_row.append(D)
                new_distances.append(current_row)
            distances.append(new_distances)

        return distances

    @staticmethod
    def _L1(p, q):
        """
        The L1 norm on R^2

        :param p: tuple of float (i,j,a)
        :param q: tuple of float (i,j,a)
        :return: L1(p,q)
        """
        return abs((p[:2] - q[:2])[0]) + abs((p[:2] - q[:2])[1])

    @staticmethod
    def _L2(p, q):
        """
        The L2 norm on R^2

        :param p: tuple of float (i,j,a)
        :param q: tuple of float (i,j,a)
        :return: L2(p,q)
        """
        return np.linalg.norm(p[:2] - q[:2])

    @staticmethod
    def _L22(p, q):
        """
        The L2 norm on R^2, squared

        :param p: tuple of float (i,j,a)
        :param q: tuple of float (i,j,a)
        :return: L2^2(p,q)
        """
        return np.linalg.norm((p[:2] - q[:2])) ** 2

    @staticmethod
    def _diffL2(p, q):
        """
        Returns the L2 norm weighted by abs(a_p - a_q)

        :param p: tuple of float (i,j,a)
        :param q: tuple of float (i,j,a)
        :return: L2(p,q)*abs(a_p - a_q)
        """
        return np.abs(p[2] - q[2]) * np.linalg.norm((p[:2] - q[:2]))

    @staticmethod
    def _diffL22(p, q):
        """
        Returns the L2 norm, squared, weighted by abs(a_p-a_q)
        :param p: tuple of float (i,j,a)
        :param q: tuple of float (i,j,a)
        :return: L2(p,q)^2 abs(a_p-a_q)
        """
        return np.abs(p[2] - q[2]) * np.linalg.norm((p[:2] - q[:2])) ** 2

    @staticmethod
    def _L23d(p, q):
        """
        Returns the L2 norm in R^3
        :param p: tuple of float (i,j,a)
        :param q: tuple of float (i,j,a)
        :return: L2(p,q)
        """
        return np.linalg.norm(p - q)

    @staticmethod
    def _sumL2(p, q):
        """
        Returns the L2 norm, weighted by 1/(sum of activations)
        :param p: tuple of float (i,j,a)
        :param q: tuple of float (i,j,a)
        :return: L2(p,q)/(a_p + a_q)
        """
        a_sum = p[2] + q[2]
        if a_sum < 0.0001:
            a_sum = 0.0001
        return np.linalg.norm(p[:2] - q[:2]) / a_sum

    @staticmethod
    def _sumL22(p, q):
        """
        Returns the L2 norm, squared, weighted by 1/(sum of activations)
        :param p: tuple of float (i,j,a)
        :param q: tuple of float (i,j,a)
        :return: L2(p,q)^2/(a_p + a_q)
        """
        a_sum = p[2] + q[2]
        if a_sum < 0.0001:
            a_sum = 0.0001
        return np.linalg.norm(p[:2] - q[:2]) ** 2 / a_sum

    def get_img_rips_cplx2d(self, inputLayer, combine_channels=True, threshold='10p', df='L2'):
        """
        Uses GUDHI in order to build Rips complex objects from the distance matrices stored in the layerParams of
        inputLayer.

        :param inputLayer: str; name of input layer

        :param combine_channels: bool; If True, distances will be computed using averaged channel activations, otherwise
            separate distance matrices will be returned, one for each channel
            
        :param threshold: str or int; if string, must be in form 'NUMp' where NUM is an integer between 0 and 100
            denoting the percentile of distances which will serve as the cutoff; if int, represents the raw
            max_edge_length. Default: '10p'

        :param df: the distance function used in computing pixel distances. Choose from one of the following:

            1) 'L1': the L1 norm in R^2
            2) 'L2': the L2 norm in R^2
            3) 'L22': the L2 norm in R^2, squared
            4) 'diffL2': the L2 norm in R^2, weighted by (difference of activations)
            5) 'diffL22': the L2 norm in R^2, squared, weighted by (difference of activations)
            6) 'L23d': the L2 norm in R^3
            7) 'sumL2': the L2 norm in R^2, weighted by 1/(sum of abs(activation)s)
            8) 'sumL22':the L2 norm in R^2, squared, weighted by 1/(sum of abs(activations)s)

        :return: GUDHI Rips complex object for each distance matrix
        """

        distances = self.get_img_distance(inputLayer, combine_channels, df)

        melList = []
        if type(threshold) == int:
            for _ in distances:
                melList.append(threshold)
        elif type(threshold) == str:
            p = int(threshold[:-1])
            for d in distances:
                dList = []
                for r in d:
                    for c in r:
                        dList.append(c)
                melList.append(np.percentile(dList, p))

        complexes = []

        for distance_matrix, mel in zip(distances, melList):
            complexes.append(
                gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=mel).create_simplex_tree(
                    max_dimension=2))

        return complexes

    def set_img_rips_cplx2d(self, inputLayer, combine_channels=False, threshold='10p', df='L2'):
        """
        Uses GUDHI in order to build Rips complex objects from the distance matrices stored in the layerParams of
        inputLayer.

        :param inputLayer: str; name of input layer

        :param combine_channels: bool; If True, distances will be computed using averaged channel activations, otherwise
            separate distance matrices will be returned, one for each channel

        :param threshold: str or int; if string, must be in form 'NUMp' where NUM is an integer between 0 and 100
            denoting the percentile of distances which will serve as the cutoff; if int, represents the raw
            max_edge_length. Default: '10p'

        :param df: the distance function used in computing pixel distances. Choose from one of the following:

            1) 'L1': the L1 norm in R^2
            2) 'L2': the L2 norm in R^2
            3) 'L22': the L2 norm in R^2, squared
            4) 'diffL2': the L2 norm in R^2, weighted by (difference of activations)
            5) 'diffL22': the L2 norm in R^2, squared, weighted by (difference of activations)
            6) 'L23d': the L2 norm in R^3
            7) 'sumL2': the L2 norm in R^2, weighted by 1/(sum of abs(activation)s)
            8) 'sumL22':the L2 norm in R^2, squared, weighted by 1/(sum of abs(activations)s)

        :return: GUDHI Rips complex object for each distance matrix
        """
        self.layers[inputLayer].imgFeatures['rips_complexes2d'] = self.get_img_rips_cplx2d(inputLayer, combine_channels,
                                                                                           threshold, df)

    def get_img_points(self, inputLayer, combine_channels=False):
        """
        Populates the "point_clouds" field of inputLayer's imgFeatures attribute with nonzero pixels

        :param inputLayer: str; name of input layer
        :param combine_channels: bool; whether activations should be averaged across channels
        :return: no return
        """
        global current_points
        img = self.layers[inputLayer].layerParams['output_grid']
        chan, row, col = img.shape

        points_list = []
        zero_points = []

        for c in range(chan):
            current_points = []
            indices = itertools.product(range(row), range(col))
            for i, j in indices:
                # If a pixel is 0 in energy channel, it is 0 in time.
                if (i, j) in zero_points:
                    continue
                if img[c, i, j] == np.float(0) and c == 0:
                    zero_points.append((i, j))
                    continue
                current_points.append((i, j, img[c, i, j]))
            points_list.append(current_points)
        if combine_channels:
            new_points = []
            for p in range(len(current_points)):
                new_point = np.mean([points_list[c][p][2] for c in range(chan)])
                new_points.append(points_list[0][p][:2] + (new_point,))
            points_list = [new_points]

        return points_list

    def get_img_points2d(self, inputLayer):
        """
        Returns the 2d-indices of nonzero image activations. NOTE: the return-type is a list of depth 2.
        :param inputLayer: str; name of inputLayer
        :return: a depth-2 list of indices of nonzero image activations
        """
        return [p[:2] for p in self.get_img_points(inputLayer)[0]]

    def set_img_points2d(self, inputLayer):
        """
        Returns the 2d-indices of nonzero image activations. NOTE: the return-type is a list of depth 2.
        :param inputLayer: str; name of inputLayer
        :return: None
        """
        self.layers[inputLayer].imgFeatures['points_2d'] = self.get_img_points2d(inputLayer)

    def get_img_alpha_cplx2d(self, inputLayer, alpha2='inf'):
        """
        Builds 2d alpha complex from the images in a given layer. Note: the 2d alpha complex does
        not make use of activation values, so each channel produces the same alpha complex.

        :param inputLayer: str; name of input layer
        :param alpha2: float or 'inf'; determines maximal radius for alpha complex construction; if set to 'inf',
            returns the Delaunay complex

        :return: the 2d alpha complex for inputLayer
        """

        indices = self.get_img_points2d(inputLayer)

        if alpha2 == 'inf':
            return gd.AlphaComplex(points=indices).create_simplex_tree()
        else:
            return gd.AlphaComplex(points=indices).create_simplex_tree(max_alpha_square=alpha2)

    def get_img_alpha_cplx3d(self, inputLayer, alpha2_list, combine_channels=False):
        """
        Uses pixel activations as z-coordinates to construct 3d alpha-complex.

        :param inputLayer: str; name of input layer
        :param alpha2_list: list of float of 'inf'; determines maximal radius for alpha complex construction; if set to
            'inf', returns the Delaunay complex.
        :param combine_channels: bool; whether or not to average activation values across all channels and return a
            single complex
        :return: a list of GUDHI alpha complexes, one for each channel
        """
        complexes = []

        points_list = self.get_img_points(inputLayer, combine_channels)

        for points, alpha2 in zip(points_list, alpha2_list):
            if alpha2 == 'inf':
                complexes.append(gd.AlphaComplex(points=points).create_simplex_tree())
            else:
                complexes.append(gd.AlphaComplex(points=points).create_simplex_tree(max_alpha_square=alpha2))

        return complexes

    def set_img_alpha_cplx2d(self, inputLayer, alpha2='inf'):
        """
        Sets the 'alpha_complex2d' field of inputLayers 'imgFeatures' attribute.

        :param inputLayer: str; name of input layer
        :param alpha2: float or 'inf'; determines maximal radius for alpha complex construction.
        """

        self.layers[inputLayer].imgFeatures['alpha_complex2d'] = self.get_img_alpha_cplx2d(inputLayer, alpha2)

    def set_img_alpha_cplx3d(self, inputLayer, alpha2_list, combine_channels=False):
        """
        Sets the 'alpha_complexes3d' field of inputLayers 'imgFeatures' attribute.

        :param inputLayer: str; name of input layer
        :param alpha2_list: list of float of 'inf'; determines maximal radius for alpha complex construction; if set to
            'inf', returns the Delaunay complex.
        :param combine_channels: bool; whether or not to average activation values across all channels and return a
            single complex
        """
        self.layers[inputLayer].imgFeatures['alpha_complexes3d'] = self.get_img_alpha_cplx3d(inputLayer, alpha2_list,
                                                                                             combine_channels)

    def get_img_witness_cplx2d(self, inputLayer, landmark_prop=0.25, alpha2=0):
        """
        Builds 2d (strong) witness complex from the images in a given layer. Will use 'get_n_farthest_points' to
        determine a landmark set with landmark_prop*num_points points. Optional alpha2 parameter determines the
        relaxation coefficient.

        :param inputLayer: str; name of input layer
        :param landmark_prop: float, between 0 and 1; the proportion of points to choose randomly for landmarks
            Default=0.25.
        :param alpha2: float; relaxation parameter. Default=0

        :return: the 2d strong witness complex for inputLayer
        """

        witnesses = self.get_img_points2d(inputLayer)

        landmarks = gd.pick_n_random_points(points=witnesses, nb_points=int(landmark_prop * len(witnesses)))

        return gd.EuclideanStrongWitnessComplex(witnesses=witnesses,
                                                landmarks=landmarks).create_simplex_tree(max_alpha_square=alpha2)

    def set_img_witness_cplx2d(self, inputLayer, landmark_prop=0.25, alpha2=0):
        """
        Assigns 2d (strong) witness complex to the 'witness_complex2d' field of the imgFeatures attribute of inputLayer.

        :param inputLayer: str; name of input layer
        :param landmark_prop: float, between 0 and 1; the proportion of points to choose randomly for landmarks
            Default=0.25.
        :param alpha2: float; relaxation parameter. Default=0
        """
        self.layers[inputLayer].imgFeatures['witness_complex2d'] = self.get_img_witness_cplx2d(inputLayer,
                                                                                               landmark_prop,
                                                                                               alpha2)

    def get_img_witness_cplx3d(self, inputLayer, landmark_prop=0.25, alpha2=0, combine_channels=False):
        """
        Returns a list of strong witness complexes for each channel in the input image.

        :param inputLayer: str; name of input layer
        :param landmark_prop: float, between 0 and 1; the proportion of points to choose randomly for landmarks
            Default = 0.25
        :param alpha2: float; relaxation parameter. Default = 0
        :param combine_channels: bool; whether or not to average activation values over all image channels and
            return a single image
        :return: a list of 3d witness complexes, one for each img channel if combine_channels is False; the list has
            length 1 if combine_channels is True
        """
        witnesses = self.get_img_points(inputLayer, combine_channels)

        landmarks = [gd.pick_n_random_points(points=p, nb_points=int(landmark_prop * len(witnesses))) for p in
                     witnesses]

        complexes = []

        for witness, landmark in zip(witnesses, landmarks):
            complexes.append(gd.EuclideanStrongWitnessComplex(witness, landmark).create_simplex_tree(alpha2))

        return complexes

    def set_img_witness_cplx3d(self, inputLayer, landmark_prop=0.25, alpha2=0, combine_channels=False):
        """
        Sets the list of strong witness complexes for each channel in the input image to the 'witness_complexes3d'
        field of the "imgFeatures" attribute of inputLayer.

        :param inputLayer: str; name of input layer
        :param landmark_prop: float, between 0 and 1; the proportion of points to choose randomly for landmarks
            Default = 0.25
        :param alpha2: float; relaxation parameter. Default = 0
        :param combine_channels: bool; whether or not to average activation values over all image channels and
            set a single image
        """
        self.layers[inputLayer].imgFeatures['witness_complexes3d'] = self.get_img_witness_cplx3d(inputLayer,
                                                                                                 landmark_prop,
                                                                                                 alpha2,
                                                                                                 combine_channels)

    def _get_gudhi_cplxes(self, inputLayer, cplx_type):
        """
        Retrieves GUDHI complexes of a given type.
        :param inputLayer: str; name of input layer
        :param cplx_type: one of 'alpha2d', 'witness2d', 'rips2d', 'alpha3d', or 'witness3d'
        :return: the list of GUDHI complexes of the desired types
        """

        if cplx_type == 'alpha2d':
            if 'alpha_complex2d' not in self.layers[inputLayer].imgFeatures.keys():
                cplxes = [self.get_img_alpha_cplx2d(inputLayer, self.get_min_connected_alpha22d(inputLayer))]
            else:
                cplxes = [self.layers[inputLayer].imgFeatures['alpha_complex2d']]
        elif cplx_type == 'rips2d':
            if 'rips_complexes2d' not in self.layers[inputLayer].imgFeatures.keys():
                cplxes = self.get_img_rips_cplx2d(inputLayer)
            else:
                cplxes = self.layers[inputLayer].imgFeatures['rips_complexes2d']
        elif cplx_type == 'witness2d':
            if 'witness_complex2d' not in self.layers[inputLayer].imgFeatures.keys():
                cplxes = self.get_img_witness_cplx2d(inputLayer)
            else:
                cplxes = [self.layers[inputLayer].imgFeatures['witness_complex2d']]
        elif cplx_type == 'alpha3d':
            if 'alpha_complexes3d' not in self.layers[inputLayer].imgFeatures.keys():
                cplxes = self.get_img_alpha_cplx3d(inputLayer, self.get_min_connected_alpha23d(inputLayer))
            else:
                cplxes = self.layers[inputLayer].imgFeatures['alpha_complexes3d']
        elif cplx_type == 'witness3d':
            if 'witness_complexes3d' not in self.layers[inputLayer].imgFeatures.keys():
                cplxes = self.get_img_witness_cplx3d(inputLayer)
            else:
                cplxes = self.layers[inputLayer].imgFeatures['witness_complexes3d']
        else:
            print('Invalid complex type: {0}.'.format(cplx_type))
            return None
        return cplxes

    def get_simplicial_complexes(self, inputLayer, cplx_type='alpha2d'):
        """
        Returns a list of SimplicialComplex objects built from the complexes of inputLayer of type cplx_type;
        also returns a list of Embedding objects. The constructed complexes must be 2d.

        :param inputLayer: str; name of input layer
        :param cplx_type: str; one of 'alpha2d', 'witness2d', 'rips2d', 'alpha3d', or 'witness3d'
        :return: a list of tuples (simplicial.SimplicialComplex,simplicial.Embedding)
        """
        if cplx_type in ['alpha2d', 'rips2d', 'witness2d']:
            dim = '2d'
        elif cplx_type in ['alpha3d', 'witness3d']:
            dim = '3d'
        else:
            print('Invalid complex type: {0}.'.format(cplx_type))
            return None
        cplxes = self._get_gudhi_cplxes(inputLayer, cplx_type)
        scs = []
        points_list = self.get_img_points(inputLayer)
        for cplx, points in zip(cplxes, points_list):
            sc = simp.SimplicialComplex()
            faces = [f[0] for f in cplx.get_filtration()]
            for f in faces:
                if len(f) == 1:
                    sc.addSimplexWithBasis(f, id=f[0], attr=points[f[0]][2])
                else:
                    sc.addSimplexWithBasis(f, id=tuple(f))
            if dim == '2d':
                sc_em = simp.Embedding(sc)
                counter = 0
                for p in points:
                    sc_em.positionSimplex(counter, (p[1], 126 - p[0]))
                    counter += 1
                scs.append((sc, sc_em))
            else:
                sc_em = simp.Embedding(sc, 3)
                counter = 0
                for p in points:
                    sc_em.positionSimplex(counter, (p[1], 126 - p[0], p[2]))
                    counter += 1
                scs.append((sc, sc_em))
        return scs

    def set_simplicial_complexes(self, inputLayer, cplx_type='alpha2d'):
        """
        Sets a list of SimplicialComplex objects built from the complexes of inputLayer of type cplx_type;
        also returns a list of Embedding objects. The constructed complexes must be 2d.

        :param inputLayer: str; name of input layer
        :param cplx_type: str; one of 'alpha2d', 'witness2d', 'rips2d', 'alpha3d', or 'witness3d'
        """
        key = cplx_type + '_Simplicial'

        self.layers[inputLayer].imgFeatures[key] = self.get_simplicial_complexes(inputLayer, cplx_type)

    def get_min_connected_alpha22d(self, inputLayer):
        """
        Retrieves the minimal alpha^2 so that the 2d alpha complex of inputLayer's nonzero activations is connected.

        :param inputLayer: str; name of input layer
        :return: float min_connected_alpha^2
        """
        if 'alpha_complex2d' in self.layers[inputLayer].imgFeatures.keys():
            ac2d = self.layers[inputLayer].imgFeatures['alpha_complex2d']
        else:
            ac2d = self.get_img_alpha_cplx2d(inputLayer)

        pers = ac2d.persistence(2)
        zero_pers = [p[1] for p in pers if p[0] == 0]
        min_connected_alpha2 = max([p[1] for p in zero_pers if p[1] < float('Inf')]) + 0.25

        return min_connected_alpha2

    def set_min_connected_alpha22d(self, inputLayer):
        """
        Sets the minimal alpha^2 so that the 2d alpha complex of inputLayer's nonzero activations is connected.

        :param inputLayer: str; name of input layer
        :return: None
        """
        self.layers[inputLayer].imgFeatures['min_connected_alpha22d'] = self.get_min_connected_alpha22d(inputLayer)

    def get_min_connected_alpha23d(self, inputLayer):
        """
                Retrieves the minimal alpha^2 so that the 3d alpha complex of inputLayer's nonzero activations
                is connected.

                :param inputLayer: str; name of input layer
                :return: float min_connected_alpha^2
                """
        if 'alpha_complex3d' in self.layers[inputLayer].imgFeatures.keys():
            ac3d = self.layers[inputLayer].imgFeatures['alpha_complex3d']
        else:
            ac3d = self.get_img_alpha_cplx3d(inputLayer,
                                             ['inf' for _ in range(self.layers[inputLayer].layerParams['channels'])])

        mc_alpha2_list = []
        for c in ac3d:
            pers = c.persistence(2)
            zero_pers = [p[1] for p in pers if p[0] == 0]
            min_connected_alpha2 = max([p[1] for p in zero_pers if p[1] < float('Inf')])
            mc_alpha2_list.append(min_connected_alpha2)

        return mc_alpha2_list

    def get_betti_numbers(self, inputLayer, cplx_type='alpha2d'):
        """
        Returns the Betti numbers of a given complex.

        :param inputLayer: str; name of input layer
        :param cplx_type: one of 'alpha2d', 'witness2d', 'rips2d', 'alpha3d', or 'witness3d'
        :return: a list of tuples of betti numbers (b0, b1, b2, ...) for each channel
        """

        if cplx_type == 'alpha2d':
            if 'min_connected_alpha22d' in self.layers[inputLayer].imgFeatures.keys():
                alpha2 = self.layers[inputLayer].imgFeatures['min_connected_alpha22d']
            else:
                alpha2 = self.get_min_connected_alpha22d(inputLayer)
            ac2d = self.get_img_alpha_cplx2d(inputLayer, alpha2=alpha2)
            pers = ac2d.persistence(2)
            return ac2d.betti_numbers()

        cplxes = self._get_gudhi_cplxes(inputLayer, cplx_type)

        for cplx in cplxes:
            cplx.persistence(2)

        return [cplx.betti_numbers() for cplx in cplxes]

    def get_num_persistent_components(self, inputLayer, cplx_type='alpha2d', pers_scaler=0.5):
        """
        Returns a list of numbers of components of complexes which persist longer than pers_scaler*max_persistence.
        Only works for alpha complexes.

        :param inputLayer: str; name of input layer
        :param cplx_type: one of 'alpha2d', 'alpha3d'
        :param pers_scaler: float, between 0 and 1; the multiplier of max_persistence to determine
        :return: int; the number of generators of persistent H_0 which survive longer than pers_scaler*max_persistence.
        """
        if cplx_type == 'alpha2d':
            cplxes = [self.layers[inputLayer].imgFeatures['alpha_complex2d']]
        else:
            cplxes = self._get_gudhi_cplxes(inputLayer, cplx_type)

        nums = []

        if cplx_type == 'alpha2d':
            max_pers = [self.get_min_connected_alpha22d(inputLayer)]
        elif cplx_type == 'alpha3d':
            max_pers = self.get_min_connected_alpha23d(inputLayer)
        else:
            print('Invalid complex type: {}.'.format(cplx_type))
            return None

        for cplx, mp in cplxes, max_pers:
            pers = cplx.persistence(2)
            zero_pers = [p[1] for p in pers if p[0] == 0]
            nums.append(len([p for p in zero_pers if p[1] - p[0] > pers_scaler * mp]))
        if cplx_type == 'alpha2d':
            return nums[0]
        return nums

    def get_num_persistent_holes(self, inputLayer, cplx_type='alpha2d', pers_scaler=0.5):
        """
        Returns a list of numbers of holes of complexes which persist longer than pers_scaler*max_persistence.
        Only works for alpha complexes.

        :param inputLayer: str; name of input layer
        :param cplx_type: one of 'alpha2d', 'alpha3d'
        :param pers_scaler: float, between 0 and 1; the multiplier of max_persistence to determine
        :return: int; the number of generators of persistent H_1 which survive longer than pers_scaler*max_persistence.
        """
        if cplx_type == 'alpha2d':
            cplxes = [self.layers[inputLayer].imgFeatures['alpha_complex2d']]
        else:
            cplxes = self._get_gudhi_cplxes(inputLayer, cplx_type)

        nums = []

        if cplx_type == 'alpha2d':
            max_pers = [self.layers[inputLayer].imgFeatures['min_connected_alpha22d']]
        elif cplx_type == 'alpha3d':
            max_pers = self.get_min_connected_alpha23d(inputLayer)
        else:
            print('Invalid complex type: {}.'.format(cplx_type))
            return None

        for cplx, mp in zip(cplxes, max_pers):
            pers = cplx.persistence(2)
            one_pers = [p[1] for p in pers if p[0] == 1]
            nums.append(len([p for p in one_pers if p[1] - p[0] > pers_scaler * mp]))
        if cplx_type == 'alpha2d':
            return nums[0]
        return nums

    def get_num_persistent_voids(self, inputLayer, pers_scaler=0.5):
        """
        Returns a list of numbers of voids of complexes which persist longer than pers_scaler*max_persistence.
        Only works for 3d alpha complexes.

        :param inputLayer: str; name of input layer
        :param pers_scaler: float, between 0 and 1; the multiplier of max_persistence to determine
        :return: int; the number of generators of persistent H_2 which survive longer than pers_scaler*max_persistence.
        """
        cplxes = self._get_gudhi_cplxes(inputLayer, 'alpha3d')

        nums = []

        max_pers = self.get_min_connected_alpha23d(inputLayer)

        for cplx, mp in zip(cplxes, max_pers):
            pers = cplx.persistence(2)
            two_pers = [p[1] for p in pers if p[0] == 2]
            nums.append(len([p for p in two_pers if p[1] - p[0] > pers_scaler * mp]))

        return nums

    def get_bottleneck(self, inputLayer1, inputLayer2, cplx_type):
        """
        Computes the bottleneck distances between persistence diagrams of complexes of type cplx_type between
        inputLayer1 and inputLayer2. A list of distances is returned, one for each channel.
        :param inputLayer1: str; name of first input layer
        :param inputLayer2: str; name of second input layer
        :param cplx_type: one of 'alpha2d', 'witness2d', 'rips2d', 'alpha3d', or 'witness3d'
        :return: list of floats; bottleneck distances for each channel
        """
        if cplx_type == 'alpha2d':
            cplxes1 = [self.layers[inputLayer1].imgFeatures['alpha_complex2d']]
            cplxes2 = [self.layers[inputLayer2].imgFeatures['alpha_complex2d']]
        else:
            cplxes1 = self._get_gudhi_cplxes(inputLayer1, cplx_type)
            cplxes2 = self._get_gudhi_cplxes(inputLayer2, cplx_type)

        bottlenecks = []

        for cplx1, cplx2 in zip(cplxes1, cplxes2):
            diag1 = [p[1] for p in cplx1.persistence(2)]
            diag2 = [p[1] for p in cplx2.persistence(2)]

            bottlenecks.append(gd.bottleneck_distance(diag1, diag2))
        if cplx_type == 'alpha2d':
            return bottlenecks[0]

        return bottlenecks

    def get_delaunay_edges(self, inputLayer):
        """
        Returns the number of edges in the Delaunay complex of inputLayer.

        :param inputLayer: str; the layer name.
        :return: int num_edges
        """
        points = self.layers[inputLayer].imgFeatures['points_2d']
        skel1 = self.layers[inputLayer].imgFeatures['alpha_complex2d'].get_skeleton(1)
        return len(skel1)-len(points)

    def get_alpha_edges(self, inputLayer):
        """
        Returns the number of edges in the minimal-alpha complex of inputLayer.
        :param inputLayer: str; the layer name.
        :return: int num_edges
        """
        points = self.layers[inputLayer].imgFeatures['points_2d']
        ac2d = gd.AlphaComplex(points, self.layers[inputLayer].imgFeatures['min_connected_alpha22d']).create_simplex_tree()
        return len(ac2d.get_skeleton(1))-len(points)

    def get_EC(self,inputLayer):
        """
        Returns Euler characteristic of minimal connected alpha complex of inputLayer.

        :param inputLayer: str; the layer name.
        :return: int euler_characteristic
        """
        betti = self.get_betti_numbers(inputLayer)

        return betti[0]-betti[1]

    def inter_layer_bottleneck_avg(self):
        """
        Returns the average bottleneck distance of Delaunay complexes across all input layers
        :return: float inter_layer_bottleneck_avg
        """
        bottlenecks = []
        for ip in self.inputLayers:
            self.set_img_alpha_cplx2d(ip)
        for ip1,ip2 in itertools.combinations(self.inputLayers,2):
            pers1 = self.layers[ip1].imgFeatures['alpha_complex2d'].persistence(2)
            pers2 = self.layers[ip2].imgFeatures['alpha_complex2d'].persistence(2)

            diag1 = [p[1] for p in pers1]
            diag2 = [p[2] for p in pers2]

            bottlenecks.append(gd.bottleneck_distance(diag1, diag2))

        return np.mean(bottlenecks)

    def draw_alpha2d(self, inputLayer):
        """
        Produces Axes object for drawing the 2d alpha complex of input layer.

        :param inputLayer: str; name of input layer
        """

        if 'alpha2d_Simplicial' not in self.layers[inputLayer].imgFeatures.keys():
            print('Please generate Simplicial complex first.')
            return None

        sc, sc_em = self.layers[inputLayer].imgFeatures['alpha2d_Simplicial'][0]

        points2d = self.get_img_points2d(inputLayer)

        xmin = min([p[1] for p in points2d])
        xmax = max([p[1] for p in points2d])

        ymin = min([126 - p[0] for p in points2d])
        ymax = max([126 - p[0] for p in points2d])

        simp.draw_complex(sc, sc_em, xlims=[xmin - 10, xmax + 10], ylims=[ymin - 10, ymax + 10])

    def draw_alpha3d(self, inputLayer, channel=0):
        """
        Produces Axes3D objects for drawing the 3d alpha complex of input layer.

        :param inputLayer: str; name of input layer
        :param channel: the channel whose alpha complex should be drawn; Default = 0
        """
        if 'alpha3d_Simplicial' not in self.layers[inputLayer].imgFeatures.keys():
            print('Please generate Simplicial complex first.')
            return None

        sc, sc_em = self.layers[inputLayer].imgFeatures['alpha3d_Simplicial'][channel]

        points = self.get_img_points(inputLayer)[channel]

        f = plt.figure(figsize=(8, 6))

        xmin = min([p[1] for p in points])
        xmax = max([p[1] for p in points])

        ymin = min([126 - p[0] for p in points])
        ymax = max([126 - p[0] for p in points])

        zmin = min([p[2] for p in points])
        zmax = max([p[2] for p in points])

        ax = Axes3D(f)

        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(ymin, ymax)
        ax.set_zlim3d(zmin, zmax)

        vertices = [sc_em[s] for s in sc.simplicesOfOrder(0)]

        xs, ys, zs = [v[0] for v in vertices], [v[1] for v in vertices], [v[2] for v in vertices]

        ax.scatter(xs, ys, zs, c='b')

        for s in sc.simplicesOfOrder(1):
            x0, y0, z0 = sc_em[s[0]]
            x1, y1, z1 = sc_em[s[1]]

            edge = Line3DCollection([[(x0, y0, z0), (x1, y1, z1)]])
            edge.set_color('k')
            ax.add_collection3d(edge)

        for s in sc.simplicesOfOrder(2):
            x0, y0, z0 = sc_em[s[0]]
            x1, y1, z1 = sc_em[s[1]]
            x2, y2, z2 = sc_em[s[2]]

            tri = Poly3DCollection([[(x0, y0, z0), (x1, y1, z1), (x2, y2, z2)]])
            tri.set_alpha(0.5)
            tri.set_edgecolor('k')
            tri.set_facecolor('r')
            ax.add_collection3d(tri)

        for s in sc.simplicesOfOrder(3):
            x0, y0, z0 = sc_em[s[0]]
            x1, y1, z1 = sc_em[s[1]]
            x2, y2, z2 = sc_em[s[2]]
            x3, y3, z3 = sc_em[s[3]]

            tetra = Poly3DCollection([[(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]])
            tetra.set_alpha(0.5)
            tetra.set_edgecolor('k')
            tetra.set_facecolor('g')
            ax.add_collection3d(tetra)

    # %%

    # ###########SCORING FUNCTIONS##########

    def first_last_bottleneck(self, layer_path):
        """
        Returns the bottleneck distance between the alpha complexes corresponding to layer_path[0], and the flow
        of layer_path[0] to the final layer in layer_path. Note: it is assumed that an image is already loaded into
        the output grid of layer_path[0]. The alpha complex at each stage is generated by sampling points from all
        possible points in the flow in such a way that the spread of the sample is maximal. The first point chosen in
        this sampling process is always random. Also assumes that the input layer has been pre-loaded with its
        "points2d" field. The initial sample of points chooses 10% of the input pixels. Subsequent samplings choose
        enough points to maintain the ratio of nonzero activations to total pixels, and they account for the dilation
        in pixel size.

        :param layer_path: a list of strings which are layer names
        :return: float; bottleneck_distance
        """

        ip_layer = layer_path[0]
        A = self.layers[ip_layer].imgFeatures['points_2d']
        A = gd.choose_n_farthest_points(A, nb_points=int(np.ceil(0.1 * len(A))))

        ip_alpha = gd.AlphaComplex(points=A).create_simplex_tree()

        for op_layer in layer_path[1:]:
            flow = self.get_flow(A, ip_layer, [ip_layer, op_layer])[0][1]
            if flow is None:
                return np.NaN

            ip_shape = self.layers[ip_layer].layerParams['output_grid'][0].shape
            op_shape = self.layers[op_layer].layerParams['output_grid'][0].shape
            ip_area = ip_shape[0] * ip_shape[1]
            op_area = op_shape[0] * op_shape[1]
            k_w = self.layers[op_layer].layerParams['kernel_w']
            k_h = self.layers[op_layer].layerParams['kernel_h']
            k_a = k_w * k_h
            num_samples = max([int((op_area / ip_area) * len(flow) / k_a), 1])
            for p in flow:
                self.layers[op_layer].layerParams['output_grid'][0, p[0], p[1]] = 1
            A = gd.choose_n_farthest_points(flow, nb_points=num_samples)
            ip_layer = op_layer

        op_alpha = gd.AlphaComplex(points=A).create_simplex_tree()

        ip_pers = ip_alpha.persistence(2)
        op_pers = op_alpha.persistence(2)

        ip_diag = [p[1] for p in ip_pers]
        op_diag = [p[1] for p in op_pers]

        return gd.bottleneck_distance(ip_diag, op_diag)

    def avg_consecutive_bottleneck(self, layer_path):
        """
        Computes the average bottleneck distance between consecutive layers in layer_path.

        :param layer_path: list of layer names (strings)
        :return: float average bottleneck distance over layer_path
        """
        bottlenecks = []

        ip_layer = layer_path[0]
        A = self.layers[ip_layer].imgFeatures['points_2d']
        A = gd.choose_n_farthest_points(A, nb_points=int(np.ceil(0.1 * len(A))))

        for op_layer in layer_path[1:]:
            flow = self.get_flow(A, ip_layer, [ip_layer, op_layer])[0][1]
            if flow is None:
                return np.NaN
            ip_alpha = gd.AlphaComplex(points=A).create_simplex_tree()

            ip_shape = self.layers[ip_layer].layerParams['output_grid'][0].shape
            op_shape = self.layers[op_layer].layerParams['output_grid'][0].shape
            ip_area = ip_shape[0] * ip_shape[1]
            op_area = op_shape[0] * op_shape[1]
            k_w = self.layers[op_layer].layerParams['kernel_w']
            k_h = self.layers[op_layer].layerParams['kernel_h']
            k_a = k_w * k_h
            num_samples = max([int((op_area / ip_area) * len(flow) / k_a), 1])
            for p in flow:
                self.layers[op_layer].layerParams['output_grid'][0, p[0], p[1]] = 1
            A = gd.choose_n_farthest_points(flow, nb_points=num_samples)
            op_alpha = gd.AlphaComplex(points=A).create_simplex_tree()
            ip_pers = ip_alpha.persistence(2)
            op_pers = op_alpha.persistence(2)

            ip_diag = [p[1] for p in ip_pers]
            op_diag = [p[1] for p in op_pers]

            bottlenecks.append(gd.bottleneck_distance(ip_diag, op_diag))

            ip_layer = op_layer

        return np.mean(bottlenecks)

    def total_bottleneck_variation(self, layer_path):
        """
        Returns the total variation in bottleneck distance between successive layers in layer_path. Total variation
        is defined as \Sigma_{l=1}^{N-1}|bd_{L+1}-bd_{l}|, where l is the layer index and N is the number of layers
        in layer_path.

        :param layer_path: list of strings
        :return: float average change in bottleneck distances
        """
        bottlenecks = []

        ip_layer = layer_path[0]
        A = self.layers[ip_layer].imgFeatures['points_2d']
        A = gd.choose_n_farthest_points(A, nb_points=int(np.ceil(0.1 * len(A))))

        for op_layer in layer_path[1:]:
            flow = self.get_flow(A, ip_layer, [ip_layer, op_layer])[0][1]
            if flow is None:
                return np.NaN
            ip_alpha = gd.AlphaComplex(points=A).create_simplex_tree()

            ip_shape = self.layers[ip_layer].layerParams['output_grid'][0].shape
            op_shape = self.layers[op_layer].layerParams['output_grid'][0].shape
            ip_area = ip_shape[0] * ip_shape[1]
            op_area = op_shape[0] * op_shape[1]
            k_w = self.layers[op_layer].layerParams['kernel_w']
            k_h = self.layers[op_layer].layerParams['kernel_h']
            k_a = k_w * k_h
            num_samples = max([int((op_area / ip_area) * len(flow) / k_a), 1])
            for p in flow:
                self.layers[op_layer].layerParams['output_grid'][0, p[0], p[1]] = 1
            A = gd.choose_n_farthest_points(flow, nb_points=num_samples)
            op_alpha = gd.AlphaComplex(points=A).create_simplex_tree()
            ip_pers = ip_alpha.persistence(2)
            op_pers = op_alpha.persistence(2)

            ip_diag = [p[1] for p in ip_pers]
            op_diag = [p[1] for p in op_pers]

            bottlenecks.append(gd.bottleneck_distance(ip_diag, op_diag))

            ip_layer = op_layer

        start = bottlenecks[0]
        changes = []
        for end in bottlenecks[1:]:
            changes.append(abs(end - start))
            start = end
        return np.mean(changes)

    def first_layer_bottleneck(self, ip_layer):
        """
        Returns the bottleneck distance between the given input_layer and the first pooling or convolutional layer which
        acts upon it.

        :param ip_layer: str, a layer name
        :return: float bottleneck distance
        """
        A = self.layers[ip_layer].imgFeatures['points_2d']
        op_layer = self.layers[ip_layer].top[0]
        if self.layers[op_layer].type not in ['Convolution', 'Pooling']:
            return np.NaN
        ip_alpha = gd.AlphaComplex(points=A).create_simplex_tree()
        A = self.get_flow(A, ip_layer, [ip_layer,op_layer])[0][1]
        op_alpha = gd.AlphaComplex(points=A).create_simplex_tree()

        ip_pers = ip_alpha.persistence(2)
        op_pers = op_alpha.persistence(2)

        ip_diag = [p[1] for p in ip_pers]
        op_diag = [p[1] for p in op_pers]

        return gd.bottleneck_distance(ip_diag, op_diag)

    def first_layer_min_alpha2(self, ip_layer):
        """
        Returns the minimal alpha2 required to produce a connected alpha-complex in the first layer following the given
        input layer.
        :param ip_layer: str; name of input layer
        :return: float min_alpha2
        """
        A = self.layers[ip_layer].imgFeatures['points_2d']
        op_layer = self.layers[ip_layer].top[0]
        if self.layers[op_layer].type not in ['Convolution', 'Pooling']:
            return np.NaN
        A = self.get_flow(A, ip_layer, [ip_layer, op_layer])[0][1]

        ac2d = gd.AlphaComplex(points = A).create_simplex_tree()
        pers = ac2d.persistence(2)
        zero_pers = [p[1] for p in pers if p[0] == 0]
        return max([p[1] for p in zero_pers if p[1] < float('Inf')]) + 0.25

    def first_layer_flow(self, ip_layer):
        """
        Returns the number of activations in the first layer after the given input layer which receive information
        from the input.
        :param ip_layer: str; the layer name
        :return: int num_activations
        """
        A = self.layers[ip_layer].imgFeatures['points_2d']
        op_layer = self.layers[ip_layer].top[0]
        if self.layers[op_layer].type not in ['Convolution', 'Pooling']:
            return np.NaN
        A = self.get_flow(A, ip_layer, [ip_layer, op_layer])[0][1]

        return len(A)


# %%

# #########UNION ITERATOR OBJECT#################


class UnionIter:
    """
    This is an iterator object to allow the user to iterate through a union
    of receptive fields in an activation grid.
    
    Input:
        
        fields:
            list of tuples of length 4 (i_1, j_1, i_2, j_2),
            where (i_1, j_1) is the upper-left corner of a field and (i_2,j_2) is
            the lower-right corner. (Note: in the degenerate case of a singleton
            set all of i_1, j_1, i_2, and j_2 are equal.)
        
    """

    def __init__(self, fields):
        self.fields = fields
        self.min_i1 = min(field[0] for field in self.fields)
        self.i = self.min_i1
        self.active_i = [field for field in self.fields if
                         field[0] == self.min_i1]
        self.list_j = [(field[1], field[3]) for field in self.active_i]
        self.min_j1 = min(field[0] for field in self.list_j)
        self.max_j2 = max(field[1] for field in self.list_j
                          if field[0] == self.min_j1)
        self.list_j = [field for field in self.list_j if field[1] >= self.max_j2]
        self.stop_i = [field[2] for field in self.active_i]

        self.j = self.min_j1

    def __iter__(self):
        return self

    def __next__(self):
        if not self.active_i:
            raise StopIteration
        else:
            return_i = self.i
            return_j = self.j

            # iterate j
            if self.j < self.max_j2:
                self.j = self.j + 1
            else:
                self.list_j.remove((self.min_j1, self.max_j2))
                if self.list_j:
                    self.min_j1 = min(field[0] for field in self.list_j)
                    self.max_j2 = max(field[1] for field in self.list_j
                                      if field[0] == self.min_j1)
                    self.list_j = [field for field in self.list_j if
                                   field[1] >= self.max_j2]
                    self.j = self.min_j1
                else:
                    # Need to iterate i
                    # Remove fields whose lower i are above current i
                    if self.i in self.stop_i:
                        self.active_i = [field for field in self.active_i if
                                         field[2] > self.i]
                        self.fields = [field for field in self.fields if
                                       field[2] > self.i]
                    if self.active_i:
                        self.i = self.i + 1
                        new_active_fields = [field for field in self.fields if
                                             field[0] == self.i]
                        if new_active_fields:
                            self.active_i = self.active_i + new_active_fields
                            self.stop_i = self.stop_i + [field[2] for field in
                                                         new_active_fields]
                        self.list_j = [(field[1], field[3]) for field in self.active_i]
                        self.min_j1 = min(field[0] for field in self.list_j)
                        self.max_j2 = max(field[1] for field in self.list_j
                                          if field[0] == self.min_j1)
                        self.list_j = [field for field in self.list_j if
                                       field[1] >= self.max_j2]
                        self.j = self.min_j1
                    else:
                        self.fields = [field for field in self.fields if
                                       field[0] > self.i]
                        if self.fields:
                            self.min_i1 = min(field[0] for field in self.fields)
                            self.active_i = [field for field in self.fields if
                                             field[0] == self.min_i1]
                            self.list_j = [(field[1], field[3]) for field in self.active_i]
                            self.min_j1 = min(field[0] for field in self.list_j)
                            self.max_j2 = max(field[1] for field in self.list_j
                                              if field[0] == self.min_j1)
                            self.list_j = [field for field in self.list_j if
                                           field[1] >= self.max_j2]
                            self.stop_i = [field[2] for field in self.active_i]

                            self.i = self.min_i1
                            self.j = self.min_j1

            return return_i, return_j

        # %%


# ############# LAYER OBJECT ######################


class Layer:
    """
    This is a layer container object. It converts Caffe's layer objects
    into a form more usable for the purposes of extracting network
    architecture.

    Parameters:
        caffeLayer:
            a Caffe layer protobuf message

        inputLayer:
            a dict constructed by _getInputInfo

        dummy:
            bool; indicates whether the layer is referenced in the caffeLayer.top attribute
                without being a layer in the parent caffeNetwork
    """

    def __init__(self, caffeLayer=None, inputLayer=None, dummy=False):
        # Non-layer-specific params: all layers have the following
        if inputLayer is None:
            inputLayer = {}
        if caffeLayer is None:
            caffeLayer = {}
        if not dummy:
            if caffeLayer:
                self.name = caffeLayer.name
                self.type = caffeLayer.type
                self.bottom = caffeLayer.bottom
                self.top = []
                self.imgFeatures = {}
                if caffeLayer.include:
                    if caffeLayer.include[0].phase == 0:
                        self.phase = 'TRAIN'
                    elif caffeLayer.include[0].phase == 1:
                        self.phase = 'TEST'
                else:
                    self.phase = 'ALL'
            elif inputLayer:
                self.name = inputLayer['name']
                self.type = inputLayer['type']
                self.bottom = inputLayer['bottom']
                self.top = inputLayer['top']
                self.imgFeatures = {}
                if inputLayer['include']:
                    if inputLayer['include'][0].phase == 0:
                        self.phase = 'TRAIN'
                    elif inputLayer['include'][0].phase == 1:
                        self.phase = 'TEST'
                else:
                    self.phase = 'ALL'
            else:
                print('Need initialization data.')

            # dict of type-specific layer parameters
            self.layerParams = self._getLayerParams(caffeLayer, inputLayer)
        else:
            self.name = inputLayer['name']
            self.type = inputLayer['type']
            self.top = inputLayer['top']
            self.bottom = inputLayer['bottom']
            self.layerParams = {}
            self.imgFeatures = {}
            self.phase = inputLayer['phase']

    def _getLayerParams(self, caffeLayer=None, inputLayer=None):
        """
        Method to extract layer-specific features
        """
        if inputLayer is None:
            inputLayer = {}
        if caffeLayer is None:
            caffeLayer = {}
        if (not caffeLayer) and (not inputLayer):
            print('Invalid initialization data in ' + self.name)
            return []
        if caffeLayer:
            layer_types = ['Data',
                           'Silence',
                           'Split',
                           'Concat',
                           'Slice',
                           'DummyData',
                           'Convolution',
                           'ReLU',
                           'Sigmoid',
                           'Pooling',
                           'Flatten',
                           'InnerProduct',
                           'Dropout',
                           'Accuracy',
                           'SoftmaxWithLoss',
                           'GradientScaler',
                           'SigmoidCrossEntropyLoss',
                           'LRN',
                           'Eltwise']
            param_types = {
                'Data': self._getDataParams,
                'Silence': self._getSilenceParams,
                'Split': self._getSplitParams,
                'Concat': self._getConcatParams,
                'Slice': self._getSliceParams,
                'DummyData': self._getDummyDataParams,
                'Convolution': self._getConvolutionParams,
                'ReLU': self._getReLUParams,
                'Sigmoid': self._getSigmoidParams,
                'Pooling': self._getPoolingParams,
                'Flatten': self._getFlattenParams,
                'InnerProduct': self._getInnerProductParams,
                'Dropout': self._getDropoutParams,
                'Accuracy': self._getAccuracyParams,
                'SoftmaxWithLoss': self._getSoftmaxWithLossParams,
                'GradientScaler': self._getGradientScalerParams,
                'SigmoidCrossEntropyLoss': self._getSigmoidCrossEntropyLossParams,
                'LRN': self._getLRNParams,
                'Eltwise': self._getEltwiseParams
            }
            if caffeLayer.type not in layer_types:
                print("Unexpected type: " + caffeLayer.type + " in layer " +
                      caffeLayer.name)
                return []
            else:
                return param_types[caffeLayer.type](caffeLayer)
        else:
            return self._getInputParams(inputLayer)

    # Methods to extract parameters from traditional caffeLayers
    @staticmethod
    def _getDataParams(caffeLayer):
        """
        This extracts only parameters in Caffe's "Data" layer type. More 
        advanced parameter extraction, such as grid dimensions, will be done
        by the network itself.
        """
        return {
            'source': caffeLayer.data_param.source,
            'batch_size': caffeLayer.data_param.batch_size,
            'backend': caffeLayer.data_param.backend
        }

    @staticmethod
    def _getSilenceParams(caffeLayer):
        if caffeLayer:
            pass
        return {}

    @staticmethod
    def _getSplitParams(caffeLayer):
        if caffeLayer:
            pass
        return {}

    @staticmethod
    def _getConcatParams(caffeLayer):
        return {'axis': caffeLayer.concat_param.axis - 1}

    @staticmethod
    def _getSliceParams(caffeLayer):
        return {
            'axis': caffeLayer.slice_param.axis,
            'slice_point': caffeLayer.slice_param.slice_point
        }

    @staticmethod
    def _getDummyDataParams(caffeLayer):
        if caffeLayer.dummy_data_param.shape:
            dims = caffeLayer.dummy_data_param.shape[0].dim
            num = dims[0]
            channels = dims[1]
            height = dims[2]
            width = dims[3]
        else:
            num = caffeLayer.dummy_data_param.num
            channels = caffeLayer.dummy_data_param.channels
            height = caffeLayer.dummy_data_param.height
            width = caffeLayer.dummy_data_param.width
        return {
            'data_filler': (caffeLayer.dummy_data_param.data_filler[0].type,
                            caffeLayer.dummy_data_param.data_filler[0].value),
            'num': num,
            'channels': channels,
            'height': height,
            'width': width
        }

    def _getConvolutionParams(self, caffeLayer):
        kernel = caffeLayer.convolution_param.kernel_size

        if kernel:
            kernel_h = kernel[0]
            kernel_w = kernel[0]
        else:
            kernel_h = caffeLayer.convolution_param.kernel_h
            kernel_w = caffeLayer.convolution_param.kernel_w

        stride = caffeLayer.convolution_param.stride

        if stride:
            stride_h = stride[0]
            stride_w = stride[0]
        else:
            stride_h = caffeLayer.convolution_param.stride_h
            stride_w = caffeLayer.convolution_param.stride_w
            if stride_h == 0:
                stride_h = 1
            if stride_w == 0:
                stride_w = 1

        pad = caffeLayer.convolution_param.pad

        if pad:
            pad_h = pad[0]
            pad_w = pad[0]
        else:
            pad_h = caffeLayer.convolution_param.pad_h
            pad_w = caffeLayer.convolution_param.pad_w
        return {
            'num_output': caffeLayer.convolution_param.num_output,
            'bias_term': caffeLayer.convolution_param.bias_term,
            'dilation': caffeLayer.convolution_param.dilation,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'kernel_h': kernel_h,
            'kernel_w': kernel_w,
            'stride_h': stride_h,
            'stride_w': stride_w,
            'input_grid': None,
            'output_grid': None,
            'nonlinearity': None,
            'features': self._getFeatures(caffeLayer),
            'biases': np.zeros(caffeLayer.convolution_param.num_output)
        }

    @staticmethod
    def _getReLUParams(caffeLayer):
        if caffeLayer:
            pass
        return {}

    @staticmethod
    def _getSigmoidParams(caffeLayer):
        if caffeLayer:
            pass
        return {}

    @staticmethod
    def _getPoolingParams(caffeLayer):
        kernel = caffeLayer.pooling_param.kernel_size

        if kernel:
            kernel_h = kernel
            kernel_w = kernel
        else:
            kernel_h = caffeLayer.pooling_param.kernel_h
            kernel_w = caffeLayer.pooling_param.kernel_w

        stride = caffeLayer.pooling_param.stride

        if stride:
            stride_h = stride
            stride_w = stride
        else:
            stride_h = caffeLayer.pooling_param.stride_h
            stride_w = caffeLayer.pooling_param.stride_w

        pad = caffeLayer.pooling_param.pad

        if pad:
            pad_h = pad
            pad_w = pad
        else:
            pad_h = caffeLayer.pooling_param.pad_h
            pad_w = caffeLayer.pooling_param.pad_w
        return {
            'pool': caffeLayer.pooling_param.pool,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'kernel_h': kernel_h,
            'kernel_w': kernel_w,
            'stride_h': stride_h,
            'stride_w': stride_w,
            'num_output': None,
            'input_grid': None,
            'output_grid': None
        }

    @staticmethod
    def _getFlattenParams(caffeLayer):
        return {
            'axis': caffeLayer.flatten_param.axis - 1,
            'end_axis': caffeLayer.flatten_param.end_axis - 1,
            'input_grid': None,
            'num_output': None
        }

    @staticmethod
    def _getInnerProductParams(caffeLayer):
        return {
            'num_output': caffeLayer.inner_product_param.num_output,
            'num_input': None,
            'bias_term': caffeLayer.inner_product_param.bias_term,
            'weights': None,
            'biases': np.zeros(caffeLayer.inner_product_param.num_output),
            'nonlinearity': None
        }

    @staticmethod
    def _getDropoutParams(caffeLayer):
        return {
            'dropout_ratio': caffeLayer.dropout_param.dropout_ratio
        }

    @staticmethod
    def _getAccuracyParams(caffeLayer):
        return {
            'top_k': caffeLayer.accuracy_param.top_k,
            'axis': caffeLayer.accuracy_param.ignore_label,
            'ignore_label': caffeLayer.accuracy_param.ignore_label
        }

    @staticmethod
    def _getSoftmaxWithLossParams(caffeLayer):
        return {
            'normalization': caffeLayer.loss_param.normalization,
            'normalize': caffeLayer.loss_param.normalize
        }

    @staticmethod
    def _getGradientScalerParams(caffeLayer):
        return {
            'lower_bound': caffeLayer.gradient_scaler_param.lower_bound,
            'upper_bound': caffeLayer.gradient_scaler_param.upper_bound,
            'alpha': caffeLayer.gradient_scaler_param.alpha,
            'max_iter': caffeLayer.gradient_scaler_param.max_iter
        }

    @staticmethod
    def _getSigmoidCrossEntropyLossParams(caffeLayer):
        return {
            'loss_weight': caffeLayer.loss_weight
        }

    @staticmethod
    def _getLRNParams(caffeLayer):
        if caffeLayer.lrn_param.norm_region == 0:
            norm_region = 'ACROSS_CHANNELS'
        else:
            norm_region = 'WITHIN_CHANNEL'
        return {
            'local_size': caffeLayer.lrn_param.local_size,
            'alpha': caffeLayer.lrn_param.alpha,
            'beta': caffeLayer.lrn_param.beta,
            'norm_region': norm_region,
            'k': caffeLayer.lrn_param.k,
            'num_output': None,
            'input_grid': None,
            'output_grid': None,
            'kernel_h': 1,
            'kernel_w': 1,
            'stride_h': 1,
            'stride_w': 1,
            'pad_h': 0,
            'pad_w': 0
        }

    @staticmethod
    def _getEltwiseParams(caffeLayer):
        if caffeLayer.eltwise_param.operation == 0:
            op = 'PROD'
        elif caffeLayer.eltwise_param.operation == 1:
            op = 'SUM'
        else:
            op = 'MAX'
        return {
            'operation': op,
            'coeff': caffeLayer.eltwise_param.coeff
        }

    # Function to extract params from
    @staticmethod
    def _getInputParams(inputLayer):
        # Expects inputLayer to be a dict; constructed by parent network
        return {
            'output_grid': np.zeros((inputLayer['channels'],
                                     inputLayer['height'],
                                     inputLayer['width'])),
            'channels': inputLayer['channels'],
            'height': inputLayer['height'],
            'width': inputLayer['width'],
            'num_output': inputLayer['channels']
        }

    @staticmethod
    def _getFeatures(caffeLayer):
        if caffeLayer.convolution_param.kernel_size:
            return np.zeros([caffeLayer.convolution_param.num_output] +
                            [caffeLayer.convolution_param.kernel_size[0]] +
                            [caffeLayer.convolution_param.kernel_size[0]])
        else:
            return np.zeros([caffeLayer.convolution_param.num_output,
                             caffeLayer.convolution_param.kernel_h,
                             caffeLayer.convolution_param.kernel_w])

    def print_layer(self):
        print("Name: ", self.name)
        print("Type: ", self.type)
        print("Top: ", self.top)
        print("Bottom: ", self.bottom)
        print("Phase: ", self.phase)
        if self.layerParams:
            for key in self.layerParams.keys():
                print(key, ': ', self.layerParams[key])
        else:
            print('No layer parameters loaded.')
        if self.type == 'Input':
            if self.imgFeatures:
                for key in self.imgFeatures.keys():
                    print(key, ': ', self.imgFeatures[key])
            else:
                print('No image features loaded.')
