
"""
Author: Jesse Hamer

This is a collection of classes meant for handling caffe layer objects for
the purposes of extracting network architecture information.

The classes are designed specifically to parse Caffe networks used to analyze
data from the MINERvA and NOvA experiments. They are not intended as general
Caffe network containers.
"""

import caffe_pb2
from google.protobuf import text_format
import numpy as np
import os

class Network:
    """
    This class is the main container for caffe network objects. It will 
    consist of several layer objects and methods for those objects.
    
    Currently, many methods are specific to networks designed for MINERvA
    data, and will need to be modified so as to accomodate more generic
    network structures. Specifically, an all-purpose method to build input
    layers from a given caffe network specification is needed.
    """
    
    
    #caffeNet is a path to a file containing a Caffe network protobuf
    def __init__(self,caffeNet, mode = 'minerva'):
        #List used to add nonlinearity information to neuron layers
        nonlinearity_list = ['ReLU', 'PReLU', 'ELU', 'Sigmoid', 'TanH', 
                    'Power', 'Exp', 'Log', 'BNLL', 'Threshold', 
                    'Bias', 'Scale']
        #List used to handle MINERvA-specific Caffe layers whose "top" is not
        #the same as the layer's name.
        minerva_dummy_list = ['target_label','target_label_planecode',
                              'source_data','label_planecode','data',
                              'Slice NodeX1', 'Slice NodeX2', 'bottleneck',
                              'discard_features', 'keep_features','dc_labels']
        nova_dummy_list = ['label','ipFinal']
        
        tmp_net = caffe_pb2.NetParameter()
        with open(caffeNet) as f:
            text_format.Merge(f.read(),tmp_net)
        self.name = tmp_net.name
        self.layers = {}
        for layer in tmp_net.layer:
            #Need special handler for inception concatenators in NOVA networks
            if layer.type == 'Concat' and 'inception' in layer.name:
                layer_name = layer.top[0]
            else:
                layer_name = layer.name
            self.layers[layer_name] = Layer(caffeLayer = layer)
            #Build dummy layers, if necessary:
            if layer.top:
                for top in layer.top:
                    if ((top in minerva_dummy_list) or (top in nova_dummy_list)) \
                    and (not top in self.layers.keys()):
                        dummyInfo = {'name':top,'type':'Placeholder', 'top':[], 
                                     'bottom':layer.name}
                        self.layers[top] = Layer(inputLayer = dummyInfo,
                                   dummy = True)
                        self.layers[layer.name].top.append(top)
            #Build input layers, if necessary.
            if layer.top:
                for top in layer.top:
                    if 'data0' in top:
                        inputInfo = self._getInputInfo(layer, top, mode)
                        self.layers[inputInfo['name']] = Layer(inputLayer = inputInfo)
            #Update the tops of other layers.
            for bottom in layer.bottom:
                self.layers[bottom].top.append(layer.name)
            #Handle concatenation, if necessary:
            if layer.type == 'Concat':
                self._concat_handler(layer_name,layer)
            #Update num_outputs of pooling or LRN layers.
            if layer.type in ['Pooling','LRN']:
                self.layers[layer.name].layerParams['num_output'] = \
                self.layers[layer.bottom[0]].layerParams['num_output']
            #Handle Flatten and IP layers.    
            if layer.type == 'Flatten':
                self.layers[layer.name].layerParams['input_grid'] = \
                self.layers[layer.bottom[0]].layerParams['input_grid']
                self.layers[layer.name].layerParams['num_output'] = \
                np.prod(self.layers[layer.name].layerParams['input_grid'].shape)
                
            if layer.type =='InnerProduct':
                self.layers[layer.name].layerParams['num_input'] = \
                self.layers[layer.bottom[0]].layerParams['num_output']
                
            #Add outputs to special layers.
            
            if layer.name in ['alias_to_bottleneck','slice_features',
                              'bottleneck_alias','grl']:
                self.layers[layer.name].layerParams['num_output'] = \
                self.layers[layer.bottom[0]].layerParams['num_output']
                for top in layer.top:
                    if top in ['bottleneck', 'keep_features']:
                        self.layers[top].layerParams['num_output'] = \
                        self.layers[layer.name].layerParams['num_output']
            
            #Now check to see if we have layers whose grid attributes need
            #updating.
            if layer.type in ['Pooling', 'Convolution','LRN']:
                #add input grid, and output grid
                self.layers[layer.name].layerParams['input_grid'],\
                self.layers[layer.name].layerParams['output_grid']=\
                    self.get_grids(self.layers[layer.bottom[0]],
                                   self.layers[layer.name])
            if layer.type in nonlinearity_list:
                self.layers[layer.bottom[0]].layerParams['nonlinearity'] =\
                    layer.type
        
        
    def _getInputInfo(self, caffeLayer, name, mode):
        #NOTE: THIS IS SPECIFIC TO MINERVA DATA. CODE MUST BE MADE MORE GENERAL
        #TO ACCOMODATE OTHER DATASETS.
        if mode == 'minerva':
            input_c = 2
            input_h = 127
            input_w = 47
            if name in ['data0_1', 'data0_2']:
                return {
                        'name':name,
                        'type':'Input',
                        'bottom':[caffeLayer.name],
                        'top':[],
                        'channels':input_c,
                        'height':input_h,
                        'width':input_w,
                        'include':caffeLayer.include
                        }
            else:
                return {
                        'name':name,
                        'type': 'Input',
                        'bottom':[caffeLayer.name],
                        'top':[],
                        'channels':input_c,
                        'height':input_h,
                        'width':2*input_w,
                        'include':caffeLayer.include
                        }
        elif mode == 'nova':
            return {
                    'name':name,
                    'type': 'Input',
                    'bottom':[caffeLayer.name],
                    'top':[],
                    'channels':1,
                    'height':100,
                    'width':80,
                    'include':caffeLayer.include
                    }
        else:
            print("Unexpected dataset: " + mode)
            return {}
        
    def _concat_handler(self, layer_name, caffe_layer):
        """
        Special handler for concatenation layers.
        """
        #In NOVA networks, need to change name of inception concatenators
        if 'inception' in self.layers[layer_name].name:
            self.layers[layer_name].name = layer_name
        
        
        axis = self.layers[layer_name].layerParams['axis']
        
        self.layers[layer_name].layerParams['input_grid'] = []
        
        arrays = []
        
        lengths = []
        
        for bottom in self.layers[layer_name].bottom:
            bottom_layer = self.layers[bottom]
            if bottom_layer.type in ['Convolution', 'Pooling']:
                input_grid = bottom_layer.layerParams['output_grid']
                arrays = arrays + [input_grid]
                self.layers[layer_name].layerParams['input_grid'].append(input_grid)
            if bottom_layer.type in ['Flatten']:
                lengths.append(bottom_layer.layerParams['num_output'])
        if arrays:
            self.layers[layer_name].layerParams['output_grid'] = np.concatenate(arrays,axis=axis)
            self.layers[layer_name].layerParams['num_output'] = \
                self.layers[layer_name].layerParams['output_grid'].shape[0]
        if lengths:
            self.layers[layer_name].layerParams['num_output'] = sum(lengths)
        
    def get_grids(self, input_layer, output_layer):
        
        """
        Computes the (padded) input activation grid and output activation grid
        of output_layer on input_layer.
        
        Parameters:
            input_layer: A Layer object.
            output_layer: A layer object.
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
            
        input_grid = np.zeros((ip_channels,ip_layer_grid_h + 2*pad_h,
                               ip_layer_grid_w +2*pad_w))
        
        
        
        op_grid_h = 1 + (ip_layer_grid_h + 2*pad_h - kernel_h)/stride_h
        op_grid_w = 1 + (ip_layer_grid_w + 2*pad_w - kernel_w)/stride_w
        
        #NOTE: ROUNDING MAY OCCUR-NEED TO IMPLEMENT CHECK TO ENSURE THAT THE
        #KERNEL FITS EVENLY INTO THE INPUT GRID.
        
        if not op_grid_h - int(op_grid_h) == 0:
            print("WARNING: KERNEL_H DOES NOT EVENLY DIVIDE INPUT_H in " +
                  output_layer.name)
            print(op_grid_h)
        if not op_grid_w - int(op_grid_w) == 0:
            print("WARNING: KERNEL_W DOES NOT EVENLY DIVIDE INPUT_W in " +
                  output_layer.name)
            print(op_grid_w)
        output_grid = np.zeros((op_channels, int(op_grid_h),
                                 int(op_grid_w)))
        
        return input_grid, output_grid
    
    def feed_image(self, img, mode = 'minerva'):
        """
        Takes an input image and feeds it into network's internal image
        containers. For MINERvA data, these are the layers data0_0, data0_1, 
        and data0_2. NOVA implementation still needed.
        
        Parameters:
            img: an image from either of the MINERvA or NOVA datasets
            
            mode: either 'minerva' or 'nova'; default 'minerva'. Indicates
                how images should be preprocessed
        """
        if mode == 'minerva':
            #Preprocess image into desired shape, if necessary
            #Assume image has shape (8, 127, 47)
            img0_X1 = img[0:2]
            img0_X2 = img[2:4]
            img0 = np.concatenate((img0_X1,img0_X2),axis = 2)
            img1 = img[4:6]
            img2 = img[6:8]
            self.layers['data0_0'].layerParams['output_grid'] = img0
            self.layers['data0_1'].layerParams['output_grid'] = img1
            self.layers['data0_2'].layerParams['output_grid'] = img2
            return 1
        else:
            print('Cannot yet handle images from the dataset' + mode)
            return None
    
    def get_layer(self,layer_name):
        """
        Returns layer with given name, if it exists.
        
        Parameters:
            layer_name: a string giving the name of the layer to retrieve
            
        Returns:
            a layer object
        """
        if not layer_name in self.layers.keys():
            print('Layer ' + layer_name + ' does not exist in ' + self.name)
            return 0
        else:
            return self.layers[layer_name]
    
    def intersect_fields(self, fields):
        """
        Returns the upper-left and lower-right corner points of the
        intersection of a set of rectangular fields.
        
        Parameters:
        
            fields: list of tuples of length 4: (i_1, j_1, i_2, j_2),
            where (i_1, j_1) is the upper-left corner of a field and (i_2,j_2) is
            the lower-right corner. (Note: in the degenerate case of a singleton
            set all of i_1, j_1, i_2, and j_2 are equal.)
        
        returns: a 4-tuple (i_1,j_1, i_2, j_2) denoting the upper-left and 
        lower-right corner points of a rectangular subgrid.
        """
        
        max_ulr = max([field[0] for field in fields])
        max_ulc = max([field[1] for field in fields])
        min_lrr = min([field[2] for field in fields])
        min_lrc = min([field[3] for field in fields])
        
        if (max_ulr <= min_lrr) and (max_ulc <= min_lrc):
            return (max_ulr, max_ulc, min_lrr, min_lrc)
        else:
            return None
        
        
    def union_fields(self, fields = [], points = []):
        """
        Returns an interator object representing the union of rectangular
        fields or points.
        
        Parameters:
        
            fields: list of tuples of length 4: (i_1, j_1, i_2, j_2),
            where (i_1, j_1) is the upper-left corner of a field and (i_2,j_2) is
            the lower-right corner. (Note: in the degenerate case of a singleton
            set all of i_1, j_1, i_2, and j_2 are equal.)
            
            points: a list of ordered pairs
        
        returns: a _Union_iter iterator object
        """
        if points:
            union_input = [point + point for point in points]
        else:
            union_input = fields
        return UnionIter(union_input)
                
        
        
        
    def get_flow(self, A, inputLayer, layerPath = [], D = 0):
        """
        Returns a list of activations of all layers of layerPath
        into which the activations of A flow. If D is specified, will
        return the flow of A into all child layers of inputLayer of depth <= D.
        
        Parameters:
            A: A list or tuple of ordered pairs denoting the activations whose
            flow is desired.
            inputLayer: The layer whose output grid contains A.
            layerPath: A list of layer names. The first element is taken to be
            inputLayer, and any consecutive layers should have a bottom->
            top relationship.
            D: The depth relative to layer1 of layers for which we would like
            to know the flow of A.
                
        Returns:
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
        
        if layerPath:
            for i in range(len(layerPath)-1):
                if self.layers[layerPath[i]].name not in self.layers[layerPath[i+1]].bottom:
                    print("Invalid path: " + layerPath[i] + "is not bottom of " +
                          layerPath[i+1])
                    return None
            return_list = []
            activations = A
            for i in range(len(layerPath)-1):
                bottom_layer = self.layers[layerPath[i]]
                top_layer = self.layers[layerPath[i+1]]
                
                flow = self._flow_once(activations, bottom_layer,top_layer)
                return_list.append((top_layer.name, flow))
                activations = [point for point in self.union_fields(flow)]
            
            return return_list
            
            
        elif D:
            return_list = []
            activations = A
            
            
            layer_path = self._get_layer_paths(inputLayer, D)
            
            for path in layer_path:
                return_list.append((path, self.get_flow(activations, 
                                                        inputLayer,
                                                        path)))
            return return_list
            
            
                
            
        else:
            return None
        
    def _get_layer_paths(self, inputLayer, D):
        
        """
        Auxiliary recursive funciton to get the list of all depth D layer 
        paths starting at inputLayer.
        
        """
        
        tops = [t for t in self.layers[inputLayer].top if self.layers[t].type in 
                ['Convolution','Pooling']]
        
        if D ==1:
            return [[inputLayer,t] for t in tops]
        else:
            return_list = []
            for t in tops:
                return_list = return_list + [[inputLayer] + path for path in 
                                             self._get_layer_paths(t, D-1)]
            return return_list
        
    
    def _flow_once(self, A, bottom, top):
        """
        Returns a list of fields, expressed as tuples (i1, j1, i2, j2), where
        (i1,j1) and (i2,j2) are the upper-left and lower-right corners of the
        rectangualar field. Each field is the flow of a particular activation
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
            
            i1 = int(np.ceil(max([0, (i + pad_h - kernel_h)/stride_h])))
            j1 = int(np.ceil(max([0, (j + pad_w - kernel_w)/stride_w])))
            i2 = int(np.floor((i + pad_h)/stride_h))
            j2 = int(np.floor((j + pad_w)/stride_w))
                    
            return_list.append([i1,j1,i2,j2])
        
        return return_list
    
        
    def get_ERF(self, A, layer_path):
        """
        Returns a list of tuples (layer_name, points), where points
        is a list of ordered pairs constituting the ERF of A in the layer
        layer_name, and A is a set of activations in the last element of
        layer_path.
        
        Parameters:
            A: a list of ordered pairs
            
            layer_path: a list of layer names, each of which feeds into the
            next
            
        Returns:
            list of tuples of length 2
        """
        
        D = len(layer_path)
        
        points = A
        
        return_list = []
        
        for d in range(1, D):
            
            top = self.layers[layer_path[D-d]]
            bottom = self.layers[layer_path[D-d-1]]
            
            fields = []
            
            kernel_h = top.layerParams['kernel_h']
            kernel_w = top.layerParams['kernel_w']
                
            stride_h = top.layerParams['stride_h']
            stride_w = top.layerParams['stride_w']
                
            pad_h = top.layerParams['pad_h']
            pad_w = top.layerParams['pad_w']
                
            r,c = bottom.layerParams['output_grid'][0].shape
            
            for point in points:
                i,j = point
                
                m,n = kernel_h,kernel_w
                
                
                i1,j1 = i*stride_h,j*stride_w
                
                if i1 < pad_h:
                    m = m-(pad_h-i1)
                    i1 = pad_h
                if j1 < pad_w:
                    n = n-(pad_w-j1)
                    j1 = pad_w
                
                i2 = i1 + m-1
                j2 = j1 + n-1
                
                if i2 > pad_h + r:
                    i2 = pad_h + r
                if j2 > pad_w + c:
                    j2 = pad_w + c
                    
                fields.append([i1,j1,i2,j2])
            union = self.union_fields(fields)
            
            
            points = [point for point in union]
            
            return_list.append((bottom.name,points))
            
        return return_list
        
        
    
    
class UnionIter:
    """
    This is an iterator object to allow the user to iterate through a union
    of receptive fields in an activation grid.
    
    Input:
        
        fields: list of tuples of length 4: (i_1, j_1, i_2, j_2),
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
        self.list_j = [(field[1],field[3]) for field in self.active_i]
        self.min_j1 = min(field[0] for field in self.list_j)
        self.max_j2 = max(field[1] for field in self.list_j 
                          if field[0]==self.min_j1)
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
            
            #iterate j
            if self.j < self.max_j2:
                self.j = self.j +1
            else:
                self.list_j.remove((self.min_j1,self.max_j2))
                if self.list_j:
                    self.min_j1 = min(field[0] for field in self.list_j)
                    self.max_j2 = max(field[1] for field in self.list_j 
                                      if field[0]==self.min_j1)
                    self.list_j = [field for field in self.list_j if 
                                   field[1] >= self.max_j2]
                    self.j = self.min_j1
                else:
                    #Need to iterate i
                    #Remove fields whose lower i are above current i
                    if self.i in self.stop_i:
                        self.active_i = [field for field in self.active_i if
                                         field[2] > self.i]
                        self.fields = [field for field in self.fields if
                                       field[2] > self.i]
                    if self.active_i:
                        self.i = self.i+1
                        new_active_fields = [field for field in self.fields if
                                             field[0] == self.i]
                        if new_active_fields:
                            self.active_i = self.active_i + new_active_fields
                            self.stop_i = self.stop_i + [field[2] for field in
                                                         new_active_fields]
                        self.list_j = [(field[1],field[3]) for field in self.active_i]
                        self.min_j1 = min(field[0] for field in self.list_j)
                        self.max_j2 = max(field[1] for field in self.list_j 
                                          if field[0]==self.min_j1)
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
                            self.list_j = [(field[1],field[3]) for field in self.active_i]
                            self.min_j1 = min(field[0] for field in self.list_j)
                            self.max_j2 = max(field[1] for field in self.list_j 
                                              if field[0]==self.min_j1)
                            self.list_j = [field for field in self.list_j if 
                                           field[1] >= self.max_j2]
                            self.stop_i = [field[2] for field in self.active_i]
                            
                            self.i = self.min_i1
                            self.j = self.min_j1
                            
            return (return_i, return_j)    
                    
                    
            
            
    
    
    
    
class Layer:
    """
    This is a layer container object. It converts Caffe's layer objects
    into a form more usable for the purposes of extracting network
    architecture.
    """
    def __init__(self, caffeLayer = {}, inputLayer = {}, dummy = False):
        #Non-layer-specific params: all layers have the following
        if not dummy:
            if caffeLayer:
                self.name = caffeLayer.name
                self.type = caffeLayer.type
                self.bottom = caffeLayer.bottom
                self.top = []
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
                if inputLayer['include']:
                    if inputLayer['include'][0].phase == 0:
                        self.phase = 'TRAIN'
                    elif inputLayer['include'][0].phase == 1:
                        self.phase = 'TEST'
                else:
                    self.phase = 'ALL'
            else:
                print('Need initialization data.')
                return None
                
            
            #dict of type-specific layer parameters
            self.layerParams = self._getLayerParams(caffeLayer,inputLayer)
        else:
            self.name = inputLayer['name']
            self.type = inputLayer['type']
            self.top = inputLayer['top']
            self.bottom = inputLayer['bottom']
            self.layerParams = {}
        
    def _getLayerParams(self,caffeLayer = {}, inputLayer = {}):
        """
        Method to extract layer-specific features
        """
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
                    'LRN':self._getLRNParams,
                    'Eltwise':self._getEltwiseParams
                    }
            if caffeLayer.type not in layer_types:
                print("Unexpected type: " + caffeLayer.type + " in layer " +
                      caffeLayer.name)
                return [] 
            else:
                return param_types[caffeLayer.type](caffeLayer)
        else:
            return self._getInputParams(inputLayer)
      
    #Methods to extract parameters from traditional caffeLayers    
    def _getDataParams(self, caffeLayer):
        """
        This extracts only parameters in Caffe's "Data" layer type. More 
        advanced parameter extraction, such as grid dimensions, will be done
        by the network itself.
        """
        return {
                'source':caffeLayer.data_param.source,
                'batch_size':caffeLayer.data_param.batch_size,
                'backend':caffeLayer.data_param.backend
                }
        
    def _getSilenceParams(self, caffeLayer):
        return {}
        
    def _getSplitParams(self,caffeLayer):
        return {}
        
    def _getConcatParams(self,caffeLayer):
        return {'axis':caffeLayer.concat_param.axis-1}
        
    def _getSliceParams(self, caffeLayer):
        return {
                'axis':caffeLayer.slice_param.axis,
                'slice_point':caffeLayer.slice_param.slice_point
                }
        
    def _getDummyDataParams(self,caffeLayer):
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
                'num':num,
                'channels':channels,
                'height':height,
                'width':width
                }
        
    def _getConvolutionParams(self,caffeLayer):
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
            if stride_h ==0:
                stride_h =1
            if stride_w ==0:
                stride_w =1
            
        pad = caffeLayer.convolution_param.pad
        
        if pad:
            pad_h = pad[0]
            pad_w = pad[0]
        else:
            pad_h = caffeLayer.convolution_param.pad_h
            pad_w = caffeLayer.convolution_param.pad_w
        return {
                'num_output':caffeLayer.convolution_param.num_output,
                'bias_term':caffeLayer.convolution_param.bias_term,
                'dilation':caffeLayer.convolution_param.dilation,
                'pad_h':pad_h,
                'pad_w':pad_w,
                'kernel_h':kernel_h,
                'kernel_w':kernel_w,
                'stride_h':stride_h,
                'stride_w':stride_w,
                'input_grid':None,
                'output_grid':None,
                'nonlinearity':None,
                'features': self._getFeatures(caffeLayer),
                'biases': np.zeros(caffeLayer.convolution_param.num_output)
                }
        
    def _getReLUParams(self,caffeLayer):
        return {}
        
    def _getSigmoidParams(self,caffeLayer):
        return {}
    def _getPoolingParams(self,caffeLayer):
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
                'pool':caffeLayer.pooling_param.pool,
                'pad_h':pad_h,
                'pad_w':pad_w,
                'kernel_h':kernel_h,
                'kernel_w':kernel_w,
                'stride_h':stride_h,
                'stride_w':stride_w,
                'num_output':None,
                'input_grid':None,
                'output_grid':None
                }
        
    def _getFlattenParams(self,caffeLayer):
        return {
                'axis':caffeLayer.flatten_param.axis-1,
                'end_axis':caffeLayer.flatten_param.end_axis-1,
                'input_grid':None,
                'num_output':None
                }
        
    def _getInnerProductParams(self, caffeLayer):
        return {
                'num_output':caffeLayer.inner_product_param.num_output,
                'num_input': None,
                'bias_term':caffeLayer.inner_product_param.bias_term,
                'weights':None,
                'biases':np.zeros(caffeLayer.inner_product_param.num_output),
                'nonlinearity':None
                }
        
    def _getDropoutParams(self, caffeLayer):
        return {
                'dropout_ratio':caffeLayer.dropout_param.dropout_ratio
                }
        
    def _getAccuracyParams(self,caffeLayer):
        return {
                'top_k':caffeLayer.accuracy_param.top_k,
                'axis':caffeLayer.accuracy_param.ignore_label,
                'ignore_label':caffeLayer.accuracy_param.ignore_label
                }
        
    def _getSoftmaxWithLossParams(self,caffeLayer):
        return {
                'normalization':caffeLayer.loss_param.normalization,
                'normalize':caffeLayer.loss_param.normalize
                }
        
    def _getGradientScalerParams(self, caffeLayer):
        return {
                'lower_bound':caffeLayer.gradient_scaler_param.lower_bound,
                'upper_bound':caffeLayer.gradient_scaler_param.upper_bound,
                'alpha':caffeLayer.gradient_scaler_param.alpha,
                'max_iter':caffeLayer.gradient_scaler_param.max_iter
                }
        
    def _getSigmoidCrossEntropyLossParams(self, caffeLayer):
        return {
                'loss_weight':caffeLayer.loss_weight
                }
        
    def _getLRNParams(self, caffeLayer):
        if caffeLayer.lrn_param.norm_region == 0:
            norm_region = 'ACROSS_CHANNELS'
        else:
            norm_region = 'WITHIN_CHANNEL'
        return {
                'local_size':caffeLayer.lrn_param.local_size,
                'alpha':caffeLayer.lrn_param.alpha,
                'beta':caffeLayer.lrn_param.beta,
                'norm_region':norm_region,
                'k':caffeLayer.lrn_param.k,
                'num_output':None,
                'input_grid':None,
                'output_grid':None,
                'kernel_h':1,
                'kernel_w':1,
                'stride_h':1,
                'stride_w':1,
                'pad_h':0,
                'pad_w':0
                }
        
    def _getEltwiseParams(self, caffeLayer):
        if caffeLayer.eltwise_param.operation ==0:
            op = 'PROD'
        elif caffeLayer.eltwise_param.operation == 1:
            op = 'SUM'
        else:
            op = 'MAX'
        return {
                'operation':op,
                'coeff':caffeLayer.eltwise_param.coeff
                }
        
        
    #Function to extract params from 
    def _getInputParams(self,inputLayer):
        #Expects inputLayer to be a dict; constructed by parent network
        return {
                'output_grid': np.zeros((inputLayer['channels'],
                                         inputLayer['height'],
                                         inputLayer['width'])),
                'channels':inputLayer['channels'],
                'height':inputLayer['height'],
                'width':inputLayer['width']
                }
        
    def _getFeatures(self,caffeLayer):
        if caffeLayer.convolution_param.kernel_size:
            return np.zeros([caffeLayer.convolution_param.num_output] +
                             [caffeLayer.convolution_param.kernel_size[0]] +
                             [caffeLayer.convolution_param.kernel_size[0]])
        else:
            return np.zeros([caffeLayer.convolution_param.num_output,
                             caffeLayer.convolution_param.kernel_h,
                             caffeLayer.convolution_param.kernel_w])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
