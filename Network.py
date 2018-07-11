
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
        nova_dummy_list = ['label']
        
        tmp_net = caffe_pb2.NetParameter()
        with open(caffeNet) as f:
            text_format.Merge(f.read(),tmp_net)
        self.name = tmp_net.name
        self.layers = {}
        for layer in tmp_net.layer:
            #Need special handler for inception concatenators in NOVA networks
            if layer.type == 'Concat' and 'inception' in layer.name:
                layer_name = layer.top[0]
            elif layer.name == 'finalip':
                layer_name = layer.top[0]
            else:
                layer_name = layer.name
            self.layers[layer_name] = Layer(caffeLayer = layer)
            if layer.name == 'finalip':
                self.layers[layer_name].name = layer_name
            #Build dummy layers, if necessary:
            if layer.top:
                for top in layer.top:
                    if ((top in minerva_dummy_list) or (top in nova_dummy_list)):
                        if not top in self.layers.keys():
                            if layer.include:
                                if layer.include[0].phase == 0:
                                    temp_phase = 'TRAIN'
                                elif layer.include[0].phase == 1:
                                    temp_phase = 'TEST'
                            else:
                                temp_phase = 'ALL'
                            dummyInfo = {'name':top,'type':'Placeholder', 'top':[], 
                                         'bottom':[layer_name], 'phase':temp_phase}
                            self.layers[top] = Layer(inputLayer = dummyInfo,
                                       dummy = True)
                            self.layers[layer_name].top.append(top)
                        else:
                            if layer.include:
                                if layer.include[0].phase != self.layers[top].phase:
                                    self.layers[top].phase = 'ALL'
                            else:
                                self.layers[top].phase = 'ALL'
                            self.layers[layer_name].top.append(top)
                            self.layers[top].bottom.append(layer_name)
            #Build input layers, if necessary.
            if layer.top:
                for top in layer.top:
                    if 'data0' in top:
                        inputInfo = self._getInputInfo(layer, top, mode)
                        self.layers[inputInfo['name']] = Layer(inputLayer = inputInfo)
            #Update the tops of other layers.
            for bottom in layer.bottom:
                self.layers[bottom].top.append(layer_name)
            #Handle concatenation, if necessary:
            if layer.type == 'Concat':
                self._concat_handler(layer_name,layer)
            #Update num_outputs of pooling or LRN layers.
            if layer.type in ['Pooling','LRN']:
                self.layers[layer_name].layerParams['num_output'] = \
                self.layers[layer.bottom[0]].layerParams['num_output']
            #Handle Flatten and IP layers.    
            if layer.type == 'Flatten':
                self.layers[layer_name].layerParams['input_grid'] = \
                self.layers[layer.bottom[0]].layerParams['input_grid']
                self.layers[layer_name].layerParams['num_output'] = \
                np.prod(self.layers[layer_name].layerParams['input_grid'].shape)
                
            if layer.type =='InnerProduct':
                self.layers[layer_name].layerParams['num_input'] = \
                self.layers[layer.bottom[0]].layerParams['num_output']
                
            #Add outputs to special layers.
            
            if layer_name in ['alias_to_bottleneck','slice_features',
                              'bottleneck_alias','grl']:
                self.layers[layer_name].layerParams['num_output'] = \
                self.layers[layer.bottom[0]].layerParams['num_output']
                for top in layer.top:
                    if top in ['bottleneck', 'keep_features']:
                        self.layers[top].layerParams['num_output'] = \
                        self.layers[layer_name].layerParams['num_output']
            
            #Now check to see if we have layers whose grid attributes need
            #updating.
            if layer.type in ['Pooling', 'Convolution','LRN']:
                #add input grid, and output grid
                self.layers[layer_name].layerParams['input_grid'],\
                self.layers[layer_name].layerParams['output_grid']=\
                    self.get_grids(self.layers[layer.bottom[0]],
                                   self.layers[layer_name])
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
        
        Inputs:
            inputLayer: string name of Layer
            
            D: int depth
            
        Returns:
            a list of lists of layer names
        
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
        
        
    def get_max_paths(self, inputLayer, weightsOnly = False, convOnly = False,
                      phases = ['ALL'], inception_unit = False, 
                      include_pooling = False):
        """
        Returns the maximal length path starting at inputLayer.
        
        Parameters:
            inputLayer: str; name of layer at which all desired paths begin
            
            weightsOnly: bool; whether or not to consider only layers which
            have weights/biases. Effectively reduces to considering only
            convolutional and inner product layers
            
            convOnly: bool; whether or not to consider only convolutional
            layers. If True, the path search will stop when a "Flatten" layer
            is met, as this signals the end of the convolutional segment of
            the network.
            
            phases: list; sublist of ['ALL', 'TEST', 'TRAIN']; indicates the 
            phases for which layers should be considered. Default is ['ALL'].
            
            inception_unit: bool; whether or not to treat an inception module
            as a single 'layer' (thus contributing only 1 to depth). Default:
            True
            
            include_pooling: bool; whether or not to add pooling layers to
            paths when weightsOnly or convOnly is set to True. Default: False
            
        Returns:
            a list of lists representing all paths from inputLayer to an
            output. The first entry in every path is always inputLayer, and
            the last entry is always a leaf node in the network.
        """
        ignore_list = ['ReLU', 'PReLU', 'ELU', 'Sigmoid', 'TanH', 
                    'Power', 'Exp', 'Log', 'BNLL', 'Threshold', 
                    'Bias', 'Scale', 'Dropout']
        #For use when weightsOnly == True or convOnly == True
        silence_list = ['Concat', 'Flatten', 'Pooling','Split', 'Slice',
                        'Placeholder', 'LRN']
        silence = False
        if self.layers[inputLayer].type =='Pooling':
            h = self.layers[inputLayer].layerParams['kernel_h']
            w = self.layers[inputLayer].layerParams['kernel_w']
            if h ==1 and w ==1:
                silence = True
                
        if weightsOnly or convOnly:
            if include_pooling:
                silence_list.remove('Pooling')
            if self.layers[inputLayer].type in silence_list:
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
                    return_list = return_list +\
                    [inputList + path 
                     for path in self.get_max_paths(t, weightsOnly,
                                                   convOnly,phases,inception_unit)]
                    
            return return_list
        
    def path_between_layers(self, shallow_layer,deep_layer, weightsOnly = False, 
                            convOnly = False,phases = ['ALL'], 
                            inception_unit = False, include_pooling = False):
        """
        Returns all paths starting at shallow_layer and ending at deep_layer.
        
        Parameters:
            shallow_layer: str; the layer at which the path begins
            deep_layer: str; the layer at which the path ends
        
            weightsOnly: bool; whether or not to consider only layers which
            have weights/biases. Effectively reduces to considering only
            convolutional and inner product layers
            
            convOnly: bool; whether or not to consider only convolutional
            layers. If True, the path search will stop when a "Flatten" layer
            is met, as this signals the end of the convolutional segment of
            the network.
            
            phases: list; sublist of ['ALL', 'TEST', 'TRAIN']; indicates the 
            phases for which layers should be considered. Default is ['ALL'].
            
            inception_unit: bool; whether or not to treat an inception module
            as a single 'layer' (thus contributing only 1 to depth). Default:
            True
            
            include_pooling: bool; whether or not to add pooling layers to
            paths when weightsOnly or convOnly is set to True. Default: False
            
        Returns:
            a list of lists representing all paths from shallow_layer to 
            deep_layer. The first entry in every path is always inputLayer, and
            the last entry is always a leaf node in the network.
        """
        return_list = []
        all_paths = self.get_max_paths(shallow_layer, weightsOnly, convOnly, 
                                       phases, inception_unit, include_pooling)
        
        for path in all_paths:
            if deep_layer in path:
                index = path.index(deep_layer)
                return_list.append(path[:index+1])
        
        return return_list
    
    def get_depth(self, start_layer, key = 'MAX', weightsOnly = False, 
                            convOnly = False,phases = ['ALL'], 
                            inception_unit = False, include_pooling = False):
        """
        Uses key to return a summary statistic of network depth relative to start_layer.
        Network depth is measured as the length of a path starting at start_layer
        and ending at a network leaf-node (e.g. a loss layer).
        
        Parameters:
            start_layer: str; the first layer in every considered path.
            
            key: str, one of 'MAX', 'MIN', or 'AVG': specifies whether the
            maximum, minimum, or average depth should be returned. Default: MAX
            
            weightsOnly: bool; whether or not to consider only layers which
            have weights/biases. Effectively reduces to considering only
            convolutional and inner product layers
            
            convOnly: bool; whether or not to consider only convolutional
            layers. If True, the path search will stop when a "Flatten" layer
            is met, as this signals the end of the convolutional segment of
            the network.
            
            phases: list; sublist of ['ALL', 'TEST', 'TRAIN']; indicates the 
            phases for which layers should be considered. Default is ['ALL'].
            
            inception_unit: bool; whether or not to treat an inception module
            as a single 'layer' (thus contributing only 1 to depth). Default:
            True
            
            include_pooling: bool; whether or not to add pooling layers to
            paths when weightsOnly or convOnly is set to True. Default: False
            
        Returns:
            int/float summary_stat, str paths; paths is a list of all paths achieving
            the 'MAX' or 'MIN', if key is set to either of these
            
        """
        
        all_paths = self.get_max_paths(start_layer, weightsOnly, convOnly, 
                                       phases,inception_unit, include_pooling)
        if key == 'MAX':
            return_value = max([len(path) for path in all_paths])
            return_list = [path for path in all_paths if len(path)==return_value]
            return return_value,return_list
            
        elif key == 'MIN':
            return_value = min([len(path) for path in all_paths])
            return_list = [path for path in all_paths if len(path)==return_value]
            return return_value,return_list
            
        elif key == 'AVG':
            return_value = np.mean([len(path) for path in all_paths])
            return return_value
        else:
            print('Invalid key: ', key)
            return None
    
    def num_conv_layers(self, no_1x1 = False, phases = ['ALL'], inception_unit = False):
        """
        Returns the number of convolutional layers in the network. If no_1x1 is
        set to True, will ignore convolutional layers with 1x1 kernels. Will
        only count layers that participate in phases in 'phases'
        
        Parameters:
            no_1x1: bool; Default: False
            
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            inception_unit: bool; indicates whether or not convolutional layers
            within inception modules should be counted. Default: False.
            
        Returns:
            int
        """
        
        counter = 0
        
        for key in self.layers.keys():
            if self.layers[key].type == 'Convolution' and self.layers[key].phase in phases:
                if no_1x1:
                    h = self.layers[key].layerParams['kernel_h']
                    w = self.layers[key].layerParams['kernel_w']
                    if h==1 and w ==1:
                        continue
                if inception_unit and 'inception' in self.layers[key].name:
                    continue
                counter+=1
                
        return counter
    
    def num_inception_module(self, phases = ['ALL']):
        """
        Returns the number of inception module in the network which participate
        in the phases of phases.
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int
        """
        
        counter = 0
        
        for key in self.layers.keys():
            if self.layers[key].type == 'Concat' and\
            self.layers[key].phase in phases and 'inception' in key:
                counter +=1
                
        return counter
    
    def num_pooling_layers(self,phases = ['ALL'],inception_unit = False):
        """
        Returns the number of pooling layers in the network. Only considers
        layers which participate in the phases of 'phases'.
        
        Parameters:
            phases:list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            inception_unit: bool; indicates whether or not pooling layers
            within inception modules should be counted. Default: False.
            
        Returns:
            int
        """
        
        counter = 0
        
        for key in self.layers.keys():
            if self.layers[key].type == 'Pooling' and self.layers[key].phase in phases:
                h = self.layers[key].layerParams['kernel_h']
                w = self.layers[key].layerParams['kernel_w']
                if h==1 and w ==1:
                    continue
                if inception_unit and 'inception' in key:
                    continue
                counter+=1
                
        return counter
    
    def num_IP_layers(self, phases = ['ALL']):
        """
        Returns the number of inner product (fully-connected) layers in the network.
        Only considers layers which participate in the phases of 'phases'.
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int
        """
        
        counter = 0
        
        for key in self.layers.keys():
            if self.layers[key].type == 'InnerProduct' and self.layers[key].phase in phases:
                counter+=1
                
        return counter
    
    def num_skip_connections(self, phases = ['ALL']):
        """
        Returns the number of skip connections in phases of 'phases'.
        ***NEEDS IMPLEMENTATION***
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int
        """
        return None
    def len_skip_connections(self, key = 'MAX', phases = ['ALL']):
        """
        Uses 'key' to return a summary statistic for the length of skip connections
        in the network which participate in the phases of 'phases'.
        ****NEEDS IMPLEMENTATION******
        
        Parameters:
            
            key: str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
            statistic should be computed. Default: 'MAX'
            
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            
            int/float summary_stat, str layer_name; layer_name is the layer achieving
            the 'MAX' or 'MIN', if key is set to either of these
        """
        
        return None
    
    def num_IP_neurons(self, key = 'MAX', phases = ['ALL']):
        """
        Uses 'key' to return a summary statistic for the number of neurons
        per inner product layer.
        
        Parameters:
            key: str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
            statistic should be computed. Default: 'MAX'
            
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        IP_layers = [layer for layer in self.layers.keys() if 
                     self.layers[layer].type == 'InnerProduct' and
                     self.layers[layer].phase in phases]
        
        if key == 'MAX':
            return_value = max([self.layers[layer].layerParams['num_output'] for
                                layer in IP_layers])
            return_list = [layer for layer in IP_layers 
                           if self.layers[layer].layerParams['num_output']==return_value]
            return return_value,return_list
            
        elif key == 'MIN':
            return_value = min([self.layers[layer].layerParams['num_output'] for
                                layer in IP_layers])
            return_list = [layer for layer in IP_layers 
                           if self.layers[layer].layerParams['num_output']==return_value]
            return return_value,return_list
            
        elif key == 'AVG':
            return_value = np.mean([self.layers[layer].layerParams['num_output'] for
                                layer in IP_layers])
            return return_value
        else:
            print('Invalid key: ', key)
            return None
        
    def num_IP_weights(self, key = 'MAX', phases = ['ALL']):
        """
        Uses 'key' to return a summary statistic for the number of weights in
        a given IP layer. The number of weights is computed as num_input*num_output.
        Only layers participating in the phases of 'phases' are considered.
        
        Parameters:
            key: str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
            statistic should be computed. Default: 'MAX'
            
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
            
        """
        IP_layers = [layer for layer in self.layers.keys() if 
                     self.layers[layer].type == 'InnerProduct' and
                     self.layers[layer].phase in phases]
        if key == 'MAX':
            return_value = max([self.layers[layer].layerParams['num_output']*\
                                self.layers[layer].layerParams['num_input'] for
                                layer in IP_layers])
            return_list = [layer for layer in IP_layers 
                           if self.layers[layer].layerParams['num_output']*\
                           self.layers[layer].layerParams['num_input']==return_value]
            return return_value,return_list
            
        elif key == 'MIN':
            return_value = min([self.layers[layer].layerParams['num_output']*\
                                self.layers[layer].layerParams['num_input'] for
                                layer in IP_layers])
            return_list = [layer for layer in IP_layers 
                           if self.layers[layer].layerParams['num_output']*\
                           self.layers[layer].layerParams['num_input']==return_value]
            return return_value,return_list
            
        elif key == 'AVG':
            return_value = np.mean([self.layers[layer].layerParams['num_output']*\
                                self.layers[layer].layerParams['num_input'] for
                                layer in IP_layers])
            return return_value
        else:
            print('Invalid key: ', key)
            return None
        
    def num_splits(self, phases = ['ALL'], return_splits = False):
        """
        Returns the number of splits in the network. A 'split' is said to
        occur if a layer has multiple tops. Only layers participating in phases
        of 'phases' are considered. Nonlinearity layers are not counted towards
        the count to determine if a layer splits or not.
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            return_splits: bool; determines whether a list of split layers
            should be returned. Default: False
            
        Returns:
            int, [list of str]
        """
        
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
                    counter +=1
                
            if counter > 1:
                split_counter+=1
                if return_splits:
                    splits.append(layer)
                    
        if return_splits:
            return split_counter, splits
        else:
            return split_counter
        
    def split_widths(self, key = 'MAX', phases = ['ALL']):
        """
        Uses 'key' to return a summary statistic for the widths of splits
        in the network. The width of a split is defined as the number of
        layers to which it points (not including nonlinearity layers). Only
        layers participating in the phases of 'phases' are considered.
        
        Parameters:
            key: str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
            statistic should be computed. Default: 'MAX'
            
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        ignore_types = ['ReLU', 'PReLU', 'ELU', 'Sigmoid', 'TanH', 
                    'Power', 'Exp', 'Log', 'BNLL', 'Threshold', 
                    'Bias', 'Scale', 'Dropout']
        
        splits = self.num_splits(phases = phases, return_splits = True)[1]
        
        widths = []
        for split in splits:
            tops = [top for top in self.layers[split].top if\
                    top not in ignore_types]
            widths.append((len(tops),split))
        if key == 'MAX':
            return_value = max([width[0] for width in widths])
            return_list = [width[1] for width in widths if width[0]==return_value]
            return return_value,return_list
        elif key== 'MIN':
            return_value = min([width[0] for width in widths])
            return_list = [width[1] for width in widths if width[0]==return_value]
            return return_value,return_list
        elif key =='AVG':
            return_value = np.mean([width[0] for width in widths])
            return return_value
        else:
            print("Invalid key: ", key)
            return None
        
    def num_concats(self, phases = ['ALL'], return_concats = False):
        """
        Returns the number of concatenations in the network. A concatenation
        is defined as any layer which receives input from multiple layers; that
        is, any layer with multiple entries in its 'bottom' list. Only layers 
        participating in the phases of 'phases' are considered.
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            return_concats: bool; if True will return a list of all layers
            where a concatenation occurs
            
        Returns:
            int, list of str
        """
        
        concat_counter = 0
        concats = []
        
        for layer in self.layers.keys():
            if len(self.layers[layer].bottom) > 1:
                concat_counter +=1
                if return_concats:
                    concats.append(layer)
        if return_concats:
            return concat_counter,concats
        else:
            return concat_counter
        
    def concat_widths(self, key = 'MAX', phases = ['ALL']):
        """
        Uses 'key' to return a summary statistic for the widths of concats
        in the network. The width of a concat is defined as the number of
        layers which feed into it. Only layers participating in the phases 
        of 'phases' are considered.
        
        Parameters:
            key: str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
            statistic should be computed. Default: 'MAX'
            
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        concats = self.num_concats(phases = phases, return_concats = True)[1]
        
        widths = []
        for concat in concats:
            bottoms = [bottom for bottom in self.layers[concat].bottom]
            widths.append((len(bottoms),concat))
        if key == 'MAX':
            return_value = max([width[0] for width in widths])
            return_list = [width[1] for width in widths if width[0]==return_value]
            return return_value,return_list
        elif key== 'MIN':
            return_value = min([width[0] for width in widths])
            return_list = [width[1] for width in widths if width[0]==return_value]
            return return_value,return_list
        elif key =='AVG':
            return_value = np.mean([width[0] for width in widths])
            return return_value
        else:
            print("Invalid key: ", key)
            return None
        
    def conv_ker_area(self, key = 'MAX', phases = ['ALL'], include_inception = True):
        """
        Uses 'key' to return a summary statistic for the areas of convolutional 
        kernels in the network. Only layers participating in the phases 
        of 'phases' are considered.
        
        Parameters:
            key: str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
            statistic should be computed. Default: 'MAX'
            
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception: If false, convolutional layers appearing in 
            inception modules will not be considered.
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        conv_layers = [layer for layer in self.layers.keys() if 
                       self.layers[layer].type == 'Convolution']
        if not include_inception:
            conv_layers = [layer for layer in conv_layers if 
                           'inception' not in layer]
        areas = []
        for layer in conv_layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']
            area = h*w
            areas.append((area, layer))
        
        if key == 'MAX':
            return_value = max([area[0] for area in areas])
            return_list = [area[1] for area in areas if area[0]==return_value]
            return return_value,return_list
        elif key == 'MIN':
            return_value = min([area[0] for area in areas])
            return_list = [area[1] for area in areas if area[0]==return_value]
            return return_value,return_list
        elif key == 'AVG':
            return_value = np.mean([area[0] for area in areas])
            return return_value
        else:
            print("Invalid key: ", key)
            return None
    def num_conv_features(self, key = 'MAX', phases = ['ALL'], include_inception = True):
        """
        Uses 'key' to return a summary statistic for the number of features in 
        convolutional kernels in the network. Only layers participating in the 
        phases of 'phases' are considered.
        
        Parameters:
            key: str, one of 'MAX', 'MIN', or 'AVG'; indicates which summary
            statistic should be computed. Default: 'MAX'
            
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception: If false, convolutional layers appearing in 
            inception modules will not be considered.
            
        Returns:
            int/float summary_stat, list of str layer_names; layer_names is the
            list of layers achieving the 'MAX' or 'MIN', if key is set 
            to either of these.
        """
        conv_layers = [layer for layer in self.layers.keys() if 
                       self.layers[layer].type == 'Convolution']
        if not include_inception:
            conv_layers = [layer for layer in conv_layers if 
                           'inception' not in layer]
        features = []
        for layer in conv_layers:
            feat = self.layers[layer].layerParams['num_output']
            features.append((feat, layer))
        
        if key == 'MAX':
            return_value = max([feat[0] for feat in features])
            return_list = [feat[1] for feat in features if feat[0]==return_value]
            return return_value,return_list
        elif key == 'MIN':
            return_value = min([feat[0] for feat in features])
            return_list = [feat[1] for feat in features if feat[0]==return_value]
            return return_value,return_list
        elif key == 'AVG':
            return_value = np.mean([feat[0] for feat in features])
            return return_value
        else:
            print("Invalid key: ", key)
            return None
    
    def prop_conv_into_pool(self, phases = ['ALL'], include_inception = True):
        """
        Returns the proportion of convolutional layers which are followed by 
        a pooling layer. Only layers participating in phases of 'phases' are
        considered. Also returns the list of layers which are followed by
        pooling layers.
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception: If false, convolutional layers appearing in 
            inception modules will not be considered.
        
        Returns:
            float proportion, list of str pooled_conv_layers
            
        """
        pool_layers = [layer for layer in self.layers.keys() if 
                       self.layers[layer].type == 'Pooling']
        follows_pooling = []
        for layer in pool_layers:
            for bottom in self.layers[layer].bottom:
                if self.layers[bottom].type == 'Pooling':
                    h = self.layers[bottom].layerParams['kernel_h']
                    w = self.layers[bottom].layerParams['kernel_w']
                    if h==1 and w ==1:
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
        
        if num_conv_layers ==0 or len(pool_layers)==0:
            return 0,[]
        
        pooled_conv_layers = []
        
        for layer in pool_layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']
            if h==1 and w ==1:
                continue
            for bottom in self.layers[layer].bottom:
                if self.layers[bottom].type in ['LRN', 'Concat', 'Convolution']:
                    pooled_conv_layers.append(bottom)
                    break
            
        
        proportion = len(pooled_conv_layers)/num_conv_layers
        return proportion,pooled_conv_layers
    
    def prop_pool_into_pool(self, phases = ['ALL'], include_inception = True):
        """
        Returns the proportion of pooling layers which are followed by 
        a pooling layer. Only layers participating in phases of 'phases' are
        considered. Also returns the list of layers which are followed by
        pooling layers.
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception: If false, convolutional layers appearing in 
            inception modules will not be considered.
        
        Returns:
            float proportion, list of str pooled_pool_layers
            
        """
        pool_layers = [layer for layer in self.layers.keys() if 
                       self.layers[layer].type == 'Pooling']
        fake_pooling = []
        for layer in pool_layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']
            if h==1 and w ==1:
                fake_pooling.append(layer)
        pool_layers = [layer for layer in pool_layers if layer not in fake_pooling]
                    
        if not include_inception:
            pool_layers = [layer for layer in pool_layers if 
                           'inception' not in layer]
            
        num_pool_layers = len(pool_layers)
        
        if num_pool_layers ==0:
            return 0,[]
        
        pooled_pool_layers = []
        
        for layer in pool_layers:
            for bottom in self.layers[layer].bottom:
                if bottom in pool_layers:
                    pooled_pool_layers.append(bottom)
                    break
            
        
        proportion = len(pooled_pool_layers)/num_pool_layers
        return proportion,pooled_pool_layers
    
    def prop_padded_conv(self, phases = ['ALL'], include_inception = True):
        """
        Returns the proportion of convolutional layers which are padded. 
        Only layers participating in phases of 'phases' are considered. 
        Also returns the list of convolutional layers which are padded.
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception: If false, convolutional layers appearing in 
            inception modules will not be considered.
        
        Returns:
            float proportion, list of str padded_conv_layers
            
        """
        conv_layers = [layer for layer in self.layers.keys() if 
                       self.layers[layer].type == 'Convolution']
        if not include_inception:
            conv_layers = [layer for layer in conv_layers if 
                           'inception' not in layer]
        padded_layers = []
        
        num_conv_layers = len(conv_layers)
        
        if conv_layers ==0:
            return 0,[]
        
        for layer in conv_layers:
            h = self.layers[layer].layerParams['pad_h']
            w = self.layers[layer].layerParams['pad_w']
            
            if h == 0 and w ==0:
                continue
            else:
                padded_layers.append(layer)
                
        proportion = len(padded_layers)/num_conv_layers
        return proportion,padded_layers
        
    def prop_same_padded_conv(self, phases = ['ALL'], include_inception = True):
        """
        Returns the proportion of convolutional layers which are same-padded. 
        Only layers participating in phases of 'phases' are considered. 
        Also returns the list of convolutional layers which are same-padded.
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception: If false, convolutional layers appearing in 
            inception modules will not be considered.
        
        Returns:
            float proportion, list of str same_padded_conv_layers
            
        """
        conv_layers = [layer for layer in self.layers.keys() if 
                       self.layers[layer].type == 'Convolution']
        if not include_inception:
            conv_layers = [layer for layer in conv_layers if 
                           'inception' not in layer]
        same_padded_layers = []
        
        num_conv_layers = len(conv_layers)
        
        if num_conv_layers ==0:
            return 0,[]
        
        for layer in conv_layers:
            h = self.layers[layer].layerParams['pad_h']
            w = self.layers[layer].layerParams['pad_w']
            
            ip_grid = self.layers[layer].layerParams['input_grid'][0]
            r = ip_grid.shape[0]
            c = ip_grid.shape[1]
            ip_grid = ip_grid[h:r-h, w:c-w]
            op_grid = self.layers[layer].layerParams['input_grid'][0]
            
            if h == 0 and w ==0:
                continue
            elif ip_grid.shape!=op_grid.shape:
                continue
            else:
                same_padded_layers.append(layer)
                
        proportion = len(same_padded_layers)/num_conv_layers
        return proportion,same_padded_layers
    
    def prop_1x1_conv(self, phases = ['ALL'], include_inception = True):
        """
        Returns the proportion of convolutional layers wwith 1x1 kernels. 
        Only layers participating in phases of 'phases' are considered. 
        Also returns the list of convolutional layers with 1x1 kernels.
        
        Parameters:
            phases: list of 'ALL', 'TEST', or 'TRAIN'; Default: ['ALL']
            
            include_inception: If false, convolutional layers appearing in 
            inception modules will not be considered.
        
        Returns:
            float proportion, list of str same_padded_conv_layers
            
        """
        conv_layers = [layer for layer in self.layers.keys() if 
                       self.layers[layer].type == 'Convolution']
        if not include_inception:
            conv_layers = [layer for layer in conv_layers if 
                           'inception' not in layer]
        
        num_conv_layers = len(conv_layers)
        
        if num_conv_layers ==0:
            return 0,[]
        
        conv_1x1 = []
            
        for layer in conv_layers:
            h = self.layers[layer].layerParams['kernel_h']
            w = self.layers[layer].layerParams['kernel_w']
            
            if h==1 and w ==1:
                conv_1x1.append(layer)
                
        proportion = len(conv_1x1)/num_conv_layers
        return proportion, conv_1x1
    
        
    
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
            self.phase = inputLayer['phase']
        
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
        
    def print_layer(self):
        print("Name: ",self.name)
        print("Type: ",self.type)
        print("Top: ", self.top)
        print("Bottom: ", self.bottom)
        print("Phase: ", self.phase)
        for key in self.layerParams.keys():
            print(key,': ', self.layerParams[key])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
