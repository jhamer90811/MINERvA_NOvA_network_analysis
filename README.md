MINERvA_NOvA_network_analysis
=============================

Introduction
-----------------------------

This repository contains Python scripts for the handling of Caffe neural
networks designed for vertex finding in the MINERvA dataset and also neural
networks designed for the NOvA prong dataset. 

* The network handler is written in
[Network.py](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/blob/master/Network.py).

* A script to test whether all necessary libraries are installed and load
properly may be found in [test_imports.py](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/blob/master/test_imports.py).

* [test_networks](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/test_networks)
contains sample networks (as protobuf files) from both the MINERvA vertex finders as well as NOvA
prong. Pictures of the networks are also included.

* [caffe_files](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/caffe_files)
contains the original Caffe network protobuf file, as well as the Python class
compiled by Google's protobuf software. One minor modification was made to the
original protobuf file in order to implement the "gradient scaler layer".
Besides this, the file is the same as can be found on [Caffe's repository](https://github.com/BVLC/caffe).

* Data collection scripts for simple and complex attributes can be found
in [data_collection_scripts](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/data_collection_scripts).

* Bash scripts which submit these data-collecting Python scripts to the Wilson
Cluster as batch jobs may be found in [simple_batch_jobs](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/simple_batch_jobs)
and [complex_batch_jobs](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/complex_batch_jobs).

* Raw datasets for simple attributes can be found in [simple_datasets](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/simple_datasets).

* Data analysis was performed according to the scripts in [analysis_scripts](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/analysis_scripts)
as well as [R_models.R](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/blob/master/R_models.R). 

* For complex attribute datasets, as well as cleaner simple attribute datasets,
the user should consult the following Dropbox folders:
    - [Simple Attributes](https://www.dropbox.com/sh/mirz9pw0tlp87cq/AADoMgpPyabxeKr_9QRNlgk5a?dl=0)
    - [Complex Attributes](https://www.dropbox.com/sh/q6o4xvm2voq7jop/AAAvc9rXj_Ek6DpHbgZQkzwKa?dl=0)
    
* [Simplicial](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/simplicial)
is a Python package originally written in Python2 by Simon Dobson (the original
can be found [here](https://simplicial.readthedocs.io/en/latest/)). The version
in this repository makes changes for compatibility with Python3.

* The [Network.py](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/blob/master/Network.py)
file requires an installation of [GUDHI](http://gudhi.gforge.inria.fr) in the
Python directory. GUDHI is a library for topological data analysis. A Singularity
container which implements GUDHI with Python3 may be found [here](https://www.singularity-hub.org/collections/1333)
as "ubuntu16.04-network_topology2". It is recommended that this container be
used for any Wilson Cluster batch jobs which require the files in this repository.
    - See below for advice on getting an operational version of this container
    on the Wilson Cluster.
    
For testing purposes (and further data collection), it is advised that the user 
seek out the full datasets of MINERvA and NOvA networks trained by MENNDL, 
as well as the image datasets (as HDF5 files) collected by the MINERvA and 
NOvA detectors.
    
[Network.py](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/blob/master/Network.py)
-------------------------------

This file is the main handler for networks trained for MINERvA vertex finding
(with DANN) and networks trained on the NOvA prong dataset (with inception 
modules). With modifications to the constructor (as well as auxiliary 
constructor functions), it could also handle a wider array of networks. The
Network class is the main class of the file which parses Caffe prototxt files.

The main restriction to handling a more general class of networks comes from the conventions employed by
the original authors of the Caffe network prototxt files. For example, networks
designed for the NOvA prong dataset permit inception modules. These inception
modules always begin with a split layer and end with a concat layer, and the
names of all constituent layers are prefaced by "inception". Several methods
of Network.py make this check whenever a split or concat layer is encountered.
Other types of networks may have similar conventions which require modifications
to Network.py for desired handling.

The Network class also implements representations of input and output grids for layers
where this is appropriate (if padding is present, the dimensions of input
grids reflect this). This, along with the "feed_image" method, allows users to
feed an input image to all of the input layers of the network. The method 
"get_flow" then allows the user to flow the image through the network and
populate the input/output grids of deeper layers.

Any 1x1 pooling layers in Caffe prototxt files are removed, and the constructor 
modifies the original top/bottom lists of relevant layers to account for this.

 The constructor also implements new layer types (most notably, "Input" layers)
 for layers which in Caffe prototxt files were only referenced in the "top"
 attribute of another layer.
 
 The "Layer" class is the main handler for specific layers. The "layerParams"
 attribute houses any relevant attributes specified for the layer in the 
 Caffe protobuf. The "layerParams" attribute also houses the input and output
 grids for the layer, where appropriate. The "imgFeatures" attribute has
 fields pertaining to a specific image fed to the network by "feed_image".
 
 The "UnionIter" class is an iterator which allows the user to compute and
 iterate through the union of multiple sets of points of an activation grid.
 Sets of points may be specified either as a list of points (2-tuples), or
 as lists of "fields": a length-4 list whose first two entries are the
 coordinates of the upper-left corner point and whose last two entries are
 the coordinates of the lower-right corner point of a rectangular subregion
 of the image.
 
 [test_imports.py](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/blob/master/test_imports.py)
 ---------------------
 
 This file checks to ensure that all necessary libraries are load properly.
 The user should beware, however, that although GUDHI may import, if it is not
 installed properly then it could be missing key packages which allow for
 full GUDHI functionality. In particular, the user should ensure that recent
 versions of CGAL and Eigen3 are installed. See the [GUDHI documentation](http://gudhi.gforge.inria.fr)
 for more on how to install GUDHI properly.
 
 The user may also need to change certain import statements, depending on where
 the relevant libraries are being kept and the working directory from which
 the script is called.
 
 [data_collection_scripts](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/data_collection_scripts)
 ------------------------
 
 Both get_complex_attributes.py and get_simple_attributes.py expect network
 prototxt files and corresponding output .txt files to be in the same directory.
 The scripts iterate through the genealogies indicated by the command line
 arguments, scraping output files for initial and final accuracy, and network
 prototxt files for the network specifications themselves (followed by the
 extraction of all network attributes indicated in the script).
 
 It should be noted that in the initial run of the simple attribute collection
 script (from which the datasets in [simple_datasets](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/simple_datasets)
 were collected), max_num_conv_features and min_num_conv_features  were
 mistakenly named as num_conv_features and num_conv_features.1. Moreover,
 the attribute named "avg_conv_ker_area" mistakenly computed the 
 avg_num_conv_features. All of these attributes were thrown out prior to
 analysis.
 
 In the complex attribute analysis script, the R^2 values for the auxiliary
 linear models trained for each score were computed, but the corresponding
 p-values were not also computed. Thus this R^2 data was ignored in the 
 data_analysis.
 
 Initial runs of the complex attribute collection script on the Wilson Cluster
 took about 24 hours on the intel12 nodes to collect data on roughly 50 genealogies. However, many
 of these jobs failed with SEGFAULT errors. The issue causing these SEGFAULT
 errors has not yet been resolved.
 
 [simple_batch_jobs](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/simple_batch_jobs) and [complex_batch_jobs](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/complex_batch_jobs)
 ---------------------------------
 
 These are the actual bash scripts submitted to the Wilson cluster for data
 collection. They require the user to specify input and output directories,
 as well as the range of genealogies from which to collect data. The simple
 attribute job took 3-4 hours to collect data on all MINERvA networks. The
 complex attribute jobs were subdivided into jobs which collected data on
 groups of 45 genealogies at a time. (This was the expected number of genealogies
 from which data could be collected in a 24-hour window, although the number
 was underestimated to ensure that the jobs actually finish within 24-hours.
 That said, as indicated above many of the jobs failed with SEGFAULT errors
 after collecting data on several networks. The number of networks from which
 data could be collected before failing with a SEGFAULT seemed to vary 
 randomly with each job. The cause of this SEGFAULT issue has still not been
 found).
 
 [analysis_scripts](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/analysis_scripts) and [R_models.R](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/blob/master/R_models.R)
 ---------------------------
 
 These scripts contain a history of the analysis performed on simple and complex
 attribute data collected on MINERvA networks. Initial data exploration, 
 correlation analysis, data cleansing, histogram plotting, PCA analysis/plotting,
 cluster analysis, and attempts at Isomap/t-SNE for complex attribute data can
 all be found in [analysis_scripts](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/analysis_scripts).
 The R script [R_models.R](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/blob/master/R_models.R)
 contains a history of linear and quadratic regression and analysis for various
 datasets.
 
 Datasets produced and used in [analysis_scripts](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/tree/master/analysis_scripts)
 and [R_models.R](https://github.com/jhamer90811/MINERvA_NOvA_network_analysis/blob/master/R_models.R)
 can be found at the following Dropbox locations:
  * [Simple Attributes](https://www.dropbox.com/sh/mirz9pw0tlp87cq/AADoMgpPyabxeKr_9QRNlgk5a?dl=0)
  * [Complex Attributes](https://www.dropbox.com/sh/q6o4xvm2voq7jop/AAAvc9rXj_Ek6DpHbgZQkzwKa?dl=0)
    
