# Learning 3D Segmentation With 2D Annotations
This is an implementation of the method described in the paper  
"[Learning to Segment 3D Linear Structures Using Only 2D Annotations](https://infoscience.epfl.ch/record/256857)".  
It contains a demonstration limited to the publicly available MRA dataset, referred to in the paper as "Angiography".

Prerequisites for running this code include a [torch](http://torch.ch/docs/getting-started.html) installation, the CUDNN libraries from NVIDIA, and their [torch bindings](https://github.com/soumith/cudnn.torch), 
and an installation of [optnet](https://github.com/fmassa/optimize-net).

Get the general network training routines, the code for preprocessing of the dataset, and the experiment code  
`git clone https://github.com/mkozinski/NetworkTraining`  
`git clone https://github.com/mkozinski/MRAdata`  
`git clone https://github.com/mkozinski/Learning3DSegmentationWith2DAnnotations`  

To fetch and preprocesss the dataset  
`cd MRAdata`  
`./prepareData.sh`

To run baseline training on 3D annotations  
`cd Learning3DSegmentationWith2DAnnotations`  
`th -e "dofile \"run_3D_annotation.lua\""`  
The training progress is logged to a directory called `log_3D_annotation`.

To run training on 2D annotations  
`cd Learning3DSegmentationWith2DAnnotations`  
`th -e "dofile \"run_mip_annotation.lua\""`  
The log directory for this training is `log_mip_annotation`.

In both cases the training progress can be plotted in gnuplot:  
a) the epoch-averaged loss on the training data `plot "<logdir>/basicEpoch.log" u 1`,  
b) the F1 performance on the test data `plot "<logdir>/testEpoch.log" u 1`.

The training loss and test performance plots are not synchronised as testing is performed once every 50 epochs.
Moreover, the logs generated for the two experiments are synchronised in terms of the number of epochs, but not in terms of the number of updates.

The networks with the trained weights are dumped in the log directories:  
a) the recent network is stored at `<logdir>/basic_net_last.t7`,  
b) the network attaining the highest F1 score on the test set is stored at `<logdir>/test_net_bestF1.t7`.

To generate prediction for the test set using a trained network run  
`th -e "dofile \"saveOutputVolumes.lua\""`.  
The name of the file containing the network, and the output directory are defined at the beginning of the script.
The output is saved in form of stacks of png images.

Example output for a test file, after around 100 000 updates (please note that the number of updates specified in the script is 50 000), for a network scoring 76% IoU on a held out set of volumes.  
![Animation:rotating brain vasculature.](http://documents.epfl.ch/users/k/ko/kozinski/www/brain_vasculature.aa)  
The segmentation has been visualized with ![slicer](https://www.slicer.org/), in particular using the "Volume Rendering" and "Screen Capture" modules.
