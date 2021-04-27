# 3D-Mesh-Classification-PointNet

 
This is a Computer Vision project aiming to apply deep learning to point clouds for object classification and part/scene semantic segmentation. The implementation of the model has been inspired from the original [PointNet](https://arxiv.org/abs/1612.00593) paper admited to CVPR 2017 at the Computer Vision and Pattern Recognition subjects. [PointNet](https://arxiv.org/abs/1612.00593) is indeed a seminal paper in 3D perception.


## Goal

## Data

PointNet takes raw point cloud data as input, which is typically collected from either a lidar or radar sensor. Unlike 2D pixel arrays (images) or 3D voxel arrays, point clouds have an unstructured representation in that the data is simply a collection (more specifically, a set) of the points captured during a lidar or radar sensor scan. In order to leverage existing techniques built around (2D and 3D) convolutions, many researchers and practitioners often discretize a point cloud by taking multi-view projections onto 2D space or quantizing it to 3D voxels. Given that the original data is manipulated, either approach can have negative impacts.
For simplicity, it’ll be assumed that a point in a point cloud is fully described by its (x, y, z) coordinates. In practice, other features may be included, such as surface normal and intensity.


In the context of this project, we aquired Princeton ModelNet dataset of 10 classes already splitted into train and test sets.
The 10 Categories are:

 - [ ] toilet
 - [ ] desk
 - [ ] table
 - [ ] bathtub
 - [ ] chair
 - [ ] bed
 - [ ] dresser
 - [ ] night_stand
 - [ ] sofa
 - [ ] monitor

## Let's visualize one of them (bed)!

                                             Visualization without the Points Cloud

<p align="center">
  <img src="https://github.com/IsmaelMekene/3D-Mesh-Classification-PointNet/blob/main/data/bed_new.gif"/>
</p>


                                          Visualization with the Points Cloud (4096 points)



<p align="center">
  <img src="https://github.com/IsmaelMekene/3D-Mesh-Classification-PointNet/blob/main/data/bedmeshpoints.png"/>
</p>



## Model

Given that PointNet consumes raw point cloud data, it was necessary to develop an architecture that conformed to the unique properties of point sets. Among these, the authors emphasize:

- Permutation (Order) Invariance: given the unstructured nature of point cloud data, a scan made up of N points has N! permutations. The subsequent data processing must be invariant to the different representations.

- Transformation Invariance: classification and segmentation outputs should be unchanged if the object undergoes certain transformations, including rotation and translation.

- Point Interactions: the interaction between neighboring points often carries useful information (i.e., a single point should not be treated in isolation). Whereas classification need only make use of global features, segmentation must be able to leverage local point features along with global point features.
 
 

                                                        PointNet Architechture
                                                        
<p align="center">
  <img src="https://github.com/IsmaelMekene/3D-Mesh-Classification-PointNet/blob/main/data/model.png"/>
</p>


The architecture is surprisingly simple and quite intuitive. The classification network uses a shared multi-layer perceptron (MLP) to map each of the n points from three dimensions (x, y, z) to 64 dimensions. It’s important to note that a single multi-layer perceptron is shared for each of the n points (i.e., mapping is identical and independent on the n points). This procedure is repeated to map the n points from 64 dimensions to 1024 dimensions. With the points in a higher-dimensional embedding space, max pooling is used to create a global feature vector in ℝ¹⁰²⁴. 

As for the segmentation network, each of the n input points needs to be assigned to one of m segmentation classes. Because segmentation relies on local and global features, the points in the 64-dimensional embedding space (local point features) are concatenated with the global feature vector (global point features), resulting in a per-point vector in ℝ¹⁰⁸⁸. Similar to the multi-layer perceptrons used in the classification network, MLPs are used (identically and independently) on the n points to lower the dimensionality from 1088 to 128 and again to m, resulting in an array of n x m.



                                                        Build and Train the Model

In the context of this project, the focus is made on the classification network. The model is inspired from the original paper and the training was set to 100 epochs with a callback of early stopping.

Model Summary:



                   Model: "pointnet"
                   __________________________________________________________________________________________________
                   Layer (type)                    Output Shape         Param #     Connected to                     
                   ==================================================================================================
                   input_3 (InputLayer)            [(None, 4096, 3)]    0                                            
                   __________________________________________________________________________________________________
                   conv1d_22 (Conv1D)              (None, 4096, 32)     128         input_3[0][0]                    
                   __________________________________________________________________________________________________
                   batch_normalization_34 (BatchNo (None, 4096, 32)     128         conv1d_22[0][0]                  
                   __________________________________________________________________________________________________
                   activation_34 (Activation)      (None, 4096, 32)     0           batch_normalization_34[0][0]     
                   __________________________________________________________________________________________________
                   conv1d_23 (Conv1D)              (None, 4096, 64)     2112        activation_34[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_35 (BatchNo (None, 4096, 64)     256         conv1d_23[0][0]                  
                   __________________________________________________________________________________________________
                   activation_35 (Activation)      (None, 4096, 64)     0           batch_normalization_35[0][0]     
                   __________________________________________________________________________________________________
                   conv1d_24 (Conv1D)              (None, 4096, 512)    33280       activation_35[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_36 (BatchNo (None, 4096, 512)    2048        conv1d_24[0][0]                  
                   __________________________________________________________________________________________________
                   activation_36 (Activation)      (None, 4096, 512)    0           batch_normalization_36[0][0]     
                   __________________________________________________________________________________________________
                   global_max_pooling1d_6 (GlobalM (None, 512)          0           activation_36[0][0]              
                   __________________________________________________________________________________________________
                   dense_17 (Dense)                (None, 256)          131328      global_max_pooling1d_6[0][0]     
                   __________________________________________________________________________________________________
                   batch_normalization_37 (BatchNo (None, 256)          1024        dense_17[0][0]                   
                   __________________________________________________________________________________________________
                   activation_37 (Activation)      (None, 256)          0           batch_normalization_37[0][0]     
                   __________________________________________________________________________________________________
                   dense_18 (Dense)                (None, 128)          32896       activation_37[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_38 (BatchNo (None, 128)          512         dense_18[0][0]                   
                   __________________________________________________________________________________________________
                   activation_38 (Activation)      (None, 128)          0           batch_normalization_38[0][0]     
                   __________________________________________________________________________________________________
                   dense_19 (Dense)                (None, 9)            1161        activation_38[0][0]              
                   __________________________________________________________________________________________________
                   reshape_4 (Reshape)             (None, 3, 3)         0           dense_19[0][0]                   
                   __________________________________________________________________________________________________
                   dot_4 (Dot)                     (None, 4096, 3)      0           input_3[0][0]                    
                                                                                    reshape_4[0][0]                  
                   __________________________________________________________________________________________________
                   conv1d_25 (Conv1D)              (None, 4096, 32)     128         dot_4[0][0]                      
                   __________________________________________________________________________________________________
                   batch_normalization_39 (BatchNo (None, 4096, 32)     128         conv1d_25[0][0]                  
                   __________________________________________________________________________________________________
                   activation_39 (Activation)      (None, 4096, 32)     0           batch_normalization_39[0][0]     
                   __________________________________________________________________________________________________
                   conv1d_26 (Conv1D)              (None, 4096, 32)     1056        activation_39[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_40 (BatchNo (None, 4096, 32)     128         conv1d_26[0][0]                  
                   __________________________________________________________________________________________________
                   activation_40 (Activation)      (None, 4096, 32)     0           batch_normalization_40[0][0]     
                   __________________________________________________________________________________________________
                   conv1d_27 (Conv1D)              (None, 4096, 32)     1056        activation_40[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_41 (BatchNo (None, 4096, 32)     128         conv1d_27[0][0]                  
                   __________________________________________________________________________________________________
                   activation_41 (Activation)      (None, 4096, 32)     0           batch_normalization_41[0][0]     
                   __________________________________________________________________________________________________
                   conv1d_28 (Conv1D)              (None, 4096, 64)     2112        activation_41[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_42 (BatchNo (None, 4096, 64)     256         conv1d_28[0][0]                  
                   __________________________________________________________________________________________________
                   activation_42 (Activation)      (None, 4096, 64)     0           batch_normalization_42[0][0]     
                   __________________________________________________________________________________________________
                   conv1d_29 (Conv1D)              (None, 4096, 512)    33280       activation_42[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_43 (BatchNo (None, 4096, 512)    2048        conv1d_29[0][0]                  
                   __________________________________________________________________________________________________
                   activation_43 (Activation)      (None, 4096, 512)    0           batch_normalization_43[0][0]     
                   __________________________________________________________________________________________________
                   global_max_pooling1d_7 (GlobalM (None, 512)          0           activation_43[0][0]              
                   __________________________________________________________________________________________________
                   dense_20 (Dense)                (None, 256)          131328      global_max_pooling1d_7[0][0]     
                   __________________________________________________________________________________________________
                   batch_normalization_44 (BatchNo (None, 256)          1024        dense_20[0][0]                   
                   __________________________________________________________________________________________________
                   activation_44 (Activation)      (None, 256)          0           batch_normalization_44[0][0]     
                   __________________________________________________________________________________________________
                   dense_21 (Dense)                (None, 128)          32896       activation_44[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_45 (BatchNo (None, 128)          512         dense_21[0][0]                   
                   __________________________________________________________________________________________________
                   activation_45 (Activation)      (None, 128)          0           batch_normalization_45[0][0]     
                   __________________________________________________________________________________________________
                   dense_22 (Dense)                (None, 1024)         132096      activation_45[0][0]              
                   __________________________________________________________________________________________________
                   reshape_5 (Reshape)             (None, 32, 32)       0           dense_22[0][0]                   
                   __________________________________________________________________________________________________
                   dot_5 (Dot)                     (None, 4096, 32)     0           activation_40[0][0]              
                                                                                    reshape_5[0][0]                  
                   __________________________________________________________________________________________________
                   conv1d_30 (Conv1D)              (None, 4096, 32)     1056        dot_5[0][0]                      
                   __________________________________________________________________________________________________
                   batch_normalization_46 (BatchNo (None, 4096, 32)     128         conv1d_30[0][0]                  
                   __________________________________________________________________________________________________
                   activation_46 (Activation)      (None, 4096, 32)     0           batch_normalization_46[0][0]     
                   __________________________________________________________________________________________________
                   conv1d_31 (Conv1D)              (None, 4096, 64)     2112        activation_46[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_47 (BatchNo (None, 4096, 64)     256         conv1d_31[0][0]                  
                   __________________________________________________________________________________________________
                   activation_47 (Activation)      (None, 4096, 64)     0           batch_normalization_47[0][0]     
                   __________________________________________________________________________________________________
                   conv1d_32 (Conv1D)              (None, 4096, 512)    33280       activation_47[0][0]              
                   __________________________________________________________________________________________________
                   batch_normalization_48 (BatchNo (None, 4096, 512)    2048        conv1d_32[0][0]                  
                   __________________________________________________________________________________________________
                   activation_48 (Activation)      (None, 4096, 512)    0           batch_normalization_48[0][0]     
                   __________________________________________________________________________________________________
                   global_max_pooling1d_8 (GlobalM (None, 512)          0           activation_48[0][0]              
                   __________________________________________________________________________________________________
                   dense_23 (Dense)                (None, 256)          131328      global_max_pooling1d_8[0][0]     
                   __________________________________________________________________________________________________
                   batch_normalization_49 (BatchNo (None, 256)          1024        dense_23[0][0]                   
                   __________________________________________________________________________________________________
                   activation_49 (Activation)      (None, 256)          0           batch_normalization_49[0][0]     
                   __________________________________________________________________________________________________
                   dropout_4 (Dropout)             (None, 256)          0           activation_49[0][0]              
                   __________________________________________________________________________________________________
                   dense_24 (Dense)                (None, 128)          32896       dropout_4[0][0]                  
                   __________________________________________________________________________________________________
                   batch_normalization_50 (BatchNo (None, 128)          512         dense_24[0][0]                   
                   __________________________________________________________________________________________________
                   activation_50 (Activation)      (None, 128)          0           batch_normalization_50[0][0]     
                   __________________________________________________________________________________________________
                   dropout_5 (Dropout)             (None, 128)          0           activation_50[0][0]              
                   __________________________________________________________________________________________________
                   dense_25 (Dense)                (None, 10)           1290        dropout_5[0][0]                  
                   ==================================================================================================
                   Total params: 748,979
                   Trainable params: 742,899
                   Non-trainable params: 6,080
                   __________________________________________________________________________________________________
                   
                   
                   
                   
                   
<p align="center">
  <img src="https://github.com/IsmaelMekene/3D-Mesh-Classification-PointNet/blob/main/data/epoch_loss.svg"/>
</p>

