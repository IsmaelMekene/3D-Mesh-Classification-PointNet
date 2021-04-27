# 3D-Mesh-Classification-PointNet

 
This is a Computer Vision project aiming to apply deep learning to point clouds for object classification and part/scene semantic segmentation. The implementation of the model has been inspired from the original [PointNet](https://arxiv.org/abs/1612.00593) paper admited to CVPR 2017 at the Computer Vision and Pattern Recognition subjects. [PointNet](https://arxiv.org/abs/1612.00593) is indeed a seminal paper in 3D perception.


## Goal

## Data

PointNet takes raw point cloud data as input, which is typically collected from either a lidar or radar sensor. Unlike 2D pixel arrays (images) or 3D voxel arrays, point clouds have an unstructured representation in that the data is simply a collection (more specifically, a set) of the points captured during a lidar or radar sensor scan. In order to leverage existing techniques built around (2D and 3D) convolutions, many researchers and practitioners often discretize a point cloud by taking multi-view projections onto 2D space or quantizing it to 3D voxels. Given that the original data is manipulated, either approach can have negative impacts.
For simplicity, it’ll be assumed that a point in a point cloud is fully described by its (x, y, z) coordinates. In practice, other features may be included, such as surface normal and intensity.


In the context of this project, we aquired Princeton ModelNet dataset of 10 classes already splitted into train and test sets.
The 10 Categories are:

 - [] toilet
 - [] desk
 - [] table
 - [] bathtub
 - [] chair
 - [] bed
 - [] dresser
 - [] night_stand
 - [] sofa
 - [] monitor

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
Permutation (Order) Invariance: given the unstructured nature of point cloud data, a scan made up of N points has N! permutations. The subsequent data processing must be invariant to the different representations.
Transformation Invariance: classification and segmentation outputs should be unchanged if the object undergoes certain transformations, including rotation and translation.
Point Interactions: the interaction between neighboring points often carries useful information (i.e., a single point should not be treated in isolation). Whereas classification need only make use of global features, segmentation must be able to leverage local point features along with global point features.
