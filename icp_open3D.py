# -*- coding: utf8 -*-
import numpy as np
import copy
import math
import time
import h5py
from utils import *
import open3d as o3


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3.draw_geometries([source_temp, target])


if __name__ == "__main__":


    Tg = generatePerturbation(theta=10, translation=0.005)
    source = o3.read_point_cloud("./data/NP5_0.pcd")
    target = copy.deepcopy(source)
    target.transform(Tg)

    '''target = o3.read_point_cloud("./ycb/ycb-scripts-master/ycb/001_chips_can/clouds/NP5_327.pcd")
    f_T = h5py.File('./ycb/ycb-scripts-master/ycb/001_chips_can/poses/NP5_327_pose.h5','r')   #打开h5文件

    Tg = f_T['H_table_from_reference_camera'][:]
    offset = f_T['board_frame_offset'][:]
    print (Tg)'''
    #Tg[:3, 3] -= offset




    # preprocess
    '''source, _ = geometry.statistical_outlier_removal(source, 20, 2)
    target, _ = geometry.statistical_outlier_removal(target, 20, 2)
    source_for_color = copy.deepcopy(source) # color icp needs more points
    source = geometry.uniform_down_sample(source, round(cloudSize / 10000))'''


    print (source)
    print (target)

    current_transformation = np.identity(4)
    draw_registration_result_original_color(
     source, target, current_transformation)
    myCriteria = o3.ICPConvergenceCriteria(1e-8, 1e-8, 100);



    # point to point ICP
    t1 = time.time()
    current_transformation = np.identity(4)
    print ("Point-to-point ICP")
    result_icp = o3.registration_icp(source, target, 0.25,
                                  current_transformation, o3.TransformationEstimationPointToPoint(),myCriteria)
    t2 = time.time()
    print ("Time Consumed: ", t2 - t1)
    print(result_icp)
    print (result_icp.transformation)
    err_t, err_R = eval_err(result_icp.transformation, Tg)
    print ("err_t: ", err_t)
    print ("err_R: ", err_R)
    print ()
    draw_registration_result_original_color(
      source, target, result_icp.transformation)




    # point to plane ICP
    current_transformation = np.identity(4);
    print("Point-to-plane ICP")
    o3.estimate_normals(source, o3.KDTreeSearchParamHybrid(
        radius=0.4, max_nn=30))
    o3.estimate_normals(target, o3.KDTreeSearchParamHybrid(
        radius=0.4, max_nn=30))
    t3 = time.time()
    result_icp = o3.registration_icp(source, target, 0.25,
            current_transformation, o3.TransformationEstimationPointToPlane(),myCriteria)
    t4 = time.time()
    print ("Time Consumed: ", t4 - t3)
    print(result_icp)
    print(result_icp.transformation)
    err_t, err_R = eval_err(result_icp.transformation, Tg)
    print("err_t: ", err_t)
    print("err_R: ", err_R)
    print()
    draw_registration_result_original_color(
          source, target, result_icp.transformation)



    # colored pointcloud registration
    # coarse to fine
    voxel_radius = [ 0.4, 0.2, 0.1 ];
    max_iter = [ 50, 30, 14 ];
    current_transformation = np.identity(4)
    print("ColoredICP")
    t5 = time.time()
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter,radius,scale])

        print("Downsample with a voxel size %.2f" % radius)
        source_down = o3.voxel_down_sample(source, radius)
        target_down = o3.voxel_down_sample(target, radius)

        print("Estimate normal.")
        o3.estimate_normals(source_down, o3.KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))
        o3.estimate_normals(target_down, o3.KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))

        print("Applying colored point cloud registration")
        result_icp = o3.registration_colored_icp(source_down, target_down,
                radius, current_transformation,
                o3.ICPConvergenceCriteria(relative_fitness = 1e-6,
                relative_rmse = 1e-8, max_iteration = iter))
        current_transformation = result_icp.transformation

    t6 = time.time()
    print(result_icp)
    print("Time Consumed: ", t6 - t5)
    print(result_icp.transformation)
    err_t, err_R = eval_err(result_icp.transformation, Tg)
    print("err_t: ", err_t)
    print("err_R: ", err_R)
    print()
    draw_registration_result_original_color(
         source, target, result_icp.transformation)
