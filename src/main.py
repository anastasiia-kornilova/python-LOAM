import copy
import numpy as np
import open3d as o3d
import os

from odometry_estimator import OdometryEstimator


def find_transformation(source, target, trans_init):
    threshold = 0.2
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    transformation = o3d.registration.registration_icp(source, target, threshold, trans_init,
                                            o3d.registration.TransformationEstimationPointToPlane()).transformation
    return transformation


def get_pcd_from_numpy(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    return pcd


if __name__ == '__main__':
    folder = '../alignment/numpy/'
    pcds_list = os.listdir(folder)
    pcds_list.sort()

    odometry = OdometryEstimator()
    global_transform = np.eye(4)
    pcds = []
    for i in range(0, 50):
        path = folder + pcds_list[i]
        pcd_np = np.load(path)[:, :3]

        T = odometry.append_pcd(pcd_np)
        global_transform = T @ global_transform
        pcd = get_pcd_from_numpy(pcd_np)
        pcds.append(pcd.transform(global_transform))

    o3d.visualization.draw_geometries(pcds)

    pcds = []
    global_transform = np.eye(4)
    for i in range(0, 50):
        path_1 = folder + pcds_list[i]
        pcd_np_1 = get_pcd_from_numpy(np.load(path_1)[:, :3])

        path_2 = folder + pcds_list[i + 1]
        pcd_np_2 = get_pcd_from_numpy(np.load(path_2)[:, :3])

        T = find_transformation(pcd_np_2, pcd_np_1, np.eye(4))
        global_transform = T @ global_transform
        pcds.append(copy.deepcopy(pcd_np_2).transform(global_transform))

    o3d.visualization.draw_geometries(pcds)
