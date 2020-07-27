import copy
import numpy as np
import open3d as o3d

# from loader_vlp16 import LoaderVLP16
from loader_kitti import LoaderKITTI
from mapping import Mapper
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
    # folder = '../../alignment/numpy/'
    folder = '/home/anastasiya/data/data_odometry_velodyne.zip/'

    loader = LoaderKITTI(folder, '00')

    odometry = OdometryEstimator()
    global_transform = np.eye(4)
    pcds = []
    mapper = Mapper()
    for i in range(loader.length()):
        pcd = loader.get_item(i)
        T, sharp_points, flat_points = odometry.append_pcd(pcd)
        mapper.append_undistorted(pcd[0], T, sharp_points, flat_points, vis=(i % 2 == 0))

    # Visual comparison with point-to-plane ICP
    pcds = []
    global_transform = np.eye(4)
    for i in range(loader.length() - 1):
        pcd_np_1 = get_pcd_from_numpy(loader.get_item(i))
        pcd_np_2 = get_pcd_from_numpy(loader.get_item(i + 1))

        T = find_transformation(pcd_np_2, pcd_np_1, np.eye(4))
        global_transform = T @ global_transform
        pcds.append(copy.deepcopy(pcd_np_2).transform(global_transform))

    o3d.visualization.draw_geometries(pcds)
