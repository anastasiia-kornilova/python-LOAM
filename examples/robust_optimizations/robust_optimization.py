import csv
import numpy as np
import open3d as o3d
import sys


def point_from_str(str_array):
    point = []
    for elem in list(filter(None, str_array[1:-1].split(' '))):
        point.append(float(elem))
    return np.array(point)


def get_pcd_from_numpy(np_pcd, color=[0, 0, 1]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd[:, :3])
    pcd.paint_uniform_color(color)
    return pcd


if __name__ == '__main__':
    edge_corresp_file = '../../src/preprocessed_corresp/edges/000000.csv'
    edge_corresp_set = []
    with open(edge_corresp_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            edge_corresp_set.append((point_from_str(row[0]), point_from_str(row[1]), point_from_str(row[2])))

    plane_corresp_file = '../../src/preprocessed_corresp/planes/000000.csv'
    plane_corresp_set = []
    with open(plane_corresp_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            plane_corresp_set.append((point_from_str(row[0]), point_from_str(row[1]), point_from_str(row[2])
                                , point_from_str(row[3])))

    # I know that is the most strange thing, TO FIX
    sys.path.append("../../src")
    from loader_kitti import LoaderKITTI
    from LOAMSolver import LOAMSolver
    solver = LOAMSolver(use_estimators=True, region_rate=0.4)
    T, inlier_errors, edge_inliers, plane_inliers = solver.fit(edge_corresp_set, plane_corresp_set)

    loader = LoaderKITTI('/home/anastasiya/data/data_odometry_velodyne/', '00')
    pcd1 = loader.get_item(0)
    pcd2 = loader.get_item(1)

    o3d.visualization.draw_geometries([get_pcd_from_numpy(pcd1[0], color=[1, 0, 0]),
                                       get_pcd_from_numpy(pcd2[0], color=[0, 1, 0]).transform(T.T())])
