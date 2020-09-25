import mrob
import numpy as np
import open3d as o3d

from feature_extractor import FeatureExtractor
from optimizer import LOAMOptimizer
from utils import get_pcd_from_numpy, matrix_dot_product


class OdometryEstimator:
    DISTANCE_SQ_THRESHOLD = 1
    SCAN_VICINITY = 2.5

    def __init__(self):
        self.extractor = FeatureExtractor()

        self.inited = False
        self.last_less_sharp_points = None
        self.last_less_flat_points = None
        self.last_position = np.eye(4)

    def append_pcd(self, pcd):
        sharp_points, less_sharp_points, flat_points, less_flat_points = self.extractor.extract_features(pcd[0], pcd[1],
                                                                                                         pcd[2])
        T = None
        if not self.inited:
            self.inited = True
            T = np.zeros(6)
        else:
            edge_corresp = self.find_edge_correspondences(sharp_points)
            surface_corresp = self.find_surface_correspondences(flat_points, pcd)
            optimizer = LOAMOptimizer(edge_corresp, surface_corresp)
            T = optimizer.optimize()
            import utils
            surf = np.vstack((surface_corresp[1], surface_corresp[2], surface_corresp[3]))
            keypoints = utils.get_pcd_from_numpy(surf)
            keypoints.paint_uniform_color([0, 1, 0])
            pcd = utils.get_pcd_from_numpy(mrob.geometry.SE3(T).transform_array(pcd[0]))
            pcd.paint_uniform_color([0, 0, 1])
            orig = utils.get_pcd_from_numpy(surface_corresp[0])
            orig.paint_uniform_color([1, 0, 0])
            # o3d.visualization.draw_geometries([pcd, keypoints, orig])

        self.last_less_sharp_points = np.vstack(less_sharp_points)
        x = get_pcd_from_numpy(np.vstack(less_flat_points))
        y = np.vstack(less_flat_points)[:, 3].reshape((-1, 1)) / 64
        x.colors = o3d.utility.Vector3dVector(np.hstack((y, y, y)))
        x = x.voxel_down_sample(0.1)
        self.last_less_flat_points = np.hstack((np.asarray(x.points), 64 * np.asarray(x.colors)[:, 0].reshape((-1, 1))))
        scan_ids = self.last_less_flat_points[:, 3]
        sorted_ind = np.argsort(scan_ids, kind='stable')
        self.last_less_flat_points = self.last_less_flat_points[sorted_ind]
        self.last_position = mrob.geometry.SE3(T).T() @ self.last_position

        return mrob.geometry.SE3(T).T(), self.last_less_flat_points, self.last_less_flat_points

    def find_edge_correspondences(self, sharp_points):
        corners_cnt = len(sharp_points)

        edge_points = []
        edge_1 = []
        edge_2 = []
        less_sharp_points_tree = o3d.geometry.KDTreeFlann(get_pcd_from_numpy(self.last_less_sharp_points))
        for i in range(corners_cnt):
            point_sel = sharp_points[i]
            _, idx, dist = less_sharp_points_tree.search_knn_vector_3d(point_sel[:3], 1)
            min_point_ind_2 = -1
            if dist[0] < self.DISTANCE_SQ_THRESHOLD:
                closest_point_ind = idx[0]
                min_point_sq_dist_2 = self.DISTANCE_SQ_THRESHOLD
                closest_point_scan_id = self.last_less_sharp_points[closest_point_ind][3]

                dist_to_sel_point = matrix_dot_product((self.last_less_sharp_points[:, :3] - point_sel[:3]),
                                                       (self.last_less_sharp_points[:, :3] - point_sel[:3]))

                for j in range(closest_point_ind + 1, len(self.last_less_sharp_points)):
                    if self.last_less_sharp_points[j][3] <= closest_point_scan_id:
                        continue
                    if self.last_less_sharp_points[j][3] > closest_point_scan_id + self.SCAN_VICINITY:
                        break

                    point_sq_dist = dist_to_sel_point[j]
                    if point_sq_dist < min_point_sq_dist_2:
                        min_point_sq_dist_2 = point_sq_dist
                        min_point_ind_2 = j

                for j in range(closest_point_ind - 1, -1, -1):
                    if self.last_less_sharp_points[j][3] >= closest_point_scan_id:
                        continue
                    if self.last_less_sharp_points[j][3] < closest_point_scan_id - self.SCAN_VICINITY:
                        break

                    point_sq_dist = dist_to_sel_point[j]
                    if point_sq_dist < min_point_sq_dist_2:
                        min_point_sq_dist_2 = point_sq_dist
                        min_point_ind_2 = j

                if min_point_ind_2 >= 0:
                    edge_points.append(point_sel)
                    edge_1.append(self.last_less_sharp_points[closest_point_ind])
                    edge_2.append(self.last_less_sharp_points[min_point_ind_2])

        edge_points = np.vstack(edge_points)[:, :3]
        edge_1 = np.vstack(edge_1)[:, :3]
        edge_2 = np.vstack(edge_2)[:, :3]

        return edge_points, edge_1, edge_2

    def find_surface_correspondences(self, flat_points, pcd):
        surface_cnt = len(flat_points)
        print('Surface count: ', surface_cnt)

        surface_points = []
        surface_1 = []
        surface_2 = []
        surface_3 = []

        less_flat_points_tree = o3d.geometry.KDTreeFlann(get_pcd_from_numpy(self.last_less_flat_points))
        for i in range(surface_cnt):
            point_sel = flat_points[i]
            _, idx, dist = less_flat_points_tree.search_knn_vector_3d(point_sel[:3], 1)
            min_point_ind_2 = -1
            min_point_ind_3 = -1

            dist_to_sel_point = matrix_dot_product((self.last_less_flat_points[:, :3] - point_sel[:3]),
                                                   (self.last_less_flat_points[:, :3] - point_sel[:3]))

            closest_point_ind = idx[0]
            v = self.last_less_flat_points[closest_point_ind][:3] - point_sel[:3]
            dist = np.dot(v, v)
            if dist < self.DISTANCE_SQ_THRESHOLD:
                closest_point_scan_id = self.last_less_flat_points[closest_point_ind][3]
                min_point_sq_dist_2 = self.DISTANCE_SQ_THRESHOLD
                min_point_sq_dist_3 = self.DISTANCE_SQ_THRESHOLD
                for j in range(closest_point_ind + 1, len(self.last_less_flat_points)):
                    if self.last_less_flat_points[j][3] > closest_point_scan_id + self.SCAN_VICINITY:
                        break

                    point_sq_dist = dist_to_sel_point[j]

                    if self.last_less_flat_points[j][3] <= closest_point_scan_id \
                            and point_sq_dist < min_point_sq_dist_2:
                        min_point_sq_dist_2 = point_sq_dist
                        min_point_ind_2 = j
                    elif self.last_less_flat_points[j][3] > closest_point_scan_id \
                            and point_sq_dist < min_point_sq_dist_3:
                        min_point_sq_dist_3 = point_sq_dist
                        min_point_ind_3 = j

                for j in range(closest_point_ind - 1, -1, -1):
                    if self.last_less_flat_points[j][3] < closest_point_scan_id - self.SCAN_VICINITY:
                        break

                    point_sq_dist = dist_to_sel_point[j]

                    if self.last_less_flat_points[j][3] >= closest_point_scan_id \
                            and point_sq_dist < min_point_sq_dist_2:
                        min_point_sq_dist_2 = point_sq_dist
                        min_point_ind_2 = j
                    elif self.last_less_flat_points[j][3] < closest_point_scan_id \
                            and point_sq_dist < min_point_sq_dist_3:
                        min_point_sq_dist_3 = point_sq_dist
                        min_point_ind_3 = j

                if min_point_ind_2 >= 0 and min_point_ind_3 >= 0:
                    surface_points.append(point_sel)
                    surface_1.append(self.last_less_flat_points[closest_point_ind])
                    surface_2.append(self.last_less_flat_points[min_point_ind_2])
                    surface_3.append(self.last_less_flat_points[min_point_ind_3])

        surface_points = np.vstack(surface_points)
        surface_1 = np.vstack(surface_1)
        surface_2 = np.vstack(surface_2)
        surface_3 = np.vstack(surface_3)

        print('output: ', surface_points.shape[0])
        import utils
        # import open3d as 0o3d
        ind = surface_1[:, 3] > 0
        surf = np.vstack((surface_1[ind], surface_2[ind], surface_3[ind]))
        keypoints = utils.get_pcd_from_numpy(surf)
        keypoints.paint_uniform_color([0, 1, 0])
        pcd = utils.get_pcd_from_numpy(pcd[0])
        pcd.paint_uniform_color([0, 0, 1])
        orig = utils.get_pcd_from_numpy(surface_points[ind])
        orig.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd, keypoints, orig])

        return surface_points[ind][:, :3], surface_1[ind][:, :3], surface_2[ind][:, :3], surface_3[ind][:, :3]
