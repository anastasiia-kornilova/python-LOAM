import numpy as np
import mrob
import open3d as o3d

from optimizer import LOAMOptimizer
from utils import get_pcd_from_numpy


class Mapper:
    COVARIANCE_CNT = 5
    EDGE_VOXEL_SIZE = 0.2
    SURFACE_VOXEL_SIZE = 0.4

    def __init__(self):
        self.init = False
        self.corners = None
        self.surfaces = None
        self.position = np.eye(4)
        self.aligned_pcds = []
        self.all_edges = None
        self.all_surfaces = None
        self.cnt = 0

    def undistort(self, pcd, T):
        N_GROUPS = 30
        global_T = mrob.geometry.SE3(T).Ln()
        groups = np.array_split(pcd, N_GROUPS)
        pcds_cut = []
        for i in range(N_GROUPS):
            pcds_cut.append(mrob.geometry.SE3((i + 1) / N_GROUPS * global_T).transform_array(groups[i][:, :3]))

        reformed = np.vstack(pcds_cut)
        return reformed

    def append_undistorted(self, pcd, T, edge_points, surface_points, vis=False):
        self.cnt += 1
        if not self.init:
            self.init = True
            pcd = get_pcd_from_numpy(pcd)
            pcd.paint_uniform_color([0, 0, 1])
            self.aligned_pcds.append(pcd)
            self.all_edges = get_pcd_from_numpy(np.vstack(edge_points))
            self.all_surfaces = get_pcd_from_numpy(np.vstack(surface_points))
        else:
            prior_position = T @ self.position
            edge_points = np.asarray(self.filter_pcd(get_pcd_from_numpy(np.vstack(edge_points)), 'edge').points)
            surface_points = np.asarray(self.filter_pcd(get_pcd_from_numpy(np.vstack(surface_points)), 'surface').points)
            restored_pcd = np.vstack(pcd)
            transformed_pcd = mrob.geometry.SE3(prior_position).transform_array(restored_pcd)
            transformed_edge_points = mrob.geometry.SE3(prior_position).transform_array(np.vstack(edge_points))
            edges_kdtree = o3d.geometry.KDTreeFlann(self.all_edges)

            surfaces_kdtree = o3d.geometry.KDTreeFlann(self.all_surfaces)
            transformed_surface_points = mrob.geometry.SE3(prior_position).transform_array(np.vstack(surface_points))

            edges = []
            edge_A = []
            edge_B = []
            for ind in range(len(edge_points)):
                point = transformed_edge_points[ind]
                _, idx, dists = edges_kdtree.search_knn_vector_3d(point, self.COVARIANCE_CNT)
                if np.max(np.linalg.norm(np.asarray(self.all_edges.points)[idx] - point, axis=1)) < 1:
                    status, point_a, point_b = self.is_edge(np.asarray(self.all_edges.points)[idx])
                    if status:
                        edges.append(point)
                        edge_A.append(point_a)
                        edge_B.append(point_b)

            surfaces = []
            surface_A = []
            surface_B = []
            for ind in range(len(surface_points)):
                point = transformed_surface_points[ind]
                _, idx, dists = surfaces_kdtree.search_knn_vector_3d(point, self.COVARIANCE_CNT)
                if np.max(np.linalg.norm(np.asarray(self.all_surfaces.points)[idx] - point, axis=1)) < 1:
                    status, norm, norm_reversed = self.is_surface(np.asarray(self.all_surfaces.points)[idx])
                    if status:
                        surfaces.append(point)
                        surface_A.append(norm)
                        surface_B.append(norm_reversed)

            if len(edges) > 0 and len(surfaces) > 0:
                optimizer = LOAMOptimizer((np.vstack(edges), np.vstack(edge_A), np.vstack(edge_B)),
                                          (np.vstack(surfaces), np.vstack(surface_A), np.vstack(surface_B)))
                T = optimizer.optimize_2()
                self.position = T.T() @ prior_position
                self.aligned_pcds.append(get_pcd_from_numpy(T.transform_array(transformed_pcd)))
                if self.cnt % 3 == 0:
                    self.all_edges += get_pcd_from_numpy(T.transform_array(transformed_edge_points))
                    self.all_surfaces += get_pcd_from_numpy(T.transform_array(transformed_surface_points))
                    self.all_edges = self.filter_pcd(self.all_edges, 'edge')
                    self.all_surfaces = self.filter_pcd(self.all_surfaces, 'surface')
            else:
                self.aligned_pcds.append(get_pcd_from_numpy(transformed_pcd))
                self.all_edges += get_pcd_from_numpy(transformed_edge_points)
                self.all_surfaces += get_pcd_from_numpy(transformed_surface_points)


            if vis:
                o3d.visualization.draw_geometries(self.aligned_pcds)

    def is_edge(self, surrounded_points):
        assert surrounded_points.shape[0] == self.COVARIANCE_CNT
        centroid = np.sum(surrounded_points, axis=0) / self.COVARIANCE_CNT
        covariance_mat = np.zeros((3, 3))
        for i in range(self.COVARIANCE_CNT):
            diff = (surrounded_points[i] - centroid).reshape((3, 1))
            covariance_mat += diff @ diff.T

        v, e = np.linalg.eig(covariance_mat)
        sorted_v_ind = np.argsort(v)
        sorted_v = v[sorted_v_ind]
        sorted_e = e[sorted_v_ind]
        unit_direction = sorted_e[:, 2]

        if sorted_v[2] > 3 * sorted_v[1]:
            point_a = 0.1 * unit_direction + centroid
            point_b = -0.1 * unit_direction + centroid
            return True, point_a, point_b
        else:
            return False, None, None

    def is_surface(self, surrounded_points):
        mat_A0 = surrounded_points
        mat_B0 = -np.ones((self.COVARIANCE_CNT, ))

        norm = np.linalg.lstsq(mat_A0, mat_B0)[0]
        norm_reversed = 1 / np.linalg.norm(norm)
        norm /= np.linalg.norm(norm)

        plane_valid = True
        for j in range(self.COVARIANCE_CNT):
            if np.abs(np.dot(norm, surrounded_points[j]) + norm_reversed) > 0.2:
                plane_valid = False
                break

        if plane_valid:
            return True, norm, norm_reversed
        else:
            return False, None, None

    def filter_pcd(self, pcd, type):
        if type == 'edge':
            return pcd.voxel_down_sample(self.EDGE_VOXEL_SIZE)
        elif type == 'surface':
            return pcd.voxel_down_sample(self.SURFACE_VOXEL_SIZE)
