import math
import numpy as np


class FeatureExtractor:
    N_SCANS = 16
    N_SEGMENTS = 6

    PICKED_NUM_LESS_SHARP = 20
    PICKED_NUM_SHARP = 2
    PICKED_NUM_FLAT = 4
    FILTER_SIZE = 5
    SURFACE_CURVATURE_THRESHOLD = 0.1
    FEATURES_REGION = 5

    def extract_features(self, pcd_numpy):
        keypoints_sharp = []
        keypoints_less_sharp = []
        keypoints_flat = []
        keypoints_less_flat = []

        laser_cloud, scan_start, scan_end = self.reorder_pcd(pcd_numpy)
        cloud_curvatures = self.get_curvatures(laser_cloud)

        cloud_label = np.zeros((pcd_numpy.shape[0]))
        cloud_neighbors_picked = np.zeros((pcd_numpy.shape[0]))
        for i in range(self.N_SCANS):
            if scan_end[i] - scan_start[i] < self.N_SEGMENTS:
                continue

            for j in range(self.N_SEGMENTS):
                sp = scan_start[i] + (scan_end[i] - scan_start[i]) * j // self.N_SEGMENTS
                ep = scan_start[i] + (scan_end[i] - scan_start[i]) * (j + 1) // self.N_SEGMENTS - 1
                segments_curvatures = cloud_curvatures[sp:ep + 1]
                sort_indices = np.argsort(segments_curvatures)

                largest_picked_num = 0
                for k in reversed(range(ep - sp)):
                    ind = sort_indices[k] + sp

                    if cloud_neighbors_picked[ind] == 0 and cloud_curvatures[ind] > self.SURFACE_CURVATURE_THRESHOLD:
                        largest_picked_num += 1
                        if largest_picked_num <= self.PICKED_NUM_SHARP:
                            keypoints_sharp.append(laser_cloud[ind])
                            keypoints_less_sharp.append(laser_cloud[ind])
                            cloud_label[ind] = 2
                        elif largest_picked_num <= self.PICKED_NUM_LESS_SHARP:
                            keypoints_less_sharp.append(laser_cloud[ind])
                            cloud_label[ind] = 1
                        else:
                            break

                        cloud_neighbors_picked = self.mark_as_picked(laser_cloud, cloud_neighbors_picked, ind)

                smallest_picked_num = 0
                for k in range(ep - sp):
                    ind = sort_indices[k] + sp

                    if cloud_neighbors_picked[ind] == 0 and cloud_curvatures[ind] < self.SURFACE_CURVATURE_THRESHOLD:
                        smallest_picked_num += 1
                        cloud_label[ind] = -1
                        keypoints_flat.append(laser_cloud[ind])

                        if smallest_picked_num >= self.PICKED_NUM_FLAT:
                            break

                        cloud_neighbors_picked = self.mark_as_picked(laser_cloud, cloud_neighbors_picked, ind)

                for k in range(sp, ep + 1):
                    if cloud_label[k] <= 0:
                        keypoints_less_flat.append(laser_cloud[k])

        return keypoints_sharp, keypoints_less_sharp, keypoints_flat, keypoints_less_flat

    def get_scan_ids(self, pcd):
        angles_rad = np.arctan(np.divide(pcd[:, 2], np.sqrt(pcd[:, 0] * pcd[:, 0] + pcd[:, 1] * pcd[:, 1])))
        angles_deg = angles_rad * 180 / math.pi
        scan_ids = ((angles_deg + self.N_SCANS - 1) / 2 + 0.5).astype(int)

        return scan_ids

    def get_curvatures(self, pcd):
        coef = [1, 1, 1, 1, 1, -10, 1, 1, 1, 1, 1]
        assert len(coef) == 2 * self.FILTER_SIZE + 1
        discr_diff = lambda x: np.convolve(x, coef, 'valid')
        x_diff = discr_diff(pcd[:, 0])
        y_diff = discr_diff(pcd[:, 1])
        z_diff = discr_diff(pcd[:, 2])
        curvatures = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
        curvatures = np.pad(curvatures, self.FILTER_SIZE)
        return curvatures

    def reorder_pcd(self, pcd):
        scan_start = np.zeros(self.N_SCANS, dtype=int)
        scan_end = np.zeros(self.N_SCANS, dtype=int)

        scan_ids = self.get_scan_ids(pcd)
        sorted_ind = np.argsort(scan_ids, kind='stable')
        sorted_pcd = pcd[sorted_ind]
        sorted_scan_ids = scan_ids[sorted_ind]

        elements, elem_cnt = np.unique(sorted_scan_ids, return_counts=True)

        start = 0
        for ind, cnt in enumerate(elem_cnt):
            scan_start[ind] = start + self.FILTER_SIZE
            start += cnt
            scan_end[ind] = start - self.FILTER_SIZE - 1

        laser_cloud = np.hstack((sorted_pcd, sorted_scan_ids.reshape((-1, 1))))
        return laser_cloud, scan_start, scan_end

    def mark_as_picked(self, laser_cloud, cloud_neighbors_picked, ind):
        cloud_neighbors_picked[ind] = 1

        diff_all = laser_cloud[ind - self.FEATURES_REGION + 1:ind + self.FEATURES_REGION + 2] \
                   - laser_cloud[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION + 1]

        sq_dist = np.einsum('ij,ij->i', diff_all[:, :3], diff_all[:, :3])

        for l in range(1, self.FEATURES_REGION + 1):
            if sq_dist[l + self.FEATURES_REGION] > 0.05:
                break
            cloud_neighbors_picked[ind + l] = 1

        for l in range(-self.FEATURES_REGION, 0):
            if sq_dist[l + self.FEATURES_REGION] > 0.05:
                break
            cloud_neighbors_picked[ind + l] = 1

        return cloud_neighbors_picked
