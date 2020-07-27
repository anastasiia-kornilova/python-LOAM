import numpy as np


class Loader:
    def length(self):
        pass

    def get_item(self, ind):
        pass

    def _get_scan_ids(self, pcd):
        pass

    def reorder_pcd(self, pcd):
        scan_start = np.zeros(self.N_SCANS, dtype=int)
        scan_end = np.zeros(self.N_SCANS, dtype=int)

        scan_ids = self._get_scan_ids(pcd)
        sorted_ind = np.argsort(scan_ids, kind='stable')
        sorted_pcd = pcd[sorted_ind]
        sorted_scan_ids = scan_ids[sorted_ind]

        elements, elem_cnt = np.unique(sorted_scan_ids, return_counts=True)

        start = 0
        for ind, cnt in enumerate(elem_cnt):
            scan_start[ind] = start
            start += cnt
            scan_end[ind] = start

        laser_cloud = np.hstack((sorted_pcd, sorted_scan_ids.reshape((-1, 1))))
        return laser_cloud, scan_start, scan_end
