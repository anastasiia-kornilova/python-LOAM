import math
import numpy as np
import os

from loader import Loader


class LoaderVLP16(Loader):
    def __init__(self, folder_path):
        self.N_SCANS = 16
        self.folder_path = folder_path
        self.pcds_list = os.listdir(self.folder_path)
        self.pcds_list.sort()

    def length(self):
        return len(self.pcds_list)

    def get_item(self, ind):
        path = os.path.join(self.folder_path, self.pcds_list[ind])
        pcd = np.load(path)[:, :3]
        return self.reorder_pcd(pcd)

    def _get_scan_ids(self, pcd):
        angles_rad = np.arctan(np.divide(pcd[:, 2], np.sqrt(pcd[:, 0] * pcd[:, 0] + pcd[:, 1] * pcd[:, 1])))
        angles_deg = angles_rad * 180 / math.pi
        scan_ids = ((angles_deg + self.N_SCANS - 1) / 2 + 0.5).astype(int)

        return scan_ids
