import numpy as np
import os

from loader import Loader


class LoaderKITTI(Loader):
    def __init__(self, dataset_path, sequence):
        self.folder_path = os.path.join(dataset_path, 'dataset', 'sequences', sequence, 'velodyne')
        self.pcds_list = os.listdir(self.folder_path)
        self.pcds_list.sort()

    def length(self):
        return len(self.pcds_list)

    def get_item(self, ind):
        path = os.path.join(self.folder_path, self.pcds_list[ind])
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
        return pcd
