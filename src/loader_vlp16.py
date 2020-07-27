import numpy as np
import os

from loader import Loader


class LoaderVLP16(Loader):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.pcds_list = os.listdir(self.folder_path)
        self.pcds_list.sort()

    def length(self):
        return len(self.pcds_list)

    def get_item(self, ind):
        path = os.path.join(self.folder_path, self.pcds_list[ind])
        pcd = np.load(path)[:, :3]
        return pcd
