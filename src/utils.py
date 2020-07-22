import numpy as np
import open3d as o3d


def get_pcd_from_numpy(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    return pcd


def matrix_dot_product(A, B):
    """
    Fast way to calculate dot product of every row in two matrices A and B
    Returns array like:
    [ np.dot(A[0], B[0]),
    np.dot(A[1], B[1]),
    np.dot(A[2], B[2]),
    ... ,
    np.dot(A[M - 1], B[M - 1])]

    :param A: MxN numpy array
    :param B: MXN numpy array
    :return: Mx1 numpy array
    """
    assert A.shape == B.shape
    return np.einsum('ij,ij->i', A, B)
