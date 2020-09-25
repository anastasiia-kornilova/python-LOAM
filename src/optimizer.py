import mrob
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import leastsq

from utils import matrix_dot_product


class LOAMOptimizer:
    def __init__(self, edge_factors, surface_factors):
        self.edge_factors = edge_factors
        self.surface_factors = surface_factors

    def optimize(self):
        opt_solution = least_squares(self.resid_function, np.zeros((6,)), xtol=1e-10, loss='huber')
        # opt_solution = leastsq(self.resid_function, np.zeros((6,)))
        T = mrob.geometry.SE3(opt_solution.x)
        print('ERROR', np.linalg.norm(self.resid_function(opt_solution.x)))
        gt = np.array([[ 0.99999435, -0.00313932, -0.00119973 , 0.6821656 ],
                       [ 0.00313629,  0.9999919,  -0.00252248,  0.00292633],
                       [ 0.00120764,  0.00251871,  0.9999961,   0.00632314],
                       [ 0.,          0. ,         0.,          1.,        ]])
        T_true = mrob.geometry.SE3(gt)
        print('GT ERROR', np.linalg.norm(self.resid_function(T_true.Ln())))
        return T

    def resid_function(self, x):
        T = mrob.geometry.SE3(x)

        aligned_edges = T.transform_array(self.edge_factors[0])
        nu = np.linalg.norm(np.cross(aligned_edges - self.edge_factors[1],
                                     aligned_edges - self.edge_factors[2]), axis=1)
        de = np.linalg.norm(self.edge_factors[1] - self.edge_factors[2], axis=1)
        edge_resid = np.divide(nu, de)

        aligned_surface = T.transform_array(self.surface_factors[0])
        normal = np.cross(self.surface_factors[1] - self.surface_factors[2],
                          self.surface_factors[1] - self.surface_factors[3])
        normal /= np.linalg.norm(normal, axis=1).reshape((-1, 1))
        surface_resid = np.abs(matrix_dot_product(aligned_surface - self.surface_factors[1], normal))

        resid = np.concatenate((edge_resid, surface_resid))
        return resid

    def optimize_2(self):
        opt_solution = least_squares(self.resid_function_2, np.zeros((6,)), loss='huber')
        T = mrob.geometry.SE3(opt_solution.x)
        return T

    def resid_function_2(self, x):
        T = mrob.geometry.SE3(x)

        aligned_edges = T.transform_array(self.edge_factors[0])
        nu = np.linalg.norm(np.cross(aligned_edges - self.edge_factors[1],
                                     aligned_edges - self.edge_factors[2]), axis=1)
        de = np.linalg.norm(self.edge_factors[1] - self.edge_factors[2], axis=1)
        edge_resid = np.divide(nu, de)

        aligned_surfaces = T.transform_array(self.surface_factors[0])
        surface_resid = matrix_dot_product(aligned_surfaces, self.surface_factors[1]) + \
                        self.surface_factors[2].reshape((-1,))

        resid = np.concatenate((edge_resid, surface_resid))
        return resid
