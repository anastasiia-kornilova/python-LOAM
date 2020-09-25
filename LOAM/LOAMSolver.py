import mrob
import numpy as np
import sys


# Optimization solver for LOAM task
# * Uses pre-calculated Jacobians for system of equations based on SE(3) Lie algebra
# * Uses M-estimator based on Truncated Least Squares to deal with outlier
class LOAMSolver:
    def __init__(self, use_estimators=False, region_rate=0.7):
        self.alpha = 1
        self.max_iter = 1e5
        self.error_region = 1e-5

        # M-estimators
        self.resid_sigmas_edge = []
        self.resid_sigmas_planes = []
        self.use_estimators = use_estimators
        self.initial_sigma_scale = 3
        self.sigma_coef = region_rate

        # Visualization
        self.visualize = False

    def fit(self, edge_corresp, plane_corresp, initial_T=None):
        T = mrob.geometry.SE3(np.zeros(6)) if not initial_T else initial_T

        iter_cnt = 0

        initial_err = self.cost_function(edge_corresp, plane_corresp, T)
        errors = [initial_err + 1, initial_err]

        inlier_errors = [initial_err + 1, initial_err]
        outlier_errors = []
        inliers_edge_perc = []
        inliers_plane_perc = []
        inlier_edge_errors = []
        inlier_plane_errors = []

        if self.use_estimators:
            for corresp in edge_corresp:
                r, _ = self.r(corresp, T, corresp_type='edge')
                self.resid_sigmas_edge.append(self.initial_sigma_scale * self.e(r))

            for corresp in plane_corresp:
                r, _ = self.r(corresp, T, corresp_type='plane')
                self.resid_sigmas_planes.append(self.initial_sigma_scale * self.e(r))

        prev_rel_inlier_err = 100
        rel_inlier_err = 0

        prev_inliers_plane = []
        prev_inliers_edge = []

        while iter_cnt < self.max_iter and abs(prev_rel_inlier_err - rel_inlier_err) > self.error_region:
            prev_rel_inlier_err = rel_inlier_err
            jac, hess, edge_inliers, plane_inliers = self.derivatives(edge_corresp, plane_corresp, T)
            inliers_edge_perc.append(len(edge_inliers) / len(edge_corresp))
            inliers_plane_perc.append(len(plane_inliers) / len(plane_corresp))

            if np.linalg.cond(hess) < 1 / sys.float_info.epsilon:
                T.update_lhs(-self.alpha * np.linalg.inv(hess) @ jac.T)
            else:
                break

            prev_inliers_edge = edge_inliers
            prev_inliers_plane = plane_inliers
            errors.append(self.cost_function(edge_corresp, plane_corresp, T))

            edge_outliers = [x for x in range(len(edge_corresp)) if x not in edge_inliers]
            plane_outliers = [x for x in range(len(plane_corresp)) if x not in plane_inliers]
            outlier_errors.append(self.cost_function_by_ind(edge_corresp, plane_corresp, T,
                                                            edge_outliers, plane_outliers))

            inlier_errors.append(self.cost_function_by_ind(edge_corresp, plane_corresp, T,
                                                           edge_inliers, plane_inliers))

            inlier_edge_errors.append(self.cost_function_by_ind(edge_corresp, plane_corresp, T,
                                                                edge_inliers, []))

            inlier_plane_errors.append(self.cost_function_by_ind(edge_corresp, plane_corresp, T,
                                                                 [], plane_inliers))

            rel_inlier_err = inlier_errors[-1] / (len(edge_inliers) + len(plane_inliers))

            iter_cnt += 1

            if self.use_estimators:
                self.resid_sigmas_edge = [self.sigma_coef * x if x > self.error_region else x for x
                                          in self.resid_sigmas_edge]
                self.resid_sigmas_planes = [self.sigma_coef * x if x > self.error_region else x for x
                                            in self.resid_sigmas_planes]

        return T, inlier_errors, prev_inliers_edge, prev_inliers_plane

    def m_estimator_condition(self, r, resid_sigma):
        if self.e(r) < resid_sigma:
            return 1
        else:
            return 0

    def r(self, corresp, T, corresp_type=None):
        if corresp_type == 'edge':
            a = corresp[1]
            b = corresp[2]
            p = T.transform_array(corresp[0].reshape(1, 3)).reshape(3)

            r = self._skew(a - b) @ (a - p).reshape(3, 1) / np.linalg.norm(a - b)
            small_jac = self._skew(a - b) @ np.hstack((self._skew(p), -np.eye(3))) / np.linalg.norm(a - b)
            return r, small_jac
        elif corresp_type == 'plane':
            a = corresp[1]
            b = corresp[2]
            c = corresp[3]
            p = T.transform_array(corresp[0].reshape(1, 3)).reshape(3)

            cross_prod = self._skew(a - b) @ (a - c).reshape(3, 1)
            n = cross_prod / np.linalg.norm(cross_prod)
            r = n.reshape(1, 3) @ (a - p).reshape(3, 1)
            small_jac = n.reshape(1, 3) @ np.hstack((self._skew(p), -np.eye(3)))
            return r, small_jac
        else:
            raise NotSupportedType()

    def e(self, r):
        return (r.reshape(1, -1) @ r.reshape(-1, 1))[0, 0]

    def cost_function(self, edge_corresp, plane_corresp, T):
        err = 0

        for corresp in edge_corresp:
            r, _ = self.r(corresp, T, corresp_type='edge')
            err += self.e(r)

        for corresp in plane_corresp:
            r, _ = self.r(corresp, T, corresp_type='plane')
            err += self.e(r)

        return err

    def cost_function_by_ind(self, edge_corresp, plane_corresp, T, edge_ind, plane_ind):
        err = 0
        for i in edge_ind:
            r, _ = self.r(edge_corresp[i], T, corresp_type='edge')
            err += self.e(r)

        for i in plane_ind:
            r, _ = self.r(plane_corresp[i], T, corresp_type='plane')
            err += self.e(r)

        return err

    def derivatives(self, edge_corresp, plane_corresp, T):
        jac = np.zeros((1, 6))
        hess = np.zeros((6, 6))

        edge_inliers = []
        for ind, corresp in enumerate(edge_corresp):
            r, jac_i = self.r(corresp, T, corresp_type='edge')
            if not self.use_estimators or (
                    self.use_estimators and self.m_estimator_condition(r, self.resid_sigmas_edge[ind])):
                jac += r.T @ jac_i
                hess += jac_i.T @ jac_i
                edge_inliers.append(ind)

        plane_inliers = []
        for ind, corresp in enumerate(plane_corresp):
            r, jac_i = self.r(corresp, T, corresp_type='plane')
            if not self.use_estimators or (
                    self.use_estimators and self.m_estimator_condition(r, self.resid_sigmas_planes[ind])):
                jac += r.T @ jac_i
                hess += jac_i.T @ jac_i
                plane_inliers.append(ind)

        return jac, hess, edge_inliers, plane_inliers

    def _skew(self, x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
