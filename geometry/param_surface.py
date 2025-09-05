import numpy as np
import sympy as sym
from .param_curve import phi, psi
from scipy.spatial import cKDTree

a,b,c = sym.symbols('x y z')
class ParamSurface:
    '''
        Parametric surface of revolution, defined by a profile curve r(phi), z(phi).
        The surface is given by:
            S(theta, phi) = ( r(phi)*cos(theta), r(phi)*sin(theta), z(phi) )
        where theta in [0, 2*pi), phi in [phi_min, phi_max].
        The profile curve is assumed to be given in a parametric form:
            r = r(phi), z = z(phi)
        with phi in [phi_min, phi_max].
        
        param: list of 3 symbolic expressions [x(phi, psi), y(phi, psi), z(phi, psi)]
    '''
    def __init__(self, params, phi_lim=(0,2*np.pi), psi_lim=(0,2*np.pi),num_points=10000):
        # Param: 2d list of symbolic expression
        self.x_param, self.y_param, self.z_param = params
        self.phi_lim = phi_lim
        self.psi_lim = psi_lim
        func = sym.lambdify((phi, psi), params)
        self.param_np = lambda x1,y1: np.squeeze(func(x1,y1)).T
        # Jacobian 矩阵 (3x2): 第一行是对phi求导，第二行是对psi求导
        self.jacobian = sym.Matrix([[f.diff(var) for var in (phi, psi)] for f in params])
        self.jacobian_func = sym.lambdify((phi, psi), self.jacobian)
        
        # 定义目标函数 f = (x-a)^2 + (y-b)^2 + (z-c)^2，其中a,b,c是符号变量
        f = (self.x_param - a)**2 + (self.y_param - b)**2 + (self.z_param - c)**2
        # Hessian 矩阵 (2x2)
        self.hessian = sym.hessian(f, [phi, psi])
        self.hessian_func = sym.lambdify((phi, psi, a, b, c), self.hessian, 'numpy')
        self.build_tree(num_points)
    
    def build_tree(self,num_points):
        self.phi_samples, self.psi_samples, self.points_3d = self.sample_points_on_surface(num_points)
        self.tree = cKDTree(self.points_3d)
        
    def proj_param(self, query_points, max_iter=10, tol=1e-6):
        """
        Project query points onto the parametric surface using k-d tree for initial guesses
        and vectorized Newton's method for refinement.
        
        Args:
            query_points: np.ndarray of shape (n, 3), query points in 3D space
            max_iter: Maximum number of Newton iterations
            tol: Tolerance for convergence (norm of parameter update)
        
        Returns:
            tuple: (phi_values, psi_values), arrays of shape (n,) with projected parameters
        """
        # Get initial guesses from k-d tree
        distances, indices = self.tree.query(query_points, k=1)
        phi_values = self.phi_samples[indices].copy()
        psi_values = self.psi_samples[indices].copy()

        # Track which points have converged
        n_points = query_points.shape[0]
        converged = np.zeros(n_points, dtype=bool)
        
        # Newton iterations
        for _ in range(max_iter):
            # Skip converged points
            if converged.all():
                break
                
            # Evaluate surface points s(phi, psi)
            s = self.param_np(phi_values, psi_values)  # Shape (n, 3)

            # Compute residuals (s - q)
            residuals = s - query_points  # Shape (n, 3)

            # Compute Jacobians for all points, ijk: k index; i: axis; j: derivative
            J = np.array(self.jacobian_func(phi_values, psi_values))  # Shape (n, 3, 2)

            # Compute gradient: 2 * J^T @ residual grad of the distance
            grad_f = 2 * np.einsum('jik,kj->ki', J, residuals)  # Shape (n, 2)

            hessian_f = self.hessian_func(phi_values,psi_values,query_points[:,0],
                                         query_points[:,1],query_points[:,2])

            hessian_f = np.transpose(hessian_f, (2, 0, 1))  # 形状 (4000, 2, 2)

            det_H = np.linalg.det(hessian_f) # 形状 (4000,)

            valid = np.abs(det_H) > 1e-6
            # Initialize updates
            delta = np.zeros((n_points, 2))

            # Compute Newton step for valid points
            if valid.any():
                # Solve H @ delta = -grad_f for valid points
                delta[valid] = np.linalg.solve(hessian_f[valid], -grad_f[valid])

            # Update parameters where not converged
            update_mask = (~converged) & (np.linalg.norm(delta, axis=1) > tol)
            phi_values[update_mask] += delta[update_mask, 0]
            psi_values[update_mask] += delta[update_mask, 1]

            # Clamp parameters to valid ranges
#             phi_values = np.clip(phi_values, self.phi_lim[0], self.phi_lim[1])
#             psi_values = np.clip(psi_values, self.psi_lim[0], self.psi_lim[1])

            # Update convergence status
            converged[update_mask] = np.linalg.norm(delta[update_mask], axis=1) < tol

        # 计算最终曲面点
        points = self.param_np(phi_values, psi_values)

        if not max_iter == 0:
            # 记录未收敛点的容差信息
            tol_info = {
                'non_converged_indices': np.where(~converged)[0],
                'final_delta_norms': np.linalg.norm(delta[~converged], axis=1)
            }
        else:
            tol_info = {}
        return phi_values, psi_values, points, tol_info
    
    
    def get_coarse_param(self, query_points):
        distances, indices = self.tree.query(query_points, k=1) 
        return distances, indices
    
    def sample_points_on_surface(self, n_points):
        """
        根据 Jacobian 面积元素在参数平面上加权采样点，并映射到曲面上。
        输入:
            n_points: 采样点总数
            phi_range: (phi_min, phi_max)
            psi_range: (psi_min, psi_max)
        
        输出:
            phi_samples, psi_samples: 参数平面上的采样
            points_3d: 映射到曲面上的 3D 坐标
        """
        # 1. 在参数平面上生成粗网格
        N_grid = int(np.sqrt(5 * n_points))  # 粗略，网格比点数多几个
        phi_lin = np.linspace(self.phi_lim[0], self.phi_lim[1], N_grid)
        psi_lin = np.linspace(self.psi_lim[0], self.psi_lim[1], N_grid)
        Phi, Psi = np.meshgrid(phi_lin, psi_lin, indexing='ij')
        Phi_flat = Phi.ravel()
        Psi_flat = Psi.ravel()

        # 2. 计算每个网格点的面积元素
        J_vals = self.jacobian_func(Phi_flat, Psi_flat)   # shape (N, 3, 2) 或 (N,6)展开的情况
        # 如果 lambdify 返回 (N,6) 扁平展开，需要 reshape
        if J_vals.ndim == 2 and J_vals.shape[1] == 6:
            J_vals = J_vals.reshape(-1,3,2)
        J_vals = np.moveaxis(J_vals, 2, 0) 
        
        # 面积元素
        self.dS = np.sqrt(np.linalg.det(np.matmul(np.transpose(J_vals, (0,2,1)), J_vals)))  
        # sqrt(det(J^T J)), shape (N,)
        # 3. 面积加权采样（用CDF逆变换）
        eps = 1e-16
        weights = self.dS / np.sum(self.dS+eps)
        cdf = np.cumsum(weights)
        
        # 从均匀 [0,1] 采样
        u = np.random.rand(n_points)
        indices = np.searchsorted(cdf, u)
        
        phi_samples = Phi_flat[indices]
        psi_samples = Psi_flat[indices]
        
        points_3d = self.param_np(phi_samples, psi_samples)  # 或用你对应的曲面函数
        return phi_samples, psi_samples, points_3d