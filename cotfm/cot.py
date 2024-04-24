import warnings
import torch
import numpy as np
import ot as pot

from torchfm.optimal_transport import OTPlanSampler

# Based on code from the torchcfm package


# Gavin: TODO Documentation
class FullCOTPlanSampler(OTPlanSampler):
    def __init__(
        self,
        condition_coordinates: list,
        eps: float,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        warn: bool = True,
        numitermax: int = 100000
    ) -> None:
        """Initialize the COTPlanSampler class.

        Parameters
        ----------
        eps: float
            scalar weighting in relaxed COT problem
        method: str
            choose which optimal transport solver you would like to use.
            Currently supported are ["exact", "sinkhorn", "unbalanced",
            "partial"] OT solvers.
        reg: float, optional
            regularization parameter to use for Sinkhorn-based iterative solvers.
        reg_m: float, optional
            regularization weight for unbalanced Sinkhorn-knopp solver.
        normalize_cost: bool, optional
            normalizes the cost matrix so that the maximum cost is 1. Helps
            stabilize Sinkhorn-based solvers. Should not be used in the vast
            majority of cases.
        warn: bool, optional
            if True, raises a warning if the algorithm does not converge
        """

        super().__init__(method, reg=reg, reg_m=reg_m, normalize_cost=normalize_cost, 
                         warn=warn, numitermax=numitermax)
        self.y_coords = condition_coordinates
        self.eps = eps

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        y_coords = self.y_coords
        u_coords = [i for i in range(x0.shape[1]) if i not in y_coords]

        u0, y0 = x0[:, u_coords], x0[:, y_coords]
        u1, y1 = x1[:, u_coords], x1[:, y_coords]
        
        u0, y0 = torch.atleast_2d(u0), torch.atleast_2d(y0)
        u1, y1 = torch.atleast_2d(u1), torch.atleast_2d(y1)

        if u0.dim() > 2:
            u0 = u0.reshape(u0.shape[0], -1)
        if y0.dim() > 2:
            y0 = y0.reshape(y0.shape[0], -1)
        if u1.dim() > 2:
            u1 = u1.reshape(u1.shape[0], -1)
        if y1.dim() > 2:
            y1 = y1.reshape(y1.shape[0], -1)

        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])

        C_u = torch.cdist(u0, u1) ** 2
        C_y = torch.cdist(y0, y1) ** 2
        M = C_y + self.eps * C_u


        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p
    
    def set_map(self, x0, x1):
        pi = self.get_map(x0, x1)
        self.cot_plan = pi
        _, self.coupling_index = np.nonzero(pi)

        if self.warn:
            n_data = pi.shape[0]
            nonzero_vals = pi[pi!=0]
            num_nonzero = len(nonzero_vals)
            if num_nonzero != n_data:
                warnings.warn('COT Plan is not one-to-one.')
            if not np.allclose(nonzero_vals, 1./n_data):
                warnings.warn('COT Plan is nonuniform.')
        
        return pi
    
    def sample_map(self, batch_size, replace=True):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        source_index = np.random.choice(self.cot_plan.shape[0], size=batch_size, replace=replace)
        target_index = self.coupling_index[source_index]
        return source_index, target_index

# Gavin: TODO Documentation
class BatchCOTPlanSampler(OTPlanSampler):
    def __init__(
        self,
        condition_coordinates: list,
        eps: float,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        warn: bool = True,
        numitermax: int = 100000
    ) -> None:
        """Initialize the COTPlanSampler class.

        Parameters
        ----------
        eps: float
            scalar weighting in relaxed COT problem
        method: str
            choose which optimal transport solver you would like to use.
            Currently supported are ["exact", "sinkhorn", "unbalanced",
            "partial"] OT solvers.
        reg: float, optional
            regularization parameter to use for Sinkhorn-based iterative solvers.
        reg_m: float, optional
            regularization weight for unbalanced Sinkhorn-knopp solver.
        normalize_cost: bool, optional
            normalizes the cost matrix so that the maximum cost is 1. Helps
            stabilize Sinkhorn-based solvers. Should not be used in the vast
            majority of cases.
        warn: bool, optional
            if True, raises a warning if the algorithm does not converge
        """

        super().__init__(method, reg=reg, reg_m=reg_m, normalize_cost=normalize_cost, 
                         warn=warn, numitermax=numitermax)
        self.y_coords = condition_coordinates
        self.eps = eps

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        y_coords = self.y_coords
        u_coords = [i for i in range(x0.shape[1]) if i not in y_coords]

        u0, y0 = x0[:, u_coords], x0[:, y_coords]
        u1, y1 = x1[:, u_coords], x1[:, y_coords]
        
        u0, y0 = torch.atleast_2d(u0), torch.atleast_2d(y0)
        u1, y1 = torch.atleast_2d(u1), torch.atleast_2d(y1)

        if u0.dim() > 2:
            u0 = u0.reshape(u0.shape[0], -1)
        if y0.dim() > 2:
            y0 = y0.reshape(y0.shape[0], -1)
        if u1.dim() > 2:
            u1 = u1.reshape(u1.shape[0], -1)
        if y1.dim() > 2:
            y1 = y1.reshape(y1.shape[0], -1)

        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])

        C_u = torch.cdist(u0, u1) ** 2
        C_y = torch.cdist(y0, y1) ** 2
        M = C_y + self.eps * C_u


        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p
    
