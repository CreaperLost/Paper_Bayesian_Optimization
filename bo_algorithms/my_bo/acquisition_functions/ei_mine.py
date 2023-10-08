from scipy.stats import norm
import numpy as np

from bo_algorithms.my_bo.acquisition_functions.base_acquisition import BaseAcquisitionFunction



 
class EI(BaseAcquisitionFunction):

    def __init__(self,model, par=0.0):

        r"""
        Computes for a given x the expected improvement as
        acquisition_functions value.
        :math:`EI(X) :=
            \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) -
                f_{t+1}(\mathbf{X}) - \xi\right] \} ]`, with
        :math:`f(X^+)` as the incumbent.

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
                 - getCurrentBestX().
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)

        par: float
            Controls the balance between exploration
            and exploitation of the acquisition_functions function. Default is 0.0
        """

        super(EI, self).__init__(model)
        #Always 0
        self.par = par
        self.eta = None 


    # updates the acquisition function.
    def update(self,eta):
        self.eta = eta

    def compute(self, X ,derivative=False, **kwargs): 
        """
        Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(1, D), The input point where the acquisition_functions function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition_functions
            function at X is returned

        eta: float
            The baseline performance y_star to compute the improvement

        Returns
        -------
        np.ndarray(1,1)
            Expected Improvement of X
        np.ndarray(1,D)
            Derivative of Expected Improvement at X (only if derivative=True)
        """
        
        m, v = self.model.predict(X)

        neg_nos = [num for num in m if num < 0]

        """if len(neg_nos) >0:
            print('Negative mean' , neg_nos)"""

        if self.eta is None:
            print('Please impute ETA. (Best configuration as parameter)')
            raise ValueError

        s = np.sqrt(v)
        
        if (s == 0).any():
            f = np.array([[0]])
            df = np.zeros((1, X.shape[1]))
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            """s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            z = (eta - m - self.par) / s
            f = s * (z * norm.cdf(z) + norm.pdf(z))
            f[s_copy == 0.0] = 0.0"""
            
        else:
            z = (self.eta - m - self.par) / s
            f = s * (z * norm.cdf(z) + norm.pdf(z))
            if derivative:
                dmdx, ds2dx = self.model.predictive_gradients(X)
                dmdx = dmdx[0]
                ds2dx = ds2dx[0][:, None]
                dsdx = ds2dx / (2 * s)
                df = (-dmdx * norm.cdf(z) + (dsdx * norm.pdf(z))).T
            if (f < 0).any():
                raise ValueError
        if derivative:
            return f, df
        else:
            return f
