from typing import List, Optional, Tuple, Union, cast
import numpy as np
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from ConfigSpace import ConfigurationSpace
from bo_algorithms.my_bo.surrogate.base_surrogate_model import BaseModel,get_types
from bo_algorithms.my_bo.surrogate.utils import Prior,SoftTopHatPrior,TophatPrior,HorseshoePrior, LognormalPrior
import sklearn.gaussian_process
from sklearn.gaussian_process.kernels import Kernel, KernelOperator
from sklearn.gaussian_process import GaussianProcessRegressor
from bo_algorithms.my_bo.surrogate.kernels import (ConstantKernel,
    HammingKernel,
    Matern,
    WhiteKernel,)

from sklearn.preprocessing import power_transform


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

 
class GaussianProcess(BaseModel):
    """Gaussian process model.

    The GP hyperparameterŝ are obtained by optimizing the marginal log likelihood.

    This code is based on the implementation of RoBO:

    Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
    RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
    In: NIPS 2017 Bayesian Optimization Workshop

    Parameters 
    ----------
    
    seed : int
        Model seed.
    normalize_y : bool
        Zero mean unit variance normalization of the output values
    n_opt_restart : int
        Number of restarts for GP hyperparameter optimization
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        seed: int,
        box_cox_enabled=None,
        n_opt_restarts: int = 10,
    ):
        super().__init__( )

        self.config_space = configspace
        
        self.rng = np.random.RandomState(seed)
        self.random_seed = seed

        self.standardize_output = box_cox_enabled


        self.n_opt_restarts = n_opt_restarts

        self.hypers = np.empty((0,))
        self.is_trained = False
        self._n_ll_evals = 0

        
        """
        types : List[int]
                        Specifies the number of categorical values of an input dimension where
                        the i-th entry corresponds to the i-th input dimension. Let's say we
                        have 2 dimension where the first dimension consists of 3 different
                        categorical choices and the second dimension is continuous than we
                        have to pass [3, 0]. Note that we count starting from 0.
        bounds : List[Tuple[float, float]]
                        bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims"""
        self.types, self.bounds = get_types(self.config_space, None)

        # After getting the types and bounds specify the desired kernels according to the types and the bounds!
        self.kernel = self.kernel_selector(self.types)
        # Then set some conditions 
        self._set_has_conditions()

    """
    kernel : george kernel object
        Specifies the kernel that is used for all Gaussian Process
    prior : prior object
        Defines a prior for the hyperparameters of the GP. Make sure that
        it implements the Prior interface."""
    def kernel_selector(self,types):
        cov_amp = ConstantKernel(
                    2.0,
                    constant_value_bounds=(np.exp(-10), np.exp(2)),
                    prior=LognormalPrior(mean=0.0, sigma=1.0, rng=self.rng),
                )

            
        cont_dims = np.where(np.array(types) == 0)[0]
        cat_dims = np.where(np.array(types) != 0)[0]

        lower_bounds = -6.754111155189306
        upper_bounds = 0.0858637988771976

        if len(cont_dims) > 0:
            exp_kernel = Matern(
                        np.ones([len(cont_dims)]),
                        [(np.exp(lower_bounds), np.exp(upper_bounds)) for _ in range(len(cont_dims))],
                        nu=2.5,
                        operate_on=cont_dims)

        if len(cat_dims) > 0:
            ham_kernel = HammingKernel(
                        np.ones([len(cat_dims)]),
                        [(np.exp(lower_bounds), np.exp(upper_bounds)) for _ in range(len(cat_dims))],
                        operate_on=cat_dims)

        assert (len(cont_dims) + len(cat_dims)) == len(self.config_space.get_hyperparameters())

        noise_kernel = WhiteKernel(
                    noise_level=1e-8,
                    noise_level_bounds=(np.exp(-25), np.exp(2)),
                    prior=HorseshoePrior(scale=0.1, rng=self.rng),
                )

        if len(cont_dims) > 0 and len(cat_dims) > 0:
            # both
            kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
        elif len(cont_dims) > 0 and len(cat_dims) == 0:
            # only cont
            kernel = cov_amp * exp_kernel + noise_kernel
        elif len(cont_dims) == 0 and len(cat_dims) > 0:
            # only cont
            kernel = cov_amp * ham_kernel + noise_kernel
        else:
            raise ValueError()

        return kernel

    def _get_all_priors(
        self,
        add_bound_priors: bool = True,
        add_soft_bounds: bool = False,
    ) -> List[List[Prior]]:
        """Returns all priors."""
        # Obtain a list of all priors for each tunable hyperparameter of the kernel
        all_priors = []
        to_visit = []
        to_visit.append(self.gp.kernel.k1)
        to_visit.append(self.gp.kernel.k2)
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                continue
            elif isinstance(current_param, Kernel):
                hps = current_param.hyperparameters
                assert len(hps) == 1
                hp = hps[0]
                if hp.fixed:
                    continue
                bounds = hps[0].bounds
                for i in range(hps[0].n_elements):
                    priors_for_hp = []
                    if current_param.prior is not None:
                        priors_for_hp.append(current_param.prior)
                    if add_bound_priors:
                        if add_soft_bounds:
                            priors_for_hp.append(
                                SoftTopHatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    rng=self.rng,
                                    exponent=2,
                                )
                            )
                        else:
                            priors_for_hp.append(
                                TophatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    rng=self.rng,
                                )
                            )
                    all_priors.append(priors_for_hp)
        return all_priors

    def _set_has_conditions(self) -> None:
        has_conditions = len(self.config_space.get_conditions()) > 0
        to_visit = []
        to_visit.append(self.kernel)
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                current_param.has_conditions = has_conditions
            elif isinstance(current_param, sklearn.gaussian_process.kernels.Kernel):
                current_param.has_conditions = has_conditions
            else:
                raise ValueError(current_param)

    def train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool = True) -> "GaussianProcess":
        """Computes the Cholesky decomposition of the covariance of X and estimates the GP
        hyperparameters by optimizing the marginal loglikelihood. The prior mean of the GP is set to
        the empirical mean of X.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        """
        #Impute inactive in the X data
        #X = self._impute_inactive(X)

        if self.standardize_output == True:
            tmp_y = y.copy()

            try:
                if y.min() <= 0:
                    y = power_transform(y / y.std(), method = 'yeo-johnson',standardize=False)
                else:
                    y = power_transform(y / y.std(), method = 'box-cox',standardize=False)
                    if y.std() < 0.5:
                        y = power_transform(y / y.std(), method = 'yeo-johnson',standardize=False)
                if y.std() < 0.5:
                    print('It failed...')
                    raise RuntimeError('Power transformation failed')
            except:    
                #Reset y.
                y = tmp_y

            y = self._normalize_y(y)


        y = y.flatten()

        # Try to optimize the parameters of GP. (Kernel)
        n_tries = 10
        for i in range(n_tries):
            try:
                self.gp = self._get_gp()
                self.gp.fit(X, y)
                break
            except np.linalg.LinAlgError as e:
                if i == n_tries:
                    raise e
                # Assume that the last entry of theta is the noise
                theta = np.exp(self.kernel.theta)
                theta[-1] += 1
                self.kernel.theta = np.log(theta)


        # Call the optimize, set the parameters and in the end fit to the data.

        if do_optimize: 
            self._all_priors = self._get_all_priors(add_bound_priors=False)
            self.hypers = self._optimize()
            self.gp.kernel.theta = self.hypers
            self.gp.fit(X, y)
        else:
            self.hypers = self.gp.kernel.theta

        self.is_trained = True
        return self

    def _get_gp(self) -> GaussianProcessRegressor:
        return GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=False,
            optimizer=None,
            n_restarts_optimizer=-1,  # Do not use scikit-learn's optimization routine
            alpha=0,  # Governed by the kernel
            random_state=self.rng,
        )


    # Computes the negative marginal log likelihood.
    def _nll(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        """Returns the negative marginal log likelihood (+ the prior) for a hyperparameter
        configuration theta. (negative because we use scipy minimize for optimization)

        Parameters
        ----------
        theta : np.ndarray(H)
            Hyperparameter vector. Note that all hyperparameter are
            on a log scale.

        Returns
        -------
        float
            lnlikelihood + prior
        """
        self._n_ll_evals += 1

        try:
            lml, grad = self.gp.log_marginal_likelihood(theta, eval_gradient=True)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros(theta.shape)

        for dim, priors in enumerate(self._all_priors):
            for prior in priors:
                lml += prior.lnprob(theta[dim])
                grad[dim] += prior.gradient(theta[dim])

        # We add a minus here because scipy is minimizing
        if not np.isfinite(lml).all() or not np.all(np.isfinite(grad)):
            return 1e25, np.zeros(theta.shape)
        else:
            return -lml, -grad


    #Optimize the Gaussian Process parameters
    def _optimize(self) -> np.ndarray:
        """Optimizes the marginal log likelihood and returns the best found hyperparameter
        configuration theta.

        Returns
        -------
        theta : np.ndarray(H)
            Hyperparameter vector that maximizes the marginal log likelihood
        """
        log_bounds = [(b[0], b[1]) for b in self.gp.kernel.bounds]

        # Start optimization from the previous hyperparameter configuration
        p0 = [self.gp.kernel.theta]
        if self.n_opt_restarts > 0:
            dim_samples = []

            prior = None  
            for dim, hp_bound in enumerate(log_bounds):
                prior = self._all_priors[dim]
                # Always sample from the first prior
                if isinstance(prior, list):
                    if len(prior) == 0:
                        prior = None
                    else:
                        prior = prior[0]
                prior = cast(Optional[Prior], prior)
                if prior is None:
                    try:
                        sample = self.rng.uniform(
                            low=hp_bound[0],
                            high=hp_bound[1],
                            size=(self.n_opt_restarts,),
                        )
                    except OverflowError:
                        raise ValueError("OverflowError while sampling from (%f, %f)" % (hp_bound[0], hp_bound[1]))
                    dim_samples.append(sample.flatten())
                else:
                    dim_samples.append(prior.sample_from_prior(self.n_opt_restarts).flatten())
            p0 += list(np.vstack(dim_samples).transpose())

        theta_star: Optional[np.ndarray] = None
        f_opt_star = np.inf
        for i, start_point in enumerate(p0):
            theta, f_opt, _ = optimize.fmin_l_bfgs_b(self._nll, start_point, bounds=log_bounds)
            if f_opt < f_opt_star:
                f_opt_star = f_opt
                theta_star = theta

        if theta_star is None:
            raise RuntimeError

        return theta_star

    def predict(
        self, X_test: np.ndarray, cov_return_type: Optional[str] = "diagonal_cov"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        cov_return_type: Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.

        Returns
        -------
        np.array(N,)
            predictive mean
        np.array(N,) or np.array(N, N) or None
            predictive variance or standard deviation
        """
        if not self.is_trained:
            raise Exception("Model has to be trained first!")

        #X_test = self._impute_inactive(X_test)

        if cov_return_type is None:
            mu = self.gp.predict(X_test)
            var = None

            if self.standardize_output == True:
                mu = self._untransform_y(mu,var)

        else:
            predict_kwargs = {"return_cov": False, "return_std": True}

            if cov_return_type == "full_cov":
                predict_kwargs = {"return_cov": True, "return_std": False}

            mu, var = self.gp.predict(X_test, **predict_kwargs)

            if cov_return_type != "full_cov":
                var = var**2  # since we get standard deviation for faster computation

            # Clip negative variances and set them to the smallest
            # positive float value
            var = np.clip(var, 0.0000001, np.inf)

            if self.standardize_output == True:
                mu, var = self._untransform_y(mu, var)

            if cov_return_type == "diagonal_std":
                var = np.sqrt(var)  # converting variance to std deviation if specified

        return mu, var

    #Currently not used. Maybe for different optimization in the future. 
    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        """Samples F function values from the current posterior at the N specified test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        -------
        function_samples: np.array(N, F)
            The F function values drawn at the N test points.
        """
        if not self.is_trained:
            raise Exception("Model has to be trained first!")

        #X_test = self._impute_inactive(X_test)
        funcs = self.gp.sample_y(X_test, n_samples=n_funcs, random_state=self.rng)

        if self.standardize_output == True:
            funcs = self._untransform_y(funcs)

        if len(funcs.shape) == 1:
            return funcs[None, :]
        else:
            return funcs
