import numpy as np
from bo_algorithms.my_bo.initial_design.sobol_design import SobolDesign
from typing import List, Optional, Tuple

 
class SobolMaximizer():

    def __init__(self,objective_function,config_space,n_cand):
        """
        Samples candidates uniformly at random and returns the point with the highest objective value.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_samples: int
            Number of candidates that are samples
        """
        self.config_space = config_space
        self.n_cand = n_cand
        self.objective_function = objective_function

    def maximize(self,configspace_to_vector,eta):
        """ 
        Maximizes the given acquisition function. 

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        float
            The value of the acquisition function
        """

        #Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6)) 

        init_design_def_kwargs = {
            "cs": self.config_space,  # type: ignore[attr-defined] # noqa F821
            "traj_logger": None,
            "rng": seed,
            "ta_run_limit": None,  # type: ignore[attr-defined] # noqa F821
            "configs": None,
            "n_configs_x_params": 0,
            "max_config_fracs": 0.0,
            "init_budget": self.n_cand
            } 
        sobol  = SobolDesign(**init_design_def_kwargs)
        population = sobol._select_configurations() #self.config_space.sample_configuration(size=initial_config_size)
        if not isinstance(population, List):
            population = [population]
        # the population is maintained in a list-of-vector form where each ConfigSpace
        # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
        X_candidates = np.array([configspace_to_vector(individual) for individual in population])
        #y = self.objective_function(X_candidates,eta =eta)
        self.objective_function.update(eta)
        y = self.objective_function(X_candidates)
        #Find the point of X_candidates with maximum Acquisition function.
        return X_candidates[y.argmax()],np.max(y)

