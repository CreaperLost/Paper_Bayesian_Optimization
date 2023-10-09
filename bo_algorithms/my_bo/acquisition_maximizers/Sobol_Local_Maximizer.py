import numpy as np
from bo_algorithms.my_bo.initial_design.sobol_design import SobolDesign
from typing import List, Optional, Tuple
import pandas as pd
 
class Sobol_Local_Maximizer():

    def __init__(self,objective_function,config_space,n_cand,local_points = None,change_to_vector=None,stdev=None):
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

        if local_points == None:
            self.local_points = 100
        else:
            self.local_points = local_points


        if stdev == None:
            self.stdev = 0.1
        else:
            self.stdev = stdev


        
        self.change_to_vector  = change_to_vector

    def maximize(self,configspace_to_vector,eta,best_config):
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

        # Put a Gaussian on the incumbent and sample from that
        loc = configspace_to_vector(best_config)
        scale = np.ones([loc.shape[0]]) * self.stdev
        rand_incs = np.array( [np.clip(np.random.normal(loc, scale), 0, 1)
                              for _ in range(int(self.local_points))])
        # the population is maintained in a list-of-vector form where each ConfigSpace
        # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
        X_candidates = np.concatenate((rand_incs, np.array([configspace_to_vector(individual) for individual in population])  )) 
        

        #y = self.objective_function(X_candidates,eta =eta)
        self.objective_function.update(eta)
        y = self.objective_function(X_candidates)
        
        #Find the point of X_candidates with maximum Acquisition function.
        return X_candidates[y.argmax()],np.max(y)
    


    def batch_maximize(self,eta,best_config):
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
        assert self.change_to_vector!=None

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

        # Put a Gaussian on the incumbent and sample from that
        loc = self.change_to_vector(best_config)
        scale = np.ones([loc.shape[0]]) * 0.1
        rand_incs = np.array( [np.clip(np.random.normal(loc, scale), 0, 1)
                              for _ in range(int(self.local_points))])
        # the population is maintained in a list-of-vector form where each ConfigSpace
        # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
        X_candidates = np.concatenate((rand_incs, np.array([self.change_to_vector(individual) for individual in population])  )) 
        

        #y = self.objective_function(X_candidates,eta =eta)
        self.objective_function.update(eta)
        y = self.objective_function(X_candidates)
        #Find the point of X_candidates with maximum Acquisition function.
        return X_candidates,y
