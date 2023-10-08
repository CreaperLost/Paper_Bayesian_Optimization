from ConfigSpace import Configuration, ConfigurationSpace
import pandas as pd
import numpy as np
import time
import hyperopt

class HyperOpt:
                

    def __init__(self,n_init,max_evals,seed,objective_function,space):
        
        self.config_space = space
        self.objective_function = objective_function
        self.max_evals = max_evals
        self.seed = seed 
        self.rng = np.random.default_rng(self.seed)
        
        self.fX = np.array([])
        self.X_group = np.array([])
        

        self.acquisition_time = np.array([0 for i in range(max_evals)])
        self.surrogate_time = np.array([0 for i in range(max_evals)])
        self.objective_time = np.array([0 for i in range(max_evals)])
        
        self.inc_score = np.inf
        self.inc_config = {}

        

        self.total_time = pd.DataFrame(columns=['Time','Score'])  
    
    def run(self):
        start_time = time.time()

        trials = hyperopt.Trials()
        print(trials)
        print(self.config_space)
        #  changed hyperopt.tpe.py vaue of _default_n_startup_jobs = 50
        # line 790!!
        hyperopt.fmin(self.objective_function, self.config_space,algo=hyperopt.tpe.suggest,trials=trials, max_evals=self.max_evals, rstate=self.rng,show_progressbar=True)

        print(trials.results)
        self.fX = np.array([trial['loss'] for trial in trials.results])

        print(self.fX)

        end_time = time.time()
            