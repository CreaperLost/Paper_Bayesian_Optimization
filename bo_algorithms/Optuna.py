from ConfigSpace import Configuration, ConfigurationSpace
import pandas as pd
import numpy as np
import time
import optuna
from optuna.samplers import TPESampler


class Optuna:
                

    def __init__(self,n_init,max_evals,seed,objective_function):
        
        self.objective_function = objective_function
        self.max_evals = max_evals
        self.seed =seed 
        

        #Basic TPE
        self.sampler = TPESampler(n_startup_trials = n_init,seed=self.seed)
        #Don't prune
        self.pruner=optuna.pruners.NopPruner()

        self.study = optuna.create_study(direction='minimize',sampler=self.sampler,pruner=self.pruner)
        



        # Extra stuff for the future
        """self.save_configuration={}

        for i in configuration_space:
            self.save_configuration[i] = pd.DataFrame()"""

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


        self.study.optimize(self.objective_function, n_trials=self.max_evals,n_jobs=1) 

        print(self.study.best_trial)


        self.fX = np.array([trial.values[0] for trial in self.study.trials])

        print(self.fX)

        end_time = time.time()
            