from ConfigSpace import Configuration, ConfigurationSpace
import pandas as pd
import numpy as np
import time
from mango import MetaTuner
class Mango:
                

    def __init__(self,configuration_space:dict,n_init,max_evals,seed,objective_functions:dict):
        
        list_config_space = []
        objective_list = []
        # LIst based configuration space, objective evaluations
        for i in configuration_space:
            list_config_space.append(configuration_space[i])
            objective_list.append(objective_functions[i])
        
        

        # Save configuration space here, Mango needs a list of configuration spaces.
        self.configspace = list_config_space

        
        # Save the list of the objective functions
        self.objective_functions = objective_list

        

        self.save_configuration={}

        for i in configuration_space:
            self.save_configuration[i] = pd.DataFrame()

        self.fX = np.array([])
        self.X_group = np.array([])
        

        self.acquisition_time = np.array([0 for i in range(max_evals)])
        self.surrogate_time = np.array([0 for i in range(max_evals)])
        self.objective_time = np.array([0 for i in range(max_evals)])
        
        self.inc_score = np.inf
        self.inc_config = {}

        self.total_time = pd.DataFrame(columns=['Time','Score'])  

        self.max_evals = max_evals -  (len(objective_list) * n_init)
        # Remove the 10 initial total random configurations...
        self.metatuner = MetaTuner(list_config_space, objective_list,n_init=n_init,n_iter=self.max_evals )
    
    def run(self):
        start_time = time.time()

        results = self.metatuner.run()

        """random_params
        random_params_objective
        random_objective_fid
        params_tried
        objective_values
        objective_fid
        best_objective
        best_params
        best_objective_fid"""


        self.fX = 1 - np.array(results['objective_values'])
        print(self.fX)
        print('best_objective:',1- results['best_objective'])
        print('best_params:',results['best_params'])
        """print(results['objective_fid'])

        print('best_objective:',results['best_objective'])
        print('best_params:',results['best_params'])
        print('best_objective_fid:',results['best_objective_fid'])
        """


        end_time = time.time()
            