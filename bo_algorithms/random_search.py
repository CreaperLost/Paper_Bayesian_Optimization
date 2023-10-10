from time import time 
from ConfigSpace import ConfigurationSpace
import numpy as np
import pandas as pd


class Random_Search:


    def __init__(self,
        f,
        configuration_space:ConfigurationSpace,
        n_init,
        max_evals,
        random_seed = int(1e6),verbose=True):


    
        self.objective_function = f 
        self.config_space = configuration_space

        #Set a seed.
        self.seed = random_seed
        self.rng = np.random.RandomState(self.seed)

        self.n_init = n_init #Initial configurations
        self.max_evals = max_evals #Maxmimum evaluations

        self.inc_config = None


        self.X_per_group = {}
        self.fX_per_group ={}

        #Initialize the X and fX dictionaries.
        for classifier_name in configuration_space['model'].choices:
            self.X_per_group[classifier_name] = pd.DataFrame()
            self.fX_per_group[classifier_name] = np.array([])

        # Overall groups.
        self.X = []
        self.fX = np.array([])

        self.surrogate_time = np.array([])
        self.objective_time = np.array([])
        self.acquisition_time = np.array([])
        self.total_time = np.array([])

        self.inc_score = np.inf
        self.n_evals = 0
        self.verbose = True

    def run(self):
        #Random Sample max_evals configurations.
        configurations = self.config_space.sample_configuration(self.max_evals)

        for config in configurations:

            #print(f'Current iteration {self.n_evals}')
            
            #Measure Objective Time
            start_time = time()
            kfold_score = self.objective_function(config.get_dictionary())
            end_time = time()-start_time

            #Save objective_time.
            self.objective_time = np.concatenate((self.objective_time,np.array([end_time])))

            #Get the objective score.
            fX_next =np.array([kfold_score])

            #Store into overall-groups
            self.fX = np.concatenate((self.fX, fX_next))

            #Store into each group.

            #find which category is this configurations in the conditional space
            model_type = config['model']

            #Store score per configuration per predictive model
            curr_model_scores = self.fX_per_group[model_type]
            self.fX_per_group[model_type] = np.concatenate((curr_model_scores, fX_next))
            
            #Store configuration per iteration. (Really-Dummy-Solution)
            self.X.append(config.get_dictionary())

            #Store configuration per group
            new_config = config.get_dictionary().copy()
            #Remove model from config.
            model_type = new_config.pop('model')
            #Get the current configurations for the specific type.
            new_row = pd.DataFrame(new_config,index=[0])
            self.X_per_group[model_type] = self.X_per_group[model_type].append(new_row,ignore_index=True)
            
            #Add other costs.
            self.surrogate_time = np.concatenate((self.surrogate_time,np.array([0])))
            self.acquisition_time = np.concatenate((self.acquisition_time,np.array([0])))
            self.n_evals += 1


            #Find incumberment.
            if self.verbose and fX_next[0] < self.inc_score:
                self.inc_config = new_config
                self.inc_score = fX_next[0]
                print(f"{self.n_evals}) New best: {self.inc_score:.4}")
                #print(self.inc_config)


            #Total Cost of running random_search
            end_time_total =  time() - start_time
            self.total_time = np.concatenate((self.total_time,np.array([end_time_total])))

        return self.inc_score
        