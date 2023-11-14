from copy import deepcopy
import numpy as np
import time

from ConfigSpace import ConfigurationSpace,Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from ConfigSpace.util import impute_inactive_values,deactivate_inactive_hyperparameters

from typing import List
from bo_algorithms.my_bo.per_group_smac import Per_Group_Bayesian_Optimization


import pandas as pd




class RF_Local:
    """The Random Forest Based Regression Local Bayesian Optimization.udnn-cu11, minio, kiwisolver, Jinja2, importlib-metadata, emcee, Deprecated, autograd, alive-progress, xgboost, torch, stevedore, scikit-learn, requests-toolbelt, paramz, pandas, matplotlib, george, debtcollector, dask, ConfigSpace, click, autograd-gamma, torchvision, statsmodels, pymoo, oslo.utils, oslo.config, openml, gpytorch, GPy, formulaic, 
    
    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    n_training_steps : Number of training steps for learning the GP hypers, int

    Example usage:
        RF1 = Random_Forest_1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        RF1.optimize()  # Run optimization
        X, fX = RF1.X, RF1.fX  # Evaluated points
    """


    def __init__(
        self,
        f,
        lb ,
        ub ,
        configuration_space:dict,
        initial_design,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        random_seed = int(1e6),
        acq_funct = 'EI',
        model = 'RF',
        maximizer  = 'Sobol',
        local_search=None,
        grid_values = None, 
        box_cox_enabled = None,
        
    ): 

        # Very basic input checks
        """assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub) 
        assert np.all(ub > lb)"""
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) 
        assert isinstance(batch_size, int)
        assert max_evals > n_init and max_evals > batch_size

        # Save function information
        #Objective function
        self.f = f

        #Set a seed.
        self.seed = random_seed
        self.rng = np.random.RandomState(self.seed)

        self.config_space =  configuration_space

        # Find the type of the configuration space, either Dictionary or ConfiguratioSpace object
        # Miscellaneous
        """self.isConfigspace = True if isinstance(self.config_space, ConfigurationSpace) else False
        self.hps = dict()
        if self.isConfigspace:
            for i, hp in enumerate(self.config_space.get_hyperparameters()):
                # maps hyperparameter name to positional index in vector form
                self.hps[hp.name] = i"""

        # Settings
        self.n_init = n_init #Initial configurations
        self.max_evals = max_evals #Maxmimum evaluations
        self.batch_size = batch_size #Number of points to maximize  acquisition for.
        self.verbose = verbose #hm

        # Best configuration and the score of the best configuration.
        self.inc_score = np.inf
        self.inc_config = None

        #History of actions. Complementary to the X,fx
        self.history = []


        # Save the full history
        self.X = []
        self.fX = np.array([])

        self.surrogate_time = np.array([])
        self.acquisition_time = np.array([])
        self.objective_time = np.array([])
        self.total_time = np.array([])

        #Number of current evaluations!
        self.n_evals = 0 
        
        self.batch_size = 1


        #Store the group that was selected at each iteration.
        self.X_group = []

        #Construct the Bayesian Optimization Objects per case.
        self.object_per_group = {}

        #Initialize the X and fX dictionaries.
        for classifier_name in configuration_space:
            classifier_specific_config_space = configuration_space[classifier_name]
            self.object_per_group[classifier_name] = Per_Group_Bayesian_Optimization(f= self.f,lb=None,ub=None,\
                                                                                    configuration_space=classifier_specific_config_space,\
                                                                                    initial_design=initial_design,n_init=n_init,max_evals=max_evals,
                                                                                    batch_size=batch_size,random_seed=random_seed,\
                                                                                      acq_funct=acq_funct,model=model,maximizer=maximizer,
                                                                                      group_name =classifier_name,\
                                                                                      local_search=local_search,grid_values = grid_values,box_cox_enabled=box_cox_enabled)
    
    # Just call each class and run the initial configurations of each.
    def run_initial_configurations(self):
        initial_time = []

        #train initial configurations and train surrogate per model.
        for classifier_name in self.object_per_group:
            self.object_per_group[classifier_name].run_initial_configurations()
            self.object_per_group[classifier_name].train_surrogate()
            total_time = self.object_per_group[classifier_name].total_time
            total_time[-1] += self.object_per_group[classifier_name].surrogate_time[-1]
                    
            initial_time.append(total_time)

        #save time cost.
        self.total_time = np.array(initial_time).flatten()

        #store evaluations
        self.n_evals = self.n_init * len(self.object_per_group)
        # Compute incumberment after initial evaluations.
        self.compute_incumberment_overall()
        self.track_initial_groups()


    # Computate the best configuration.
    def compute_current_incumberment(self):
        inc_score_list = []
        for classifier_name in self.object_per_group:
            inc_score_list.append((self.object_per_group[classifier_name].inc_config , self.object_per_group[classifier_name].inc_score))
        
        # Sort the list by the first element of each tuple
        # Reverse = False means that the min is first element ( LOWEST ERROR  )
        sorted_list = sorted(inc_score_list, key=lambda x: x[1],reverse=False)
        self.inc_config = sorted_list[0][0]
        self.inc_score = sorted_list[0][1]

    # for each classifier, get the history, 
    # and keep only the best fX for each step. (To make the error curve + the config.)
    def compute_initial_configurations_curve(self):
        df = pd.DataFrame()
        #stack fX values per group
        for classifier_name in self.object_per_group:
            df[classifier_name] = self.object_per_group[classifier_name].fX
        

        
        #Compute the incumberment per step per classifer
        self.fX= np.array(df.min(axis=1))

        #Find the group with the minimum per iteration.
        min_columns = df.idxmin(axis=1)

        min_group_per_iter = list(min_columns)
        
        for i in range(len(min_group_per_iter)):
            group_name = min_group_per_iter[i]
            self.X.append(self.object_per_group[group_name].X[i])
            

    #Compute the best configuration overall.
    def compute_incumberment_overall(self):
        inc_score_list = []
        for classifier_name in self.object_per_group:
            inc_score_list.append((self.object_per_group[classifier_name].inc_config , self.object_per_group[classifier_name].inc_score))
        
        # Sort the list by the first element of each tuple
        # Reverse = False means that the min is first element ( LOWEST ERROR  )
        sorted_list = sorted(inc_score_list, key=lambda x: x[1],reverse=False)
        potential_config = sorted_list[0][0]
        potential_score = sorted_list[0][1]


        if self.inc_score > potential_score:
            self.inc_config = potential_config
            self.inc_score = potential_score
            print(f'Best score so far : {self.inc_score}')
            return 1
        return 0
    
    def make_X_Y(self):

        # Current configuration iterator in each group.
        counter_per_group = {}
        for classifier_name in self.object_per_group:
            counter_per_group[classifier_name] = 0

        #Get each group
        for group in self.X_group:
            counter = counter_per_group[group]
            self.fX = np.append(self.fX, self.object_per_group[group].fX[counter])
            counter_per_group[group]+=1

    def track_initial_groups(self):
        for i in range(0,self.n_init):
            for group_name in self.object_per_group:
                self.X_group.append(group_name)

    def run(self):

        #Run initial configurations per algorithm
        self.run_initial_configurations()

        changed = 1

        while self.n_evals < self.max_evals:
            # Defense against the dark bugs
            assert self.n_evals < self.max_evals
            

            # if incumberment changes --> run acquisition maximization for all groups.
            if changed == 1:
                self.max_acquisitions_configs = {}
                self.max_acquisitions_score = {}
                for classifier_name in self.object_per_group:
                    X_next,acquisition_value =self.object_per_group[classifier_name].suggest_next_point(self.inc_score)
                    self.max_acquisitions_configs[classifier_name] = X_next
                    self.max_acquisitions_score[classifier_name] = acquisition_value
            else:
                    #Compute acquisition value only for the next new configuration if the incumberment has not changed.
                X_next,acquisition_value = self.object_per_group[best_next_classifier].suggest_next_point(self.inc_score)
                self.max_acquisitions_configs[best_next_classifier] = X_next
                self.max_acquisitions_score[best_next_classifier] = acquisition_value


            #Get the maximum acquisition for all.
            #Select group with highest acquisition --> check code.
            best_next_classifier = max(self.max_acquisitions_score, key=lambda k: self.max_acquisitions_score.get(k))
            #Just add the next group here.
            self.X_group.append(best_next_classifier)
            
            #Get the best configuration using the best group.
            best_next_config = self.max_acquisitions_configs[best_next_classifier]

            #Run objective on this group.
            fX_next = self.object_per_group[best_next_classifier].run_objective(best_next_config)

            #Check incumberment
            changed = self.compute_incumberment_overall()

            #Train the surrogate model for the specific group ONLY.
            self.object_per_group[best_next_classifier].train_surrogate()

            #Increase n_evals --> current evaluations by batch-size 
            #Batch Size == 1 for now.
            self.n_evals+= self.batch_size
        self.make_X_Y()

        self.acquisition_time = np.array([0 for i in range(self.max_evals)])
        self.surrogate_time = np.array([0 for i in range(self.max_evals)])
        self.objective_time = np.array([0 for i in range(self.max_evals)])
        return self.inc_score


 
                


