import time
from pathlib import Path
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer,roc_auc_score

from typing import Union, Dict

import ConfigSpace
import numpy as np

from ConfigSpace.util import deactivate_inactive_hyperparameters

from benchmark.data_manager import OpenMLDataManager
import copy 

from benchmark.hyper_parameters import *
    
    
    
metrics = dict(
    auc = roc_auc_score #accuracy_score
)

metrics_kwargs = dict(
    auc =dict(multi_class="ovr",needs_proba=True) #dict() #
)

def get_rng(rng: Union[int, np.random.RandomState, None] = None,
            self_rng: Union[int, np.random.RandomState, None] = None) -> np.random.RandomState:
    """
    Helper function to obtain RandomState from int or create a new one.

    Sometimes a default random state (self_rng) is already available, but a
    new random state is desired. In this case ``rng`` is not None and not already
    a random state (int or None) -> a new random state is created.
    If ``rng`` is already a randomState, it is just returned.
    Same if ``rng`` is None, but the default rng is given.

    Parameters
    ----------
    rng : int, np.random.RandomState, None
    self_rng : np.random.RandomState, None

    Returns
    -------
    np.random.RandomState
    """

    if rng is not None:
        return _cast_int_to_random_state(rng)
    if rng is None and self_rng is not None:
        return _cast_int_to_random_state(self_rng)
    return np.random.RandomState()


def _cast_int_to_random_state(rng: Union[int, np.random.RandomState]) -> np.random.RandomState:
    """
    Helper function to cast ``rng`` from int to np.random.RandomState if necessary.

    Parameters
    ----------
    rng : int, np.random.RandomState

    Returns
    -------
    np.random.RandomState
    """
    if isinstance(rng, np.random.RandomState):
        return rng
    if int(rng) == rng:
        # As seed is sometimes -1 (e.g. if SMAC optimizes a deterministic function) -> use abs()
        return np.random.RandomState(np.abs(rng))
    raise ValueError(f"{rng} is neither a number nor a RandomState. Initializing RandomState failed")


class Classification_Benchmark:

    def __init__(
            self,
            task_id: int,
            rng: Union[int, None] = None,
            data_path: Union[str, Path, None] = None,
            data_repo:str = 'Jad',
            use_holdout =False,
            global_seed: Union[int, None] = 1
    ):
        
        self.global_seed = global_seed

        if isinstance(rng, int):
            self.seed = rng
        else:
            self.seed = self.rng.randint(1, 10**6)

        self.rng = get_rng(rng=rng)

        self.task_id = task_id
        self.scorers = dict()
        for k, v in metrics.items():
            self.scorers[k] = make_scorer(v, **metrics_kwargs[k])

        self.data_path = 'Datasets/OpenML'

        dm = OpenMLDataManager(task_id, data_path, self.global_seed,n_folds = 5, use_holdout = use_holdout)
        dm.load()

        # Data variables
        self.train_X = dm.train_X
        self.valid_X = dm.valid_X
        self.test_X = dm.test_X
        self.train_y = dm.train_y
        self.valid_y = dm.valid_y
        self.test_y = dm.test_y
        self.train_idx = dm.train_idx
        self.test_idx = dm.test_idx
        self.task = dm.task
        self.dataset = dm.dataset
        self.preprocessor = dm.preprocessor
        self.lower_bound_train_size = dm.lower_bound_train_size
        self.n_classes = dm.n_classes

        # Observation and fidelity spaces
        self.configuration_space, _ = self.get_configuration_space(self.seed)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        raise NotImplementedError()
    
    def get_configuration_space_multifidelity(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        raise NotImplementedError()


    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        raise NotImplementedError()
    

    def _check_and_cast_configuration(self,configuration: Union[Dict, ConfigSpace.Configuration],
                                      configuration_space: ConfigSpace.ConfigurationSpace) \
            -> ConfigSpace.Configuration:
        """ Helper-function to evaluate the given configuration.
            Cast it to a ConfigSpace.Configuration and evaluate if it violates its boundaries.

            Note:
                We remove inactive hyperparameters from the given configuration. Inactive hyperparameters are
                hyperparameters that are not relevant for a configuration, e.g. hyperparameter A is only relevant if
                hyperparameter B=1 and if B!=1 then A is inactive and will be removed from the configuration.
                Since the authors of the benchmark removed those parameters explicitly, they should also handle the
                cases that inactive parameters are not present in the input-configuration.
        """
        
        if isinstance(configuration, dict):
            configuration = ConfigSpace.Configuration(configuration_space, configuration,
                                                      allow_inactive_with_values=True)
        elif isinstance(configuration, ConfigSpace.Configuration):
            configuration = configuration
        else:
            raise TypeError(f'Configuration has to be from type List, np.ndarray, dict, or '
                            f'ConfigSpace.Configuration but was {type(configuration)}')
        all_hps = set(configuration_space.get_hyperparameter_names())
        active_hps = configuration_space.get_active_hyperparameters(configuration)
        inactive_hps = all_hps - active_hps

        configuration = deactivate_inactive_hyperparameters(configuration, configuration_space)
        configuration_space.check_configuration(configuration)

        return configuration

    

    def __call__(self, configuration: Dict, **kwargs) -> float:
        """ Provides interface to use, e.g., SciPy optimizers """
        return self.objective_function(configuration, **kwargs)['function_value']




    def _train_objective(self,
                         config: Dict,
                         fidelity: Dict,
                         shuffle: bool,
                         rng: Union[np.random.RandomState, int, None] = None,
                         evaluation: Union[str, None] = "valid") :

        if rng is not None:
            rng = get_rng(rng, self.rng)

        if evaluation == "val":
            list_of_models = []
            for fold in range(len(self.train_X)):
                
                # initializing model
                model = self.init_model(config, fidelity, rng, n_feat = self.train_X[fold].shape[1])
                # preparing data -- Select the fold
                train_X = self.train_X[fold]
                train_y = self.train_y[fold]
                train_idx = self.train_idx
                
                model.fit(train_X, train_y)
                # computing statistics on training data
                list_of_models.append(model)

            # initializing model for the test set!
            model = self.init_model(config, fidelity, rng , n_feat = self.train_X[0].shape[1])
            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))
            
            # fit on the whole training + validation set once.
            model.fit(train_X[train_idx], train_y.iloc[train_idx])
            #Model trained on TRAIN + VALIDATION tests.
            list_of_models.append(model)

            #Return list of models.
            model = list_of_models

        else:
            # initializing model
            model = self.init_model(config, fidelity, rng, n_feat = self.train_X[0].shape[1])

            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))

            #Here we got 1 train set. (Train + Validation from Fold 0.)
            model.fit(train_X[train_idx], train_y.iloc[train_idx])

        return model


    # Train 1 model on 1 fold and just return it.
    def _train_objective_per_fold(self,
                         config: Dict,
                         fidelity: Dict,
                         shuffle: bool,
                         rng: Union[np.random.RandomState, int, None] = None,
                         evaluation: Union[str, None] = "valid",fold = None):

        assert fold !=None
        if rng is not None:
            rng = get_rng(rng, self.rng)

        if isinstance(fold,str):
            fold = int(fold)       

        if evaluation == "val":
            # initializing model
            model = self.init_model(config, fidelity, rng, n_feat = self.train_X[fold].shape[1])
            # preparing data -- Select the fold
            train_X = self.train_X[fold]
            train_y = self.train_y[fold]
            train_idx = self.train_idx
            # Fit the model                
            model.fit(train_X, train_y)

            """# initializing model for the test set!
            model = self.init_model(config, fidelity, rng , n_feat = self.train_X[0].shape[1])
            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))
            
            model.fit(train_X[train_idx], train_y.iloc[train_idx])
            #Model trained on TRAIN + VALIDATION tests.
            list_of_models.append(model)"""

            
        else:
            # initializing model
            model = self.init_model(config, fidelity, rng, n_feat = self.train_X[0].shape[1])

            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))

            #Here we got 1 train set. (Train + Validation from Fold 0.)
            model.fit(train_X[train_idx], train_y.iloc[train_idx])

            #This does some kind of prediction?
            # computing statistics on training data
            scores = dict()
            for k, v in self.scorers.items():
                scores[k] = v(model, train_X[train_idx], train_y.iloc[train_idx])

        return model

    # The idea is that we run only on VALIDATION SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           shuffle: bool = False,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        self._check_and_cast_configuration(configuration, self.configuration_space)
        #Get a x models trained.
        model = self._train_objective(configuration, fidelity, shuffle, rng, evaluation="val")

        #Get the Validation Score (k-fold average)
        val_scores = dict()
        for k, v in self.scorers.items():
            #Last model  is for the test set only!
            val_scores[k] = 0.0
            for model_fold in range(len(model)-1):
                val_scores[k] += v(model[model_fold], self.valid_X[model_fold], self.valid_y[model_fold])
            val_scores[k] /= (len(model)-1)
        val_loss = 1 - val_scores["auc"]

        return val_loss


    #Get the current fold, train a model and then apply on validation set to get AUC score returned.
    def objective_function_per_fold(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           shuffle: bool = False,
                           rng: Union[np.random.RandomState, int, None] = None,fold=None,
                           **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        assert fold!= None

    
        self._check_and_cast_configuration(configuration, self.configuration_space)

        #Get a model trained on the fold.
        model= self._train_objective_per_fold(configuration, fidelity, shuffle, rng, evaluation="val",fold=fold)

        #Get the Validation Score - of 1 fold.
        val_scores = dict()
        for k, v in self.scorers.items():
            #Get the score of a model on the specific set. (1-fold only run.)
            val_scores[k] = v(model, self.valid_X[fold], self.valid_y[fold])
        val_loss = 1 - val_scores["auc"]

        # check this one. Val_loss
        return val_loss

    # The idea is that we run only on VALIDATION SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def smac_objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           shuffle: bool = False,
                           seed: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        self._check_and_cast_configuration(configuration, self.configuration_space)
        #Get a x models trained.
        model = self._train_objective(configuration, fidelity, shuffle, rng=seed, evaluation="val")

        #Get the Validation Score (k-fold average)
        val_scores = dict()
        for k, v in self.scorers.items():
            #Last model  is for the test set only!
            val_scores[k] = 0.0
            for model_fold in range(len(model)-1):
                val_scores[k] += v(model[model_fold], self.valid_X[model_fold], self.valid_y[model_fold])
            val_scores[k] /= (len(model)-1)
        val_loss = 1 - val_scores["auc"]

        return val_loss


    # The idea is that we run only on TEST SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def objective_function_test(self,
                                configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                shuffle: bool = False,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the test set
        """

        self._check_and_cast_configuration(configuration, self.configuration_space)

        model = self._train_objective(configuration, fidelity, shuffle, rng, evaluation="test")

        #If evaluation == Test then you get a single model from the train_objective :D
        test_scores = dict()
        for k, v in self.scorers.items():
            test_scores[k] = v(model, self.test_X, self.test_y)
        test_loss = 1 - test_scores["auc"]

        return test_loss




    #Mango Specific Objective Functions!
    def mango_train_objective(self,
                         config: Dict,
                         fidelity: Dict,
                         shuffle: bool,
                         rng: Union[np.random.RandomState, int, None] = None,
                         evaluation: Union[str, None] = "valid",model_type = None):

        if rng is not None:
            rng = get_rng(rng, self.rng)

        assert model_type != None

        
        if evaluation == "val":
            list_of_models = []
            for fold in range(len(self.train_X)):
                
                model = self.mango_init_model(config, fidelity, rng , n_feat = self.train_X[0].shape[1],model_type=model_type)
                
                #model = self.init_model(config, fidelity, rng, n_feat = self.train_X[fold].shape[1])
                # preparing data -- Select the fold
                train_X = self.train_X[fold]
                train_y = self.train_y[fold]
                train_idx = self.train_idx
                # Fit the model
                model.fit(train_X, train_y)
                # computing statistics on training data
                list_of_models.append(model)

            # initializing model for the test set!
            model = self.mango_init_model(config, fidelity, rng , n_feat = self.train_X[0].shape[1],model_type=model_type)
            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))
            
            model.fit(train_X[train_idx], train_y.iloc[train_idx])
            #Model trained on TRAIN + VALIDATION tests.
            list_of_models.append(model)


            #Return list of models.
            model = list_of_models

        else:
            # initializing model
            model = self.init_model(config, fidelity, rng, n_feat = self.train_X[0].shape[1])

            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))

            #Here we got 1 train set. (Train + Validation from Fold 0.)
            model.fit(train_X[train_idx], train_y.iloc[train_idx])

            
           

        return model

    #This applies on configuration per type of model.
    def mango_objective_function(self,configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           shuffle: bool = False,
                           rng: Union[np.random.RandomState, int, None] = None,model_type = None,
                           **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        assert model_type !=None
        #self._check_and_cast_configuration(configuration, self.configuration_space)
        #Get a x models trained.
        model = self.mango_train_objective(configuration, fidelity, shuffle, rng, evaluation="val",model_type=model_type)

        #Get the Validation Score (k-fold average)
        val_scores = dict()
        for k, v in self.scorers.items():
            #Last model  is for the test set only!
            val_scores[k] = 0.0
            for model_fold in range(len(model)-1):
                val_scores[k] += v(model[model_fold], self.valid_X[model_fold], self.valid_y[model_fold])
            val_scores[k] /= (len(model)-1)
        #print(val_scores['auc'])
        val_loss = val_scores["auc"]

        return val_loss
    

    
    def mango_generic_objective(self, args_list, model_type):
        return [self.mango_objective_function(configuration = hyper_par,model_type = model_type) for hyper_par in args_list]
    
    def mango_objective_dt(self,args_list):
        return self.mango_generic_objective(args_list,DT_NAME)

    def mango_objective_xgb(self,args_list):
        return self.mango_generic_objective(args_list,XGB_NAME)

    def mango_objective_LinearSVM(self,args_list):
        return self.mango_generic_objective(args_list,LINEAR_SVM_NAME)
    

    def mango_objective_RBFSVM(self,args_list):
        return self.mango_generic_objective(args_list,RBF_SVM_NAME)

    def mango_objective_RF(self,args_list):
        return self.mango_generic_objective(args_list,RF_NAME)
    

    
    def optuna_train(self,config,fidelity,rng,evaluation='val'):
        #Mango Specific Objective Functions

        if rng is not None:
            rng = get_rng(rng, self.rng)

        assert config != None

        
        if evaluation == "val":
            list_of_models = []
            model_fit_time = 0
            for fold in range(len(self.train_X)):
                                
                model = self.init_model(config, fidelity, rng, n_feat = self.train_X[fold].shape[1])
                # preparing data -- Select the fold
                train_X = self.train_X[fold]
                train_y = self.train_y[fold]
                train_idx = self.train_idx
                # Fit the model
                model.fit(train_X, train_y)
                # computing statistics on training data
                list_of_models.append(model)

            # initializing model for the test set!
            model =  model = self.init_model(config, fidelity, rng, n_feat = self.train_X[fold].shape[1])
            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))
            
            model.fit(train_X[train_idx], train_y.iloc[train_idx])
            #Model trained on TRAIN + VALIDATION tests.
            list_of_models.append(model)

            #Return list of models.
            model = list_of_models

        else:
            # initializing model
            model = self.init_model(config, fidelity, rng, n_feat = self.train_X[fold].shape[1])

            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))

            #Here we got 1 train set. (Train + Validation from Fold 0.)
            model.fit(train_X[train_idx], train_y.iloc[train_idx])
            

        return model

    def optuna_objective(self,trial,rng=None):

        #Use trial to select the appropriate model.
        model_to_train = self.get_optuna_space(trial,rng)

        # Evaluate model performance -- TRAINING STEP
        model = self.optuna_train(model_to_train,None,rng,evaluation='val')

        # VALIDATION AVERAGE SCORE. 
        #Get the Validation Score (k-fold average)
        val_scores = dict()
        for k, v in self.scorers.items():
            #Last model  is for the test set only!
            val_scores[k] = 0.0
            for model_fold in range(len(model)-1):
                val_scores[k] += v(model[model_fold], self.valid_X[model_fold], self.valid_y[model_fold])
            val_scores[k] /= (len(model)-1)
        #print(val_scores['auc'])
        val_loss = 1 - val_scores["auc"]


        return val_loss
    

    def hyperopt_train_objective(self,
                         config: Dict,
                         rng: Union[np.random.RandomState, int, None] = None,
                         evaluation: Union[str, None] = "valid"):

        if rng is not None:
            rng = get_rng(rng, self.rng)

        curr_config = config['model'] # You need to access this.

        if evaluation == "val":
            list_of_models = []
            for fold in range(len(self.train_X)):
                
                # initializing model
                model = self.init_model(curr_config, rng)
                # preparing data -- Select the fold
                train_X = self.train_X[fold]
                train_y = self.train_y[fold]
                train_idx = self.train_idx
                # Fit the model
                
                model.fit(train_X, train_y)
                # computing statistics on training data
                list_of_models.append(model)

            # initializing model for the test set!
            model = self.init_model(curr_config, rng)
            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))
            
            model.fit(train_X[train_idx], train_y.iloc[train_idx])
            #Model trained on TRAIN + VALIDATION tests.
            list_of_models.append(model)


            #Return list of models.
            model = list_of_models

  
        else:
            # initializing model
            model = self.init_model(curr_config, rng)

            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            train_y = pd.concat((self.train_y[0], self.valid_y[0]))
            train_idx = np.arange(len(train_X))

            #Here we got 1 train set. (Train + Validation from Fold 0.)
            model.fit(train_X[train_idx], train_y.iloc[train_idx])



        return model


    # The idea is that we run only on VALIDATION SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def hyperopt_objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        #Get a x models trained.
        model = self.hyperopt_train_objective(
            configuration, rng, evaluation="val"
        )

        #Get the Validation Score (k-fold average)
        val_scores = dict()
        for k, v in self.scorers.items():
            #Last model  is for the test set only!
            val_scores[k] = 0.0
            for model_fold in range(len(model)-1):
                val_scores[k] += v(model[model_fold], self.valid_X[model_fold], self.valid_y[model_fold])
            val_scores[k] /= (len(model)-1)
        #print(val_scores['auc'])
        val_loss = 1 - val_scores["auc"]


        return val_loss