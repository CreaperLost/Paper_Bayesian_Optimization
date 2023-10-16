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
from benchmark.hold_out_datamanager import Holdout_OpenMLDataManager

import copy 

from benchmark.hyper_parameters import *
    
    
    
metrics = dict(
    auc = roc_auc_score #accuracy_score
)

metrics_kwargs = dict(
    auc =dict(multi_class="ovr",needs_proba=True) #dict() #
)


class Classification_Benchmark:

    def __init__(
            self,
            task_id: int,
            seed: int = None,
            data_path: Union[str, Path, None] = None,
            data_repo:str = 'Jad',
            use_holdout =False,
    ):
        assert seed != None

        self.seed = seed
        
        self.rng = np.random.RandomState(np.abs(self.seed))

        self.task_id = task_id
        self.scorers = dict()
        for k, v in metrics.items():
            self.scorers[k] = make_scorer(v, **metrics_kwargs[k])

        self.data_path = 'Datasets/OpenML'

        
        #dm = OpenMLDataManager(task_id, data_path, self.seed,n_folds = 5, use_holdout = use_holdout)
        dm = Holdout_OpenMLDataManager(task_id=task_id, data_path=data_path, seed=self.seed, n_folds=5)
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
        self.configuration_space, _ = self.get_configuration_space()

    @staticmethod
    def get_configuration_space(self) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        raise NotImplementedError()
    
    def get_configuration_space_multifidelity(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        raise NotImplementedError()


    def init_model(self, config: Union[CS.Configuration, Dict],n_feat : int):
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


    def train_model_on_fold(self, train_config: dict, fold_number: int) -> object:
        """
        This function will call .fit of the model to the respective training fold.
        returns : a trained model object.
        """
        config = train_config['config']

        seed = train_config.get('seed',self.seed)
        
        X = self.train_X[fold_number]
        y = self.train_y[fold_number]

        if train_config['optimizer'] == 'typical':
            model = self.init_model(config=config, n_feat= X.shape[1], seed = seed )
        elif train_config['optimizer'] == 'mango':
            model = self.mango_init_model(config = config, n_feat = X.shape[1] ,model_type= train_config['model_type'])
        
        model.fit(X,y)

        return model

    def train_model_on_all_data(self, train_config:dict) -> object:
        """
        responsible to fit the model on both train and validation dataset. 
        Will be used to get performance estimate on hold-out test set.
        """
        config = train_config['config']

        seed = train_config.get('seed',self.seed)


        X = np.vstack((self.train_X[0], self.valid_X[0]))
        y = pd.concat((self.train_y[0], self.valid_y[0]))
        
        train_idx = np.arange(len(X))

        if train_config['optimizer'] == 'typical':
            model = self.init_model(config=config, n_feat= X.shape[1], seed = seed )
        elif train_config['optimizer'] == 'mango':
            model = self.mango_init_model(config, n_feat = X.shape[1],model_type= train_config['model_type'], seed= seed)
        
        model.fit(X[train_idx],y[train_idx])
        
        return model


    def _train_objective(self,
                         config: Dict,
                         evaluation: Union[str, None] = None, seed = None) :

        assert evaluation != None

        pass_seed = self.seed
        if seed != None : pass_seed = seed

        print(f'Seed argument and final seed : {seed,pass_seed}')

        train_config  = { 'config' : config, 'optimizer': 'typical', 'seed': pass_seed}

        if evaluation == "val": model = [self.train_model_on_fold(train_config = train_config, fold_number = fold) for fold in range(len(self.train_X))]
        else: model = self.train_model_on_all_data(train_config)

        return model


    # Train 1 model on 1 fold and just return it.
    def _train_objective_per_fold(self,
                         config: Dict,
                         evaluation: Union[str, None] = "valid",
                         fold:[int,None] = None):

        assert fold !=None
        
        if isinstance(fold,str): fold = int(fold)       

        train_config  = { 'config' : config, 'optimizer': 'Typical'}

        if evaluation == "val": model = self.train_model_on_fold(train_config = train_config, fold_number = fold)
        else: model = self.train_model_on_all_data(train_config=train_config)
            
        return model
    
    def apply_model_to_valid_fold(self, model: object, fold:int) -> float: 
        #Get the Validation Score - of 1 fold.
        val_scores = dict()
        for k, v in self.scorers.items():
            #Get the score of a model on the specific set. (1-fold only run.)
            val_scores[k] = v(model, self.valid_X[fold], self.valid_y[fold])
        return val_scores["auc"]

    def apply_model_to_cv(self, model_list: list) -> float:
        """
        Apply each of the learned model to the appropriate validation set. 
        Get the AUC score.
        And return the average score.
        """
        val_scores = dict()
        for k, v in self.scorers.items():
            #Last model  is for the test set only!
            val_scores[k] = 0.0
            for model_fold in range(len(model_list)):
                val_scores[k] += v(model_list[model_fold], self.valid_X[model_fold], self.valid_y[model_fold])
            val_scores[k] /= (len(model_list))
            print(f'{k},{v},#Folds : {len(model_list)}')
        return val_scores["auc"]
    
    def apply_model_to_holdout(self, model:object) -> int:
        """
        Apply, a trained model on the whole train-validation dataset, on the hold-out test set.
        return the score.
        """

        #If evaluation == Test then you get a single model from the train_objective :D
        test_scores = dict()
        for k, v in self.scorers.items():
            test_scores[k] = v(model, self.test_X, self.test_y)

        return test_scores["auc"]


    # The idea is that we run only on VALIDATION SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict]) -> Dict:
        """
        Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        self._check_and_cast_configuration(configuration, self.configuration_space)

        #Get a x models trained.
        model_list = self._train_objective(configuration,  evaluation="val")

        # Apply the models.
        auc_score = self.apply_model_to_cv(model_list)

        
        return 1 - auc_score # Minimize the auc score loss.


    #Get the current fold, train a model and then apply on validation set to get AUC score returned.
    def objective_function_per_fold(self, configuration: Union[CS.Configuration, Dict], fold=None) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        assert fold!= None

        self._check_and_cast_configuration(configuration, self.configuration_space)

        #Get a model trained on the fold.
        model= self._train_objective_per_fold(configuration, evaluation="val",fold=fold)

        # Get validation performance of a specific fold.
        auc_score = self.apply_model_to_valid_fold(model,fold)

        # check this one. Val_loss
        return 1 - auc_score

    # The idea is that we run only on VALIDATION SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def smac_objective_function(self,
                           configuration: Union[CS.Configuration, Dict], seed:int) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        self._check_and_cast_configuration(configuration, self.configuration_space)

        #Get a x models trained.
        model_list = self._train_objective(configuration, evaluation="val" , seed = seed)

        # Apply the models.
        auc_score = self.apply_model_to_cv(model_list)

        return 1 - auc_score


    # The idea is that we run only on TEST SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict]) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the test set"""

        self._check_and_cast_configuration(configuration, self.configuration_space)

        model = self._train_objective(configuration,evaluation="test")

        hold_out_auc_score = self.apply_model_to_holdout(model)

        return 1- hold_out_auc_score




    #Mango Specific Objective Functions!
    def mango_train_objective(self,
                         config: Dict,
                         evaluation: Union[str, None] = "valid",
                         model_type = None) -> object:

        assert model_type != None


        train_config  = { 'config' : config, 'optimizer': 'mango', 'model_type': model_type}
        
        if evaluation == "val": model = [self.train_model_on_fold(train_config = train_config, fold_number = fold) for fold in range(len(self.train_X))]
        else: model = self.train_model_on_all_data(train_config)
            
        return model

    #This applies on configuration per type of model.
    def mango_objective_function(self,configuration: Union[CS.Configuration, Dict], model_type = None) -> int:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        assert model_type !=None
        #self._check_and_cast_configuration(configuration, self.configuration_space)
        #Get a x models trained.
        model_list = self.mango_train_objective(configuration, evaluation="val",model_type=model_type)

        # Apply the models.
        auc_score = self.apply_model_to_cv(model_list)
        
        return 1 - auc_score
    

    
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
    

    
    def optuna_train(self, config, evaluation='val'):
        #Mango Specific Objective Functions

        assert config != None

        train_config  = { 'config' : config, 'optimizer': 'typical'}

        if evaluation == "val": model = [self.train_model_on_fold(train_config = train_config, fold_number = fold) for fold in range(len(self.train_X))]
        else: model = self.train_model_on_all_data(train_config)

        return model

    def optuna_objective(self,trial):

        #Use trial to select the appropriate model.
        model_to_train = self.get_optuna_space(trial)

        # Evaluate model performance -- TRAINING STEP
        model_list = self.optuna_train(model_to_train,evaluation='val')

        # Apply the models.
        auc_score = self.apply_model_to_cv(model_list)

        return 1 - auc_score

    

    def hyperopt_train_objective(self,
                         config: Dict,
                         evaluation: Union[str, None] = "valid"):


        curr_config = config['model'] # You need to access this.


        # Make sure to put curr_config and not the original config in here.
        train_config  = { 'config' : curr_config, 'optimizer': 'typical'}

        if evaluation == "val": model = [self.train_model_on_fold(train_config = train_config, fold_number = fold) for fold in range(len(self.train_X))]
        else: model = self.train_model_on_all_data(train_config)

        return model


    # The idea is that we run only on VALIDATION SET ON THIS ONE. (K-FOLD)
    # pylint: disable=arguments-differ
    def hyperopt_objective_function(self,
                           configuration: Union[CS.Configuration, Dict]) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        #Get a x models trained.
        model_list = self.hyperopt_train_objective(configuration, evaluation="val")

        # Apply the models.
        auc_score = self.apply_model_to_cv(model_list)
        
        return 1-auc_score