from typing import Union, Dict 
import ConfigSpace as CS
import numpy as np
#from benchmarks.Group_MultiFold_MLBenchmark import Group_MultiFold_MLBenchmark
from benchmark.objective_function import Classification_Benchmark
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import uniform,loguniform
from hyperopt import hp
from benchmark.hyper_parameters import *

# this needs to change.
    # Group_MultiFold_MLBenchmark
class Classification_Configuration_Space(Classification_Benchmark):
    def __init__(self, 
                 task_id: int, rng: Union[np.random.RandomState, int, None] = None,
                 data_path: Union[str, None] = None, data_repo:str = 'OpenML', use_holdout =False,
                 ):
        super(Classification_Configuration_Space, self).__init__(task_id, rng, data_path,data_repo,use_holdout)

        # Initializer function per group.
        self.initializers = {}

        # Initialize the function per group using the specified group list.
        for model in model_list:
            if XGB_NAME == model:
                self.initializers[model] = self.init_xgb
            elif LINEAR_SVM_NAME == model:
                self.initializers[model] = self.init_linear_svm
            elif model == RBF_SVM_NAME:
                self.initializers[model] = self.init_rbf_svm
            elif DT_NAME == model:
                self.initializers[model] = self.init_dt
            elif RF_NAME == model:
                self.initializers[model] = self.init_rf
            else:
                raise RuntimeError
                
    


    def unraveling_hyper_parameter(self, dictionary_of_hp):
        """
            Get's the transformation, lower, upper bound for the requested hyper-parameter
        """
        transformation = dictionary_of_hp[TRANSFORM_TYPE]
        lower_bound = dictionary_of_hp[LOWER_BOUND]
        upper_bound = dictionary_of_hp[UPPER_BOUND]
        hp_type = dictionary_of_hp[HP_TYPE]

        return hp_type, transformation, lower_bound, upper_bound
    

    """
    
    Mango ConfigSpace Setup
    
    """



    def mango_specific_hp(self, hp_type, transformation, lower_bound, upper_bound):

        if transformation == None and hp_type == INTEGER_HP:
            return range(lower_bound, upper_bound)
        
        elif transformation == LOG_UNIFORM and hp_type == FLOAT_HP:
            return loguniform(lower_bound, upper_bound)
        
        elif transformation == UNIFORM and hp_type == FLOAT_HP:
            return uniform(lower_bound, upper_bound)
        
        else:

            raise ValueError 
        

    def mango_create_per_group_configuration_space(self, model_name ,seed : Union[int, None] = None) -> dict:
        """
        
        This method creates a sub_configuration space for each group
        
        """
        group_config_space_dictionary = {}

        for i in hyper_parameters_dictionary[model_name]:
            hp_dict = hyper_parameters_dictionary[model_name][i]
            hp_type ,transformation, lower_bound, upper_bound = self.unraveling_hyper_parameter(hp_dict)
            group_config_space_dictionary[i] = self.mango_specific_hp(hp_type, transformation, lower_bound, upper_bound)

        return group_config_space_dictionary

    
    def get_mango_config_space(self,seed=None) -> dict:
        config_dict = {}

        for model_name in model_list:
            config_dict[model_name] = self.mango_create_per_group_configuration_space(model_name, seed)
            
        return config_dict


    # SHOULD CHANGE
    def mango_init_model(self, config: Union[CS.Configuration, Dict],fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None,n_feat=1,model_type=None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """

        assert model_type!=None and model_type in model_list 

        rng = self.rng if rng is None else rng
        rng = rng if (rng is None or isinstance(rng, int)) else self.seed

        tmp_config = config.copy()

        # Get the model.
        return self.initializers[model_type](tmp_config,rng)
        



    """
    
    SMAC - MY BO Configuration Space Setup.
    
    """

    def smac_specific_hp(self,name, hp_type, transformation, lower_bound, upper_bound):

        if transformation == None and hp_type == INTEGER_HP:
            return CS.UniformIntegerHyperparameter(name, lower=lower_bound, upper=upper_bound, log=False)
        elif transformation == LOG_UNIFORM:

            if hp_type == INTEGER_HP:
                print('Warning You shouldn"t use log transformed integers...')
                return CS.UniformIntegerHyperparameter(name, lower=lower_bound, upper=upper_bound, log=True)
            elif hp_type == FLOAT_HP:
                return CS.UniformFloatHyperparameter(name, lower=lower_bound, upper=upper_bound, log=True)
            
        elif transformation == UNIFORM:
            return CS.UniformFloatHyperparameter(name, lower=lower_bound, upper=upper_bound, log=False)
        else:
            raise ValueError 
        
    def smac_create_per_group_configuration_space(self, model_name ,seed : Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        
        This method creates a sub_configuration space for each group
        
        """
        list_of_hps = []

        for i in hyper_parameters_dictionary[model_name]:
            hp_dict = hyper_parameters_dictionary[model_name][i]
            hp_type ,transformation, lower_bound, upper_bound = self.unraveling_hyper_parameter(hp_dict)
            list_of_hps.append( self.smac_specific_hp(i, hp_type, transformation, lower_bound, upper_bound) )

        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(list_of_hps)
        return cs

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        config_dict = {}

        model_list = [XGB_NAME,LINEAR_SVM_NAME,RF_NAME,DT_NAME,RBF_SVM_NAME]
        cs.add_hyperparameters([CS.CategoricalHyperparameter('model', choices = model_list)])

        for model_name in model_list:
            per_group_config_space = self.smac_create_per_group_configuration_space(model_name, seed)
            cs.add_configuration_space(prefix="",delimiter="",configuration_space=per_group_config_space,parent_hyperparameter={"parent": cs["model"], "value": model_name})
            config_dict[model_name] = per_group_config_space

        return cs,config_dict
    


    """
    
    Optuna Configuration Space Setup
    
    """
    def optuna_specific_hp(self, trial, name, hp_type, transformation, lower_bound, upper_bound):

        if transformation == None and hp_type == INTEGER_HP:
            return trial.suggest_int(name, lower_bound, upper_bound, log=False)
        elif transformation == LOG_UNIFORM:

            if hp_type == INTEGER_HP:
                print('Warning You shouldn"t use log transformed integers...')
                return trial.suggest_int(name, lower_bound, upper_bound, log=True)
            elif hp_type == FLOAT_HP:
                return trial.suggest_float(name, lower_bound, upper_bound, log=True)
            
        elif transformation == UNIFORM:
            return trial.suggest_float(name, lower_bound, upper_bound, log=False)
        else:
            raise ValueError  
        
    def get_optuna_per_group_space(self, model_name, trial, seed: Union[int, None] = None) :
        """
        Parameter space to be optimized --- contains the hyperparameters
        """
        
        group_config_space_dictionary = {}

        for hp in hyper_parameters_dictionary[model_name]:
            hp_dict = hyper_parameters_dictionary[model_name][hp]
            hp_type ,transformation, lower_bound, upper_bound = self.unraveling_hyper_parameter(hp_dict)
            group_config_space_dictionary[hp]  = self.optuna_specific_hp(trial, hp, hp_type, transformation, lower_bound, upper_bound) 

        return group_config_space_dictionary


    def get_optuna_space(self,trial,rng):

        rng = self.rng if rng is None else rng
        rng = rng if (rng is None or isinstance(rng, int)) else self.seed



        model_list = [XGB_NAME,LINEAR_SVM_NAME,RF_NAME,DT_NAME,RBF_SVM_NAME]
        # 2. Suggest values for the hyperparameters using a trial object.
        classifier_name = trial.suggest_categorical('model', model_list)
        param_dict = self.get_optuna_per_group_space(classifier_name,trial,seed=self.rng)
        param_dict['model'] = classifier_name
        return param_dict 
    


    """
    
    HyperOpt Configuration Space Setup.
    

    """

    def hyperopt_init_model(self, config: Union[CS.Configuration, Dict],
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        rng = rng if (rng is None or isinstance(rng, int)) else self.seed

        tmp_config = config['algorithm'].copy()
        model_type = tmp_config.pop('type')

        assert model_type in model_list and model_type != None

        return self.initializers[model_type](tmp_config,rng)
        

    def hyperopt_specific_hp(self, name, hp_type, transformation, lower_bound, upper_bound):
        if transformation == None and hp_type == INTEGER_HP:
            return hp.randint(name, lower_bound, upper_bound)
        elif transformation == LOG_UNIFORM:

            if hp_type == INTEGER_HP:
                print('Warning You shouldn"t use log transformed integers...')
                return RuntimeError
            elif hp_type == FLOAT_HP:
                return hp.loguniform(name,np.log(lower_bound),np.log(upper_bound))
            
        elif transformation == UNIFORM:
            return hp.uniform(name, lower_bound, upper_bound)
        else:
            raise ValueError  


    def get_hyperopt_per_group_space(self, model_name, seed: Union[int, None] = None) :
        """

        Parameter space to be optimized --- contains the hyperparameters

        """
        
        group_config_space_dictionary = {}

        for hp in hyper_parameters_dictionary[model_name]:
            hp_dict = hyper_parameters_dictionary[model_name][hp]
            hp_type ,transformation, lower_bound, upper_bound = self.unraveling_hyper_parameter(hp_dict)
            group_config_space_dictionary[hp]  = self.hyperopt_specific_hp(hp, hp_type, transformation, lower_bound, upper_bound) 

        group_config_space_dictionary['model'] = model_name
        
        return group_config_space_dictionary  
    


    def get_hyperopt_configspace(self, seed: Union[int, None] = None):
        
        config_per_group = []

        # Define the search space for all algorithms
        for model_name in model_list:
            config_per_group.append(self.get_hyperopt_per_group_space( model_name, seed))


        search_space = { 'model': hp.choice('model', config_per_group)}

        return search_space

    """
    Initializers of models given a configuration variable
    config: dict or configuration object
    rng : random_state
    """


    def init_linear_svm(self, config : Union[CS.Configuration, Dict], rng: Union[int, np.random.RandomState, None],n_feat = None):
        new_config = config.copy()
        new_config['C'] = new_config.pop('linear_C')
        model = SVC(**new_config,random_state=rng,probability=True)
        return model 


    def init_rbf_svm(self, config : Union[CS.Configuration, Dict], rng: Union[int, np.random.RandomState, None ], n_feat =None):
        new_config = config.copy()
        new_config['C'] = new_config.pop('rbf_C')
        new_config['gamma'] = new_config.pop('rbf_gamma')
        model = SVC(**new_config,random_state=rng,probability=True)
        return model
    

    def init_rf(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None, n_feat = None):

        assert n_feat != None
        new_config = config.copy()
        print(f"Features prev, root {n_feat},{config['max_features']}")
        new_config["max_features"] = int(np.rint(np.power(n_feat, config["max_features"])))
        print(f'After #feat {new_config["max_features"]}')
        model = RandomForestClassifier(**new_config,  bootstrap=True,random_state=rng,n_jobs=-1)
        return model

    
    def init_xgb(self,config : Union[CS.Configuration, Dict],rng : Union[int, np.random.RandomState, None] = None, n_feat = None):
        extra_args = dict(
            booster="gbtree",
            objective="binary:logistic",
            random_state=rng,
            eval_metric = ['auc'],
            use_label_encoder=False,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
        )
        if self.n_classes > 2:
            #Very important here. We need to use softproba to get probabilities out of XGBoost
            extra_args["objective"] = 'multi:softproba' #"multi:softmax"
            extra_args.update({"num_class": self.n_classes})

        #Handle special case here
        new_config = config.copy() 

        new_config['max_depth'] = new_config.pop('XGB_max_depth')
        new_config['n_estimators'] = new_config.pop('XGB_n_estimators')
        model = xgb.XGBClassifier(**new_config,**extra_args)
        return model


    def init_dt(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None , n_feat = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        assert n_feat != None
        
        new_config = config.copy()
       
        new_config['max_depth'] = new_config.pop('dt_max_depth')
        new_config['min_samples_leaf'] = new_config.pop('dt_min_samples_leaf')
        new_config['min_samples_split'] = new_config.pop('dt_min_samples_split')
        max_features_root =  new_config.pop('dt_max_features')
        new_config["max_features"] = int(np.rint(np.power(n_feat, max_features_root)))
        model = DecisionTreeClassifier(**new_config,random_state=rng)

        return model

    
    def init_model(self, config: Union[CS.Configuration, Dict],fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None,n_feat=None) :
        """ Function that returns the model initialized based on the configuration and fidelity
        """

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()


        rng = self.rng if rng is None else rng
        rng = rng if (rng is None or isinstance(rng, int)) else self.seed

        tmp_config = config.copy()
        
        model_type = tmp_config.pop('model')

        assert model_type in model_list and model_type != None

        return self.initializers[model_type](tmp_config,rng,n_feat=n_feat)
    


    

    