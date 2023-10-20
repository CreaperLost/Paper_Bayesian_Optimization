import numpy as np
import openml
import pandas as pd
from pathlib import Path
import warnings
#warnings.filterwarnings("ignore")
import os
import sys
sys.path.insert(0, '..')
from benchmark.hyper_parameters import *
from global_utilities.global_util import csv_postfix,parse_directory
from pathlib import Path
import numpy as np
from benchmark.configuration_space import Classification_Configuration_Space

from specify_experiments import *



from bo_algorithms.smac_hpo import SMAC_HPO
from bo_algorithms.random_search import Random_Search
from bo_algorithms.my_bo.RF_Local import RF_Local
from bo_algorithms.Mango import Mango
from bo_algorithms.Optuna import Optuna
from bo_algorithms.hyperopt import HyperOpt


from csv import writer
import time 


def run_benchmark_total(optimizers_used =[],bench_config={},save=True):
    assert optimizers_used != []
    assert bench_config != {}

    #Optimizer related
    n_init,max_evals = bench_config['n_init'],bench_config['max_evals']

    #Dataset related
    data_ids ,n_seeds = bench_config['data_ids'],bench_config['n_seeds']

    data_repo  = bench_config['data_repo']

    #Benchmark related fields
    type_of_bench = bench_config['type_of_bench'] 
    benchmark_name = bench_config['bench_name']
    benchmark_class = bench_config['bench_class']
    
    optimizers_list = optimizers_used

    assert optimizers_list != [] or optimizers_list != None


    main_directory = os.getcwd()
    
    
    for task_id in data_ids:
        task_id_str = 'Dataset' +str(task_id)
        for seed in n_seeds:
            Seed_id_str = 'Seed' + str(seed)
            for opt in optimizers_list: 
            
                score_per_optimizer_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Metric',data_repo,task_id_str,Seed_id_str,opt)
                total_time_per_optimizer_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Total_Time',data_repo,task_id_str,Seed_id_str,opt)
                config_per_group_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Configurations',data_repo,task_id_str,Seed_id_str,opt)

                benchmark_ = benchmark_class(task_id=task_id,seed=seed,optimizer=opt,experiment = EXPERIMENT)
                
                
                #Get the objective_function per fold.
                objective_function_per_fold = benchmark_.objective_function_per_fold
                
                print('Currently running ' + opt + ' on seed ' + str(seed) + ' dataset ' + str(task_id) )

                
                if opt == RANDOM_SEARCH:
                    #Get the benchmark.
                    objective_function = benchmark_.objective_function
                    #Get the config Space
                    configspace,config_dict = benchmark_.get_configuration_space()

                    Optimization = Random_Search(f=objective_function,configuration_space= configspace,n_init = n_init,max_evals= max_evals,random_seed=seed)
                # make sure smac has same intial configurations
                elif opt == SMAC:
                    #Get the benchmark objective.
                    smac_objective_function = benchmark_.smac_objective_function
                    #Get the config Space
                    configspace,config_dict = benchmark_.get_configuration_space()

                    Optimization = SMAC_HPO(configspace=configspace,config_dict=config_dict,task_id=task_id,
                    repo=data_repo,max_evals=max_evals,seed=seed,objective_function=smac_objective_function,n_workers=1,init_evals = 5*n_init)
                elif opt == RF_LOCAL:
                    #Get the benchmark.
                    objective_function = benchmark_.objective_function
                    #Get the config Space
                    configspace,config_dict = benchmark_.get_configuration_space()
                    Optimization = RF_Local(f=objective_function, model='RF' ,lb= None, ub =None ,configuration_space=config_dict,\
                                                n_init=n_init,max_evals=max_evals,initial_design=None,random_seed=seed,maximizer='Sobol_Local')
                elif opt == OPTUNA: # Optuna needs specifically the n_init over all groups
                    optuna_objective = benchmark_.optuna_objective
                    Optimization = Optuna(5*n_init,max_evals,seed,optuna_objective)
                elif opt == HYPEROPT: # HyperOpt needs specifically the n_init over all groups
                    hyperopt_objective = benchmark_.hyperopt_objective_function
                    hyperopt_space = benchmark_.get_hyperopt_configspace()
                    Optimization = HyperOpt(5*n_init,max_evals,seed,hyperopt_objective,hyperopt_space)
                elif opt == MANGO: #mango runs init per group ( * 5)
                    mango_config_space = benchmark_.get_mango_config_space()
                    mango_objectives  = { DT_NAME: benchmark_.mango_objective_dt,
                                        XGB_NAME: benchmark_.mango_objective_xgb,
                                        LINEAR_SVM_NAME : benchmark_.mango_objective_LinearSVM,
                                        RBF_SVM_NAME : benchmark_.mango_objective_RBFSVM,
                                        RF_NAME: benchmark_.mango_objective_RF}
                    Optimization = Mango(mango_config_space,n_init,max_evals,seed,mango_objectives)
                else: 
                    print(opt)
                    raise RuntimeError
                


                start_time = time.time()
                Optimization.run()
                m_time = time.time()-start_time
                print('Measured Total Time ',m_time)
                
                
                #Change this.
                y_evaluations = Optimization.fX
                total_time_evaluations = Optimization.total_time

                if save == True:
                    try:
                        Path(score_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(total_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(config_per_group_directory).mkdir(parents=True, exist_ok=True)
                    except FileExistsError:
                        print("Folder is already there")
                    else:
                        print("Folder is created there")
                        
                    pd.DataFrame(y_evaluations).to_csv( parse_directory([ score_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(total_time_evaluations).to_csv( parse_directory([ total_time_per_optimizer_directory, opt+csv_postfix ]))

                    if 'SMAC' in opt or 'ROAR' in opt or 'Hyperband' in opt:
                        #Save configurations and y results for each group.
                        for group in Optimization.save_configuration:
                            Optimization.save_configuration[group].to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    elif opt == 'Multi_RF_Local' or opt == 'MiniBatch_Progressive' or opt == 'Progr_Batch_BO' or opt == 'Progressive_BO' or opt =='Switch_BO' or opt =='GreedySM' or opt == 'RF_Local' or opt =='RF_Local_extensive' or opt == 'RF_Local_No_STD':
                        for group in Optimization.object_per_group:
                            X_df = Optimization.object_per_group[group].X_df
                            y_df = pd.DataFrame({'y':Optimization.object_per_group[group].fX})
                            pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    elif opt =='Random_Search':
                        #Save configurations and y results for each group.
                        for group in Optimization.fX_per_group:
                            X_df = Optimization.X_per_group[group]
                            y_df = pd.DataFrame({'y':Optimization.fX_per_group[group]})
                            pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                    
    

if __name__ == '__main__':
    
    opt_list = OPTIMIZERS
    
    #XGBoost Benchmark    
    xgb_bench_config =  {
        'n_init' : N_INIT,
        'max_evals' : N_MAXIMUM,
        'n_datasets' : 1000,
        'data_ids' :  FULL_CLASSIFICATION_AUTOML_LIST,
        'n_seeds' : N_SEEDS, 
        'type_of_bench': 'classification_experiments',
        'bench_name' :'GROUP',
        'bench_class' : Classification_Configuration_Space,
        'data_repo' : 'AutoML'
        } 
    run_benchmark_total(opt_list,xgb_bench_config)

