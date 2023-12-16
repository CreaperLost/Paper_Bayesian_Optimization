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
from bo_algorithms.my_bo.RF_Local_Progressive import RF_Local_Progressive


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
            for opt_setting in optimizers_list: 
                

                opt = opt_setting['name']
            
                score_per_optimizer_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Metric',data_repo,task_id_str,Seed_id_str,opt)
                total_time_per_optimizer_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Total_Time',data_repo,task_id_str,Seed_id_str,opt)
                config_per_group_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Configurations',data_repo,task_id_str,Seed_id_str,opt)

                benchmark_ = benchmark_class(task_id=task_id,seed=seed,optimizer=opt,experiment = EXPERIMENT)
                
                
                #Get the objective_function per fold.
                objective_function_per_fold = benchmark_.objective_function_per_fold
                
                print('Currently running ' + opt + ' on seed ' + str(seed) + ' dataset ' + str(task_id) )

                #Get the benchmark.
                objective_function = benchmark_.objective_function
                #Get the config Space


                n_init = opt_setting['N_INIT']
                model = opt_setting['SURROGATE']
                grid_vals = opt_setting['ACQ_GRID']
                local_search_enabled = opt_setting['LOCAL_SEARCH']
                output_transformation_enabled = opt_setting['OUTPUT_TRANSFORMATION']
                adaptive = opt_setting['ADAPTIVE']

                print(n_init, model, grid_vals, local_search_enabled, output_transformation_enabled, adaptive)

                configspace,config_dict = benchmark_.get_configuration_space()
                if adaptive == True:
                    objective_function = benchmark_.objective_function_ensemble
                    Optimization = RF_Local_Progressive(f=objective_function, model=model ,lb= None, ub =None ,
                                            configuration_space=config_dict,\
                                            n_init=n_init,max_evals=max_evals,initial_design=None,
                                            random_seed=seed,maximizer='Sobol',
                                            local_search=local_search_enabled,
                                            grid_values = grid_vals,box_cox_enabled = output_transformation_enabled)
                elif adaptive == 'Progressive':
                    objective_function = benchmark_.objective_function_ensemble
                    Optimization = RF_Local_Progressive(f=objective_function, model=model ,lb= None, ub =None ,
                                            configuration_space=config_dict,\
                                            n_init=n_init,max_evals=max_evals,initial_design=None,
                                            random_seed=seed,maximizer='Sobol',
                                            local_search=local_search_enabled,
                                            grid_values = grid_vals,box_cox_enabled = output_transformation_enabled)
                elif model == 'Ensemble_RF' or model == 'Ensemble_RF2' or model == 'RF_Pooled':
                    objective_function = benchmark_.objective_function_ensemble
                    Optimization = RF_Local(f=objective_function, model=model ,lb= None, ub =None ,
                                            configuration_space=config_dict,\
                                            n_init=n_init,max_evals=max_evals,initial_design=None,
                                            random_seed=seed,maximizer='Sobol',
                                            local_search=local_search_enabled,
                                            grid_values = grid_vals,box_cox_enabled = output_transformation_enabled)
                else:
                    Optimization = RF_Local(f=objective_function, model=model ,lb= None, ub =None ,
                                            configuration_space=config_dict,\
                                            n_init=n_init,max_evals=max_evals,initial_design=None,
                                            random_seed=seed,maximizer='Sobol',
                                            local_search=local_search_enabled,
                                            grid_values = grid_vals,box_cox_enabled = output_transformation_enabled)
                
        
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
                        pass
                        #print("Folder is already there")
                    else:
                        pass
                        #print("Folder is created there")
                        
                    pd.DataFrame(y_evaluations).to_csv( parse_directory([ score_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(total_time_evaluations).to_csv( parse_directory([ total_time_per_optimizer_directory, opt+csv_postfix ]))

                    for group in Optimization.object_per_group:
                        X_df = Optimization.object_per_group[group].X_df
                        y_df = pd.DataFrame({'y':Optimization.object_per_group[group].fX})
                        pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                    pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    
    

if __name__ == '__main__':
    
    opt_list = ABLATION_CONFIG_LIST

    #XGBoost Benchmark    
    xgb_bench_config =  {
        'n_init' : None,
        'max_evals' : N_MAXIMUM,
        'n_datasets' : 1000,
        'data_ids' :  ABLATION_DATASETS, #FULL_CLASSIFICATION_AUTOML_LIST,
        'n_seeds' : N_SEEDS, 
        'type_of_bench': 'ablation_run',
        'bench_name' :'GROUP',
        'bench_class' : Classification_Configuration_Space,
        'data_repo' : 'AutoML'
        } 
    run_benchmark_total(opt_list,xgb_bench_config)
