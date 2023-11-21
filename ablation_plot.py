import pandas as pd
import numpy as np

import os 

from sklearn.metrics import roc_auc_score
from specify_experiments  import *

import matplotlib.pyplot as plt
from global_utilities.bbc_cv import bbc,bbc_parallel

plot_for = ABLATION

main_path = os.path.join(os.getcwd(), plot_for)

 # , 3,  15, 12 
configs = list(np.arange(0,N_MAXIMUM))

def get_probs_per_configuration(path_to_config_files):

    # Dataframe that holds all probability estimates!
    probabilty_df = pd.DataFrame()

    prob_lists = []

    r_val = None

    for config_prob in configs:
        file_name = os.path.join(path_to_config_files,f'C{config_prob}.csv')
        
        # For Binary Classification
        new_pd = pd.read_csv(file_name,index_col=0)


        #print(f'Got probabilities for {file_name} and config number {config_prob} and shape {new_pd.shape[1] }')

        if new_pd.shape[1] < 3 :
            #print(new_pd,type(new_pd))
            # Name the file == to the configuration number.
            new_pd = new_pd['1']
            new_pd.rename(config_prob,inplace=True)
            #new_pd.columns = [config_prob]
            # Append each prob per config to the dataframe.
            probabilty_df = pd.concat((probabilty_df,new_pd),axis=1)
            #print(probabilty_df)
            
        else:
            prob_lists.append(new_pd)

    if len(prob_lists) >0:
        return prob_lists
    else:
        return probabilty_df

def get_holdout_probs_per_dataset_per_seed_for_optimizer(dataset_id: int, seed: int, optimizer: str) -> pd.DataFrame:
    """
    Takes as input the dataset id, seed and optimizer name.
    Returns the probabilities for each case.
    """
    path_of_configuration_files = os.path.join(main_path,str(dataset_id),str(seed),'Holdout',optimizer)
    
    prob_df = get_probs_per_configuration(path_of_configuration_files)

    return prob_df

def get_hold_out_probs(optimizer: str, data_id:int) -> dict :
    results_per_dataset_per_seed = {}
    for seed in N_SEEDS:
        results_per_dataset_per_seed[seed] = get_holdout_probs_per_dataset_per_seed_for_optimizer(data_id, seed, optimizer)
    return results_per_dataset_per_seed

# Gets the hold-out-predictions.
def get_hold_out_labels(data_id) -> dict:

    results_per_dataset_per_seed = {}
    for seed in N_SEEDS:
        labels_path = os.path.join(main_path,str(data_id),str(seed),'Holdout','labels','labels.csv')
        results_per_dataset_per_seed[seed]  = pd.read_csv(labels_path,index_col=0,float_precision='round_trip')
            
    return results_per_dataset_per_seed


# Gets the hold-out-predictions.
def get_CV_labels(data_id:int ) -> dict:
    results_per_dataset_per_seed = {}
    for seed in N_SEEDS:
        results_per_dataset_per_seed[seed] = {}
        for fold in N_FOLDS:
            label_path = os.path.join(main_path, str(data_id), str(seed), 'CV',str(fold), 'labels', 'labels.csv')
            results_per_dataset_per_seed[seed][fold]  = pd.read_csv(label_path,index_col=0,float_precision='round_trip')
    return results_per_dataset_per_seed

def compute_holdout_roc_score_per_seed(preds, labels, seed) -> pd.Series:
    """
    Create a pd.Series dataframe with the roc value achieved by each configuration.
    """
    if isinstance(preds,list):
        roc_score_array = [roc_auc_score(labels, config_prob ,multi_class='ovr') for config_prob in preds]
        roc_series = pd.Series(roc_score_array,index= list(np.arange(0,len(preds))))
    else:
        roc_score_array = [roc_auc_score(labels, preds[config],multi_class='ovr') for config in preds.columns]

        # create a dataframe where each has a seed an index
        roc_series = pd.Series(roc_score_array,index= list(preds.columns))
    roc_series.name = seed
    
    return roc_series

def compute_holdout_roc_score_per_dataset(results, labels_df):
    """
    Computes the roc auc per seed for the dataset with the given id.
    returns : a dictionary with the roc_auc_scores per seed. (pd.Series format)
    """
    roc_scores_dict = {}
    for seed in results:
        roc_scores_dict[seed] = compute_holdout_roc_score_per_seed(results[seed], labels_df[seed], seed)
        
    return roc_scores_dict

def compute_holdout_roc_score(results, labels_df) -> pd.DataFrame :
    roc_score_per_seed = compute_holdout_roc_score_per_dataset(results, labels_df)
        
    # Adds up each one to the dataframe.
    df_auc = pd.DataFrame()
    for seed in roc_score_per_seed:
        df_auc  = df_auc.append(roc_score_per_seed[seed])
    df_auc.sort_index(inplace=True)

    return df_auc



def get_CV_probs_per_dataset_per_seed_for_optimizer(dataset_id: int, seed: int, optimizer: str) -> dict:
    """
    Takes as input the dataset id, seed and optimizer name.
    Returns the probabilities for each fold.
    """
    path_for_folds = os.path.join(main_path,str(dataset_id),str(seed),'CV')
    
    Probs_per_fold = dict()
    # iterate over all configuration files.
    for folds in N_FOLDS:
        # Keep scores for the fold. 
        path_of_configuration_files = os.path.join(path_for_folds,str(folds),optimizer)

        # Get a maxtrix with probabilities.
        Probs_per_fold[folds] = get_probs_per_configuration(path_of_configuration_files) 
        

    return Probs_per_fold

def get_CV_probs(data_id:int , optimizer: str) -> dict :
    results_per_dataset_per_seed = {}
    for seed in N_SEEDS:
        results_per_dataset_per_seed[seed] = get_CV_probs_per_dataset_per_seed_for_optimizer(data_id, seed, optimizer)
    return results_per_dataset_per_seed


def compute_CV_roc_score_per_seed(preds, labels) -> pd.Series:
    """
    Create a pd.Series dataframe with the roc value achieved by each configuration.
    """
    df_per_fold = pd.DataFrame()
    for fold in preds:
        roc_score_array = []
        if isinstance(preds[fold],list ):
            for config_prob in preds[fold]:
                score = roc_auc_score(labels[fold], config_prob ,multi_class='ovr')
                roc_score_array.append(score)
            roc_series = pd.Series(roc_score_array,index= list(np.arange(0,len(preds[fold]))))
        else:

            for config in preds[fold].columns:
                score = roc_auc_score(labels[fold], preds[fold][config],multi_class='ovr')
                roc_score_array.append(score)
            #create a dataframe where each has a seed an index
            roc_series = pd.Series(roc_score_array,index= list(preds[fold].columns))
        roc_series.name = fold
        df_per_fold = df_per_fold.append(roc_series)
    
    return df_per_fold.mean(axis=0)

def compute_CV_roc_score_per_dataset(results, labels_df):
    """
    Computes the roc auc per seed for the dataset with the given id.
    returns : a dictionary with the roc_auc_scores per seed. (pd.Series format)
    """
    roc_scores_dict = {}
    for seed in results:
        series_object = compute_CV_roc_score_per_seed(results[seed], labels_df[seed])
        series_object.name = seed
        roc_scores_dict[seed] = series_object
        
    return roc_scores_dict

def compute_CV_roc_score(results, labels_df) -> pd.DataFrame:
    roc_scores =  compute_CV_roc_score_per_dataset(results, labels_df)
        
    # Adds up each one to the dataframe.
    df_auc = pd.DataFrame()
    for seed in roc_scores:
        df_auc  = df_auc.append(roc_scores[seed])
    df_auc.sort_index(inplace=True)
    return df_auc


def get_opt_CV_score_per_seed(optimizer: str, data_id: int, seed: int):
    path_to_file = os.path.join(os.getcwd(),'ablation_run','GROUP','Metric','AutoML',f'Dataset{data_id}',f'Seed{seed}',optimizer,f'{optimizer}.csv')
    results = pd.read_csv(path_to_file,index_col = 0,float_precision='round_trip')
    #print(path_to_file)
    results = results.T
    results.index = [seed]
    #print(results)
    return results
     
def aggregate_opt_CV_per_dataset(optimizer: str, data_id: int):
    results = {}
    for seed in N_SEEDS:
        results[seed] = get_opt_CV_score_per_seed(optimizer,data_id,seed)

    df_auc = pd.DataFrame()
    for seed in results:
        df_auc  = pd.concat((df_auc,results[seed]),axis=0)

    df_auc.sort_index(inplace=True)

    return 1-df_auc


def turn_multiclass_list_into_pd(list_multiclass):
    """
    takes a list of a single dataframe.
    inside it stored probability estimates for each class...
    """
    new_df = pd.DataFrame()
    # for each configuration, 
    for config_num, df in enumerate(list_multiclass):
        # get probs in list of lists
        data_numpy = df.values.tolist()
        new_df[config_num] = data_numpy

    return new_df.copy()

def merge_tables(probs_dictionary:dict) -> (dict,dict):
    per_dataset_config_matrix = {}
    per_dataset_fold_membership  ={}
    for seed in probs_dictionary:
        per_dataset_config_matrix[seed] = {}
        per_dataset_fold_membership[seed]  ={}
        config_matrix = pd.DataFrame()
            
        fold_membership = []
        for fold in probs_dictionary[seed]:
            if isinstance(probs_dictionary[seed][fold],list):
                df_to_concat = turn_multiclass_list_into_pd(probs_dictionary[seed][fold])
            else:
                df_to_concat = probs_dictionary[seed][fold]
            config_matrix = pd.concat((config_matrix,df_to_concat),axis=0)
            tmp_member = [fold] * df_to_concat.shape[0] 
            fold_membership.extend(tmp_member)
        per_dataset_config_matrix[seed] = config_matrix
        per_dataset_fold_membership[seed]  = fold_membership

    return per_dataset_config_matrix,per_dataset_fold_membership

import multiprocessing
import concurrent.futures

import time 
datasets = ABLATION_DATASETS

def get_bbc_scores():
    for data_id in datasets:
        labels_CV_df = get_CV_labels(data_id)
        for opt in ['RF']:
            

            st = time.time()

            #cvScoreOpt = aggregate_opt_CV_per_dataset(opt,data_id)
            
            rf_CV_results = get_CV_probs(data_id ,opt)

            config_matrix,fold_membership = merge_tables(rf_CV_results)
            labels_matrix,fold_membership_labels = merge_tables(labels_CV_df)


            """def process_seed(i, config_matrix, labels_matrix, fold_membership,type_pf_bbc):
                #print(f'Seed: {i}')
                seed_score = bbc(np.array(config_matrix[i]), np.array(labels_matrix[i]),
                                'classification',
                                fold_membership[i],
                                iterations=50, bbc_type=type_pf_bbc, multi_class=True)
                return (i,np.mean(seed_score))
            
            # Define fixed arguments for the process_seed function
            fixed_args = (config_matrix, labels_matrix, fold_membership,'pooled')

            # Create a ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Use a lambda function to pass the seed-specific argument
                results = list(executor.map(lambda i: process_seed(i, *fixed_args), N_SEEDS))
            """
            bbc_per_seed  = []
            for i in N_SEEDS:
                results = bbc_parallel(np.array(config_matrix[i]), np.array(labels_matrix[i]),
                                'classification',
                                fold_membership[i],
                                iterations=500, bbc_type='pooled', multi_class=True)
                bbc_per_seed.append((i,np.mean(results)))

            df = pd.DataFrame(bbc_per_seed)
            print(df)

            path = f'ablation_scores/{data_id}/BBC/'

            # Check if directories exist, and create them if needed
            if not os.path.exists(path):
                os.makedirs(path)

            df.to_csv(f'{path}{opt}.csv',index=0)
            

            print(f'Time cost {time.time()-st}')
            # Sum up the results
            #corrected_score = sum(results)
            


            """# Define fixed arguments for the process_seed function
            averaged_fixed_args = (config_matrix, labels_matrix, fold_membership,'averaged')

            # Create a ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Use a lambda function to pass the seed-specific argument
                results = list(executor.map(lambda i: process_seed(i, *fixed_args), N_SEEDS))

            # Sum up the results
            averaged_corrected_score = sum(results)"""




def per_dataset_plot():
    for data_id in datasets:

        labels_CV_df = get_CV_labels(data_id)

        # Get Labels. Per seed for a dataset
        holdout_labels_df = get_hold_out_labels(data_id)

        for opt in ['RF_GRID_LOCAL_BIG_INIT']:

            # Gets the labels and the predictions of an optimizer.
            print(opt)
            rf_holdout_results = get_hold_out_probs(opt, data_id)

            # Finds the hold_out scores per dataset for the optimizer.
            holdout_scores_per_dataset = compute_holdout_roc_score(rf_holdout_results, holdout_labels_df)

            rf_CV_results = get_CV_probs(data_id ,opt)
            CV_scores_per_dataset = compute_CV_roc_score(rf_CV_results, labels_CV_df)


            print(holdout_scores_per_dataset)
            print(holdout_scores_per_dataset.shape)
            print(CV_scores_per_dataset)
            print(CV_scores_per_dataset.shape)

            CV_directory  = f'ablation_scores/{data_id}/CV/'
            Holdout_directory = f'ablation_scores/{data_id}/Holdout/'
            # Check if directories exist, and create them if needed
            if not os.path.exists(CV_directory):
                os.makedirs(CV_directory)

            # Check if directories exist, and create them if needed
            if not os.path.exists(Holdout_directory):
                os.makedirs(Holdout_directory)
            
            
            CV_scores_per_dataset.to_csv(CV_directory+f'{opt}.csv')
            holdout_scores_per_dataset.to_csv(Holdout_directory+f'{opt}.csv')

            

            """
            #cvScoreOpt = aggregate_opt_CV_per_dataset(opt,data_id)
            

            #config_matrix,fold_membership = merge_tables(rf_CV_results)
            #labels_matrix,fold_membership_labels = merge_tables(labels_CV_df)


            def process_seed(i, config_matrix, labels_matrix, fold_membership):
                print(f'Seed: {i}')
                seed_score = bbc(np.array(config_matrix[i]), np.array(labels_matrix[i]),
                                'classification',
                                fold_membership[i],
                                iterations=500, bbc_type='pooled', multi_class=True)
                return np.mean(seed_score)
            
            # Define fixed arguments for the process_seed function
            #fixed_args = (config_matrix, labels_matrix, fold_membership)


            
            #print(CV_scores_per_dataset)

            # Create a ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Use a lambda function to pass the seed-specific argument
                results = list(executor.map(lambda i: process_seed(i, *fixed_args), N_SEEDS))

            # Sum up the results
            corrected_score = sum(results)

            
            corrected_score = 0
            for i in N_SEEDS:
                print(f'Seed: {i}')
                seed_score = bbc(np.array(config_matrix[i]),np.array(labels_matrix[i]),
                                'classification',
                                fold_membership[i],
                                iterations=50,bbc_type='pooled',multi_class=True)
                corrected_score += np.mean(seed_score)
            corrected_score /= len(N_SEEDS)
            print(corrected_score)
            
            averaged_corrected_score = 0
            for i in N_SEEDS:
                averaged_seed_score = bbc(np.array(config_matrix[i]),np.array(labels_matrix[i]),
                                'classification',
                                fold_membership[i],
                                iterations=50,bbc_type='averaged',multi_class=True)
                averaged_corrected_score += np.mean(averaged_seed_score)
            averaged_corrected_score /= len(N_SEEDS)
            print(averaged_corrected_score)"""


            if False:
                    

                    holdoutMean = holdout_scores_per_dataset.mean( axis= 0)

                    #print(holdoutMean)

                    cvMean = CV_scores_per_dataset.mean( axis = 0)
                    cv_min = CV_scores_per_dataset.min( axis = 0)
                    cv_max = CV_scores_per_dataset.max( axis = 0)


                    cvOptMean = cvScoreOpt.mean( axis = 0)
                    

                    #print(set(sorted(cvOptMean.tolist()[:10])),set(sorted(cvMean.tolist()[:10])) )

                    #plt.errorbar(np.arange(1, len( holdoutMean.tolist()) +1) , holdoutMean.tolist(), yerr=holdoutStd.tolist(), fmt='-o', label='HoldOut')

                    plt.plot(cvMean.tolist(),label= 'CV-Mean',alpha=0.5)
                    plt.plot(cv_max.tolist(),label= 'CV-Max',alpha=0.5)
                    plt.plot(cv_min.tolist(),label= 'CV-Min',alpha=0.5)
                    plt.plot(cvOptMean.tolist(), label = 'CV_Opt',alpha=0.5)
                    plt.legend()
                    plt.title(f'{opt} , {data_id}')

                    print('lmao...')
                    
                    plt.savefig(f'figures/{opt}_{data_id}.png')
                    plt.clf()
                    #plt.show()
            elif False:
                    
                    holdoutMean = holdout_scores_per_dataset.mean( axis= 0).tolist()

                    cvMean = CV_scores_per_dataset.mean( axis = 0).tolist()

                    cvScoreOpt = cvScoreOpt.mean( axis = 0).tolist()
                    

                    # index of max in CV.
                    max_val = -1
                    max_index = -1
                    cummulative_cv_score = []
                    cummulative_cv_score_opt = []
                    cummulative_holdout_score = []
                    for index,value in enumerate(cvMean):
                        if value > max_val: 
                            max_val = value
                            max_index = index
                        cummulative_cv_score.append(cvMean[max_index])
                        cummulative_holdout_score.append(holdoutMean[max_index])

                    # Same but for the optimizers whatever.
                    max_val = -1
                    max_index = -1
                    cummulative_cv_score_opt = []
                    for index,value in enumerate(cvScoreOpt):
                        if value > max_val: 
                            max_val = value
                            max_index = index
                        cummulative_cv_score_opt.append(cvScoreOpt[max_index])

                    plt.plot(cummulative_cv_score, label = 'CV')
                    plt.plot(cummulative_cv_score_opt, label = 'CVopt')
                    plt.plot(cummulative_holdout_score, label = 'Holdout')
                    #plt.scatter(len(cummulative_cv_score)-1,corrected_score,label = 'Pool BCC-CV')
                    #plt.scatter(len(cummulative_cv_score)-1,averaged_corrected_score,label = 'Avg BCC-CV')
                    plt.legend()
                    plt.title(f'{opt} , {data_id}')
                    plt.savefig(f'figures/{opt}_{data_id}.png')
                    plt.clf()
                    #plt.show()


#per_dataset_plot()
get_bbc_scores()