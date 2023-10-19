import pandas as pd
import numpy as np

import os 

from sklearn.metrics import roc_auc_score
from specify_experiments  import *

import matplotlib.pyplot as plt
from global_utilities.bbc_cv import bbc

plot_for = ABLATION


main_path = os.path.join(os.getcwd(), plot_for)



def get_holdout_probs_per_dataset_per_seed_for_optimizer(dataset_id: int, seed: int, optimizer: str) -> pd.DataFrame:
    """
    Takes as input the dataset id, seed and optimizer name.
    Returns the probabilities for each case.
    """
    path_of_configuration_files = os.path.join(main_path,dataset_id,seed,'Holdout',optimizer)
    
    # Keep scores for the dataset 
    prob_df = pd.DataFrame()

    # iterate over all configuration files.
    for configuration_probs in os.listdir(path_of_configuration_files):
        # Get the configuration x.
        file_name = os.path.join(path_of_configuration_files,configuration_probs)
        # For binary-Classification.
        new_pd = pd.read_csv(file_name,index_col=0,float_precision='round_trip')['1']
        new_pd.rename(configuration_probs.split('.')[0],inplace=True)
        prob_df = pd.concat((prob_df,new_pd),axis=1)

    col_names = list(prob_df.columns)
    col_names = [int(str(x).split('C')[1]) for x in col_names]
    col_names.sort()
    col_names = ['C'+str(x) for x in col_names]
    prob_df = prob_df.reindex(col_names, axis=1)

    return prob_df

def get_hold_out_probs(optimizer: str) -> dict :
    results_per_dataset_per_seed = {}
    for data_id in os.listdir(main_path):
        results_per_dataset_per_seed[data_id] = {}
        dataset_path  = os.path.join(main_path,data_id)
        for seed in os.listdir(dataset_path):
            results_per_dataset_per_seed[data_id][seed] = get_holdout_probs_per_dataset_per_seed_for_optimizer(data_id, seed, optimizer)
    return results_per_dataset_per_seed


# Gets the hold-out-predictions.
def get_hold_out_labels() -> dict:
    results_per_dataset_per_seed = {}
    for data_id in os.listdir(main_path):
        results_per_dataset_per_seed[data_id] = {}
        dataset_path  = os.path.join(main_path,data_id)
        for seed in os.listdir(dataset_path):
            labels_path = os.path.join(dataset_path,seed,'Holdout','labels','labels.csv')
            results_per_dataset_per_seed[data_id][seed]  = pd.read_csv(labels_path,index_col=0,float_precision='round_trip')
            
    return results_per_dataset_per_seed


# Gets the hold-out-predictions.
def get_CV_labels() -> dict:
    results_per_dataset_per_seed = {}
    for data_id in os.listdir(main_path):
        results_per_dataset_per_seed[data_id] = {}
        dataset_path  = os.path.join(main_path,data_id)
        for seed in os.listdir(dataset_path):
            results_per_dataset_per_seed[data_id][seed] = {}
            fold_path = os.path.join(dataset_path,seed,'CV')
            for fold in os.listdir(fold_path):
                label_path=os.path.join(fold_path,fold,'labels','labels.csv')
                results_per_dataset_per_seed[data_id][seed][fold]  = pd.read_csv(label_path,index_col=0,float_precision='round_trip')
    return results_per_dataset_per_seed

def compute_holdout_roc_score_per_seed(preds, labels, seed) -> pd.Series:
    """
    Create a pd.Series dataframe with the roc value achieved by each configuration.
    """
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

def compute_holdout_roc_score(results, labels_df) -> dict:
    roc_scores_dict = {}
    for data_id in results:
        roc_scores_dict[data_id] = compute_holdout_roc_score_per_dataset(results[data_id], labels_df[data_id])
        
    # Adds up each one to the dataframe.
    resulting_roc_scores_per_dataset = {}
    for data_id in roc_scores_dict:
        df_auc = pd.DataFrame()
        for seed in roc_scores_dict[data_id]:
            df_auc  = df_auc.append(roc_scores_dict[data_id][seed])
        df_auc.sort_index(inplace=True)
        resulting_roc_scores_per_dataset[data_id] = df_auc
    return resulting_roc_scores_per_dataset





def get_CV_probs_per_dataset_per_seed_for_optimizer(dataset_id: int, seed: int, optimizer: str) -> dict:
    """
    Takes as input the dataset id, seed and optimizer name.
    Returns the probabilities for each fold.
    """
    path_for_folds = os.path.join(main_path,dataset_id,seed,'CV')
    
    Probs_per_fold = dict()
    # iterate over all configuration files.
    for folds in os.listdir(path_for_folds):
        # Keep scores for the fold. 
        prob_df = pd.DataFrame()
        path_of_configuration_files = os.path.join(path_for_folds,folds,optimizer)
        for configuration_probs in os.listdir(path_of_configuration_files):
            # Get the configuration x.
            file_name = os.path.join(path_of_configuration_files,configuration_probs)
            # For binary-Classification.
            new_pd = pd.read_csv(file_name,index_col=0,float_precision='round_trip')['1']
            new_pd.rename(configuration_probs.split('.')[0],inplace=True)
            prob_df = pd.concat((prob_df,new_pd),axis=1)

        col_names = list(prob_df.columns)
        col_names = [int(str(x).split('C')[1]) for x in col_names]
        col_names.sort()
        col_names = ['C'+str(x) for x in col_names]
        prob_df = prob_df.reindex(col_names, axis=1)
        Probs_per_fold[folds] = prob_df 

    return Probs_per_fold

def get_CV_probs(optimizer: str) -> dict :
    results_per_dataset_per_seed = {}
    for data_id in os.listdir(main_path):
        results_per_dataset_per_seed[data_id] = {}
        dataset_path  = os.path.join(main_path,data_id)
        for seed in os.listdir(dataset_path):
            results_per_dataset_per_seed[data_id][seed] = get_CV_probs_per_dataset_per_seed_for_optimizer(data_id, seed, optimizer)
    return results_per_dataset_per_seed


def compute_CV_roc_score_per_seed(preds, labels) -> pd.Series:
    """
    Create a pd.Series dataframe with the roc value achieved by each configuration.
    """
    df_per_fold = pd.DataFrame()
    for fold in preds:
        roc_score_array = []
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

def compute_CV_roc_score(results, labels_df) -> dict:
    roc_scores_dict = {}
    for data_id in results:
        roc_scores_dict[data_id] = compute_CV_roc_score_per_dataset(results[data_id], labels_df[data_id])
        
    # Adds up each one to the dataframe.
    resulting_roc_scores_per_dataset = {}
    for data_id in roc_scores_dict:
        df_auc = pd.DataFrame()
        for seed in roc_scores_dict[data_id]:
            df_auc  = df_auc.append(roc_scores_dict[data_id][seed])
        df_auc.sort_index(inplace=True)
        resulting_roc_scores_per_dataset[data_id] = df_auc
    return resulting_roc_scores_per_dataset


def get_opt_CV_score_per_seed(optimizer: str, data_id: int, seed: int):
    path_to_file = os.path.join(os.getcwd(),'classification_experiments','GROUP','Metric','AutoML',f'Dataset{data_id}',f'Seed{seed}',optimizer,f'{optimizer}.csv')
    results = pd.read_csv(path_to_file,index_col = 0,float_precision='round_trip')
    results = results.T
    results.index = [seed]
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

def merge_tables(probs_dictionary:dict) -> (dict,dict):
    per_dataset_config_matrix = {}
    per_dataset_fold_membership  ={}
    for data_id in probs_dictionary:
        per_dataset_config_matrix[data_id] = {}
        per_dataset_fold_membership[data_id]  ={}
        for seed in probs_dictionary[data_id]:
            per_dataset_config_matrix[data_id][seed] = {}
            per_dataset_fold_membership[data_id][seed]  ={}
            config_matrix = pd.DataFrame()
            fold_membership = []
            for fold in probs_dictionary[data_id][seed]:
                config_matrix = pd.concat((config_matrix,probs_dictionary[data_id][seed][fold]),axis=0)
                tmp_member = [fold] * probs_dictionary[data_id][seed][fold].shape[0] 
                fold_membership.extend(tmp_member)
            per_dataset_config_matrix[data_id][seed] = config_matrix
            per_dataset_fold_membership[data_id][seed]  = fold_membership

    return per_dataset_config_matrix,per_dataset_fold_membership



for opt in OPTIMIZERS:
    # Gets the labels and the predictions of an optimizer.
    rf_holdout_results = get_hold_out_probs(opt)
    holdout_labels_df = get_hold_out_labels()

    # Finds the hold_out scores per dataset for the optimizer.
    holdout_scores_per_dataset = compute_holdout_roc_score(rf_holdout_results, holdout_labels_df)

    #print(holdout_scores_per_dataset['3'])

    rf_CV_results = get_CV_probs(opt)
    labels_CV_df = get_CV_labels()

    config_matrix,fold_membership = merge_tables(rf_CV_results)
    labels_matrix,fold_membership_labels = merge_tables(labels_CV_df)


    corrected_score = 0
    for i in N_SEEDS:
        print(f"Are two fold memberships equal? {fold_membership['3'][str(i)] == fold_membership_labels['3'][str(i)]}")
        seed_score = bbc(np.array(config_matrix['3'][str(i)]),np.array(labels_matrix['3'][str(i)]),
                        'classification',fold_membership['3'][str(i)],iterations=50,bbc_type='pooled')
        corrected_score += np.mean(seed_score)
    corrected_score /= len(N_SEEDS)
    print(corrected_score)



    #print(rf_CV_results,labels_CV_df)

    CV_scores_per_dataset = compute_CV_roc_score(rf_CV_results, labels_CV_df)
    #print(labels_CV_df['3'])

    #print(CV_scores_per_dataset['3'])

    cvScoreOpt = aggregate_opt_CV_per_dataset(opt,'3')
    #print(cvScoreOpt)

    if True:
        for data_id in holdout_scores_per_dataset:

            if data_id not in CV_scores_per_dataset:
                continue

            holdoutMean = holdout_scores_per_dataset[data_id].mean( axis= 0)

            #print(holdoutMean)

            cvMean = CV_scores_per_dataset[data_id].mean( axis = 0)

            print(cvMean.tolist())


            cvOptMean = cvScoreOpt.mean( axis = 0)

            print(cvOptMean.tolist())

            #plt.errorbar(np.arange(1, len( holdoutMean.tolist()) +1) , holdoutMean.tolist(), yerr=holdoutStd.tolist(), fmt='-o', label='HoldOut')

            plt.plot(cvMean.tolist(),label= 'CV')
            plt.plot(cvOptMean.tolist(), label = 'CV_Opt')
            plt.legend()

            plt.show()
    elif True:
        for data_id in holdout_scores_per_dataset:
            if data_id not in CV_scores_per_dataset:
                continue
            holdoutMean = holdout_scores_per_dataset[data_id].mean( axis= 0).tolist()

            cvMean = CV_scores_per_dataset[data_id].mean( axis = 0).tolist()

            cvScoreOpt = cvScoreOpt.mean( axis = 0).tolist()
            print(cvScoreOpt)

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
            plt.scatter(len(cummulative_cv_score)-1,corrected_score,label = 'BCC-CV')
            plt.legend()
            plt.show()