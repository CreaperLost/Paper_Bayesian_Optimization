import pandas as pd
import numpy as np

import os 

from sklearn.metrics import roc_auc_score
from specify_experiments  import *

import matplotlib.pyplot as plt


plot_for = ABLATION


main_path = os.path.join(os.getcwd(), plot_for)



# Gets the hold-out-predictions.
def get_hold_out_performance(optimizer: str) -> dict:
    results_per_dataset_per_seed = {}
    for data_id in os.listdir(main_path):
        results_per_dataset_per_seed[data_id] = {}
        dataset_path  = os.path.join(main_path,data_id)
        for seed in os.listdir(dataset_path):
            results_per_dataset_per_seed[data_id][seed] = pd.DataFrame()
            configurations_path = os.path.join(dataset_path,seed,'Holdout',optimizer)
            for configuration_probs in os.listdir(configurations_path):
                
                file_name = os.path.join(configurations_path,configuration_probs)
                new_pd = pd.read_csv(file_name,index_col=0)['1']
                new_pd.rename(configuration_probs.split('.')[0],inplace=True)
                results_per_dataset_per_seed[data_id][seed] = pd.concat((results_per_dataset_per_seed[data_id][seed], new_pd),axis=1)
        
            col_names = list(results_per_dataset_per_seed[data_id][seed].columns)
            col_names = [int(str(x).split('C')[1]) for x in col_names]
            col_names.sort()
            col_names = ['C'+str(x) for x in col_names]
            results_per_dataset_per_seed[data_id][seed] = results_per_dataset_per_seed[data_id][seed].reindex(col_names, axis=1)
    return results_per_dataset_per_seed


# Gets the hold-out-predictions.
def get_hold_out_labels() -> dict:
    results_per_dataset_per_seed = {}
    for data_id in os.listdir(main_path):
        results_per_dataset_per_seed[data_id] = {}
        dataset_path  = os.path.join(main_path,data_id)
        for seed in os.listdir(dataset_path):
            labels_path = os.path.join(dataset_path,seed,'Holdout','labels','labels.csv')
            print(labels_path)
            results_per_dataset_per_seed[data_id][seed]  = pd.read_csv(labels_path,index_col=0)
    return results_per_dataset_per_seed

rf_local_results = get_hold_out_performance(RF_LOCAL)
labels_dataframe = get_hold_out_labels()




roc_scores_dict = {}
for data_id in rf_local_results:
    print(rf_local_results)
    roc_scores_dict[data_id] = {}
    for seed in rf_local_results[data_id]:
        
        preds = rf_local_results[data_id][seed]
        labels = labels_dataframe[data_id][seed]
        roc_score_array = [roc_auc_score(labels, preds[config]) for config in preds.columns]
        print(roc_score_array)
        df = pd.DataFrame(roc_score_array,columns= list(preds.columns))
        roc_scores_dict[data_id][seed] = df



for data_id in roc_scores_dict:

    for seed in roc_scores_dict[data_id]:
        print(roc_scores_dict[data_id][seed])