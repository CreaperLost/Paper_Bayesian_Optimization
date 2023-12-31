"""Input
oos_matrix: An array of size (# samples X # configurations). It contains the (out of sample) predictions for all samples by all models/configurations.
labels: The true labels of the outcome.
analysis_type: Either 'regression' (using R^2 as the metric) or 'classification' (AUC). You can easily change the metrics in the code right at the top of it.
folds: A 1d array of length (# samples) containing the fold membership of each sample. Note that the code should only work for a simple CV protocol, so no Repeated CV here.
bbc_type: The default value (pooled) is the one used in the paper you have read. The other two options are 'averaged', where the fold membership info is also used, and 'fold', which is our new idea for the FoldBBC.
iterations: Number of bootstrap iterations in BBC.

Output
The bootstrap BBC values, an array of length = iterations. You can derive a point estimation from this (mean, median, etc.) or a confidence interval (this is supposed to be problematic, which is also the main - but not the only - motivation for our planned paper).
"""
from sklearn.metrics import roc_auc_score,r2_score
import numpy as np
import pandas as pd

# BBC calculation function
def bbc(oos_matrix, labels, analysis_type, folds, bbc_type='pooled', iterations=1000,multi_class = False):
    
    bbc_distribution = []


    # Select the appropriate metric.
    if analysis_type == 'classification':
        metric = roc_auc_score # you can replace this to use a different metric if you like
    else:
        metric = r2_score # you can replace this to use a different metric if you like

    N = len(labels)
    C = oos_matrix.shape[1]

    out_of_bag_performances = [] # list to store the bootstrap values for each iteration
    if bbc_type == 'pooled':
        for i in range(iterations):
            #print(f'iter {i}')
            in_bag_indices = sorted(np.random.choice(N, N, replace=True)) # Bootstrap sampling with replacement
            out_of_bag_indices = list(set(list(range(N))) - set(in_bag_indices)) # Remaining samples that will be used to calculate the performance of the winner configuration
        
            in_bag_performances = [] # List to store the configuration performances on the in_bag_indeces samples
            for j in range(C):
                if multi_class == True:
                    new_df =np.array([np.array(xi) for xi in oos_matrix[in_bag_indices,j]])
                    in_bag_performances.append(metric(labels[in_bag_indices], new_df, multi_class='ovr'))
                else:
                    in_bag_performances.append(metric(labels[in_bag_indices], oos_matrix[in_bag_indices, j]))
            winner_configuration = np.argmax(in_bag_performances) # Best configuration on the in_bag_indices data
            if multi_class == True:
                new_df =np.array([np.array(xi) for xi in oos_matrix[out_of_bag_indices, winner_configuration]])
                out_of_bag_performances.append(metric(labels[out_of_bag_indices],new_df,multi_class='ovr'))
            else:
                out_of_bag_performances.append(metric(labels[out_of_bag_indices],oos_matrix[out_of_bag_indices, winner_configuration])) # Performance of best configuration on the out_of_bag_indices data
        bbc_distribution = out_of_bag_performances
    elif bbc_type == 'averaged': # This is a different version that takes into account the fold memberships of the samples and calculated the configuration performances by fold for the in_bag and out_of_bag
        fold_ids = np.unique(folds)
        for i in range(iterations):
            #print(f'iter {i}')
            in_bag_indices = sorted(np.random.choice(N, N, replace=True))
            out_of_bag_indices = list(set(list(range(N))) - set(in_bag_indices))
            in_bag_performances = []
            for j in range(C):
                in_bag_fold_performances = []
                for f in fold_ids:
                    index_selection = [ib for ib in in_bag_indices if folds[ib] == f]
                    if ((analysis_type == 'regression') | ((analysis_type == 'classification') & (len(np.unique(labels[index_selection])) > 1))) & (len(index_selection) > 1):
                        in_bag_fold_performances.append(metric(labels[index_selection], oos_matrix[index_selection, j]))
                in_bag_performances.append(np.mean(in_bag_fold_performances))
            winner_configuration = np.argmax(in_bag_performances)
            out_of_bag_fold_performances = []
            for f in fold_ids:
                index_selection = [ib for ib in out_of_bag_indices if folds[ib] == f]
                if ((analysis_type == 'regression') | ((analysis_type == 'classification') & (len(np.unique(labels[index_selection])) > 1))) & (len(index_selection) > 1):
                    out_of_bag_fold_performances.append(metric(labels[index_selection],oos_matrix[index_selection, winner_configuration]))
            out_of_bag_performances.append(np.mean(out_of_bag_fold_performances))
        bbc_distribution = out_of_bag_performances
    return bbc_distribution


import multiprocessing
import concurrent.futures


# BBC calculation function
def bbc_parallel(oos_matrix, labels, analysis_type, folds, bbc_type='pooled', iterations=1000,multi_class = False):
    
    # Select the appropriate metric.
    if analysis_type == 'classification':
        metric = roc_auc_score # you can replace this to use a different metric if you like
    else:
        metric = r2_score # you can replace this to use a different metric if you like

    N = len(labels)
    C = oos_matrix.shape[1]

    out_of_bag_performances = [] # list to store the bootstrap values for each iteration

    def process_iteration(i,oos_matrix, labels, N, C, multiclass, metric):
            in_bag_indices = sorted(np.random.choice(N, N, replace=True)) # Bootstrap sampling with replacement
            out_of_bag_indices = list(set(list(range(N))) - set(in_bag_indices)) # Remaining samples that will be used to calculate the performance of the winner configuration
        
            in_bag_performances = [] # List to store the configuration performances on the in_bag_indeces samples
            for j in range(C):
                if multi_class == True:
                    new_df =np.array([np.array(xi) for xi in oos_matrix[in_bag_indices,j]])
                    in_bag_performances.append(metric(labels[in_bag_indices], new_df, multi_class='ovr'))
                else:
                    in_bag_performances.append(metric(labels[in_bag_indices], oos_matrix[in_bag_indices, j]))
            winner_configuration = np.argmax(in_bag_performances) # Best configuration on the in_bag_indices data
            if multi_class == True:
                new_df =np.array([np.array(xi) for xi in oos_matrix[out_of_bag_indices, winner_configuration]])
                return metric(labels[out_of_bag_indices],new_df,multi_class='ovr')
            else:
                return metric(labels[out_of_bag_indices],oos_matrix[out_of_bag_indices, winner_configuration])
    

    fixed_args = (oos_matrix, labels, N, C, multi_class ,metric)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use a lambda function to pass the seed-specific argument
        out_of_bag_performances = list(executor.map(lambda i: process_iteration(i,*fixed_args), range(iterations)))

    
    return  out_of_bag_performances