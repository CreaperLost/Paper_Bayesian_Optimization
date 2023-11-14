import pandas as pd
import numpy as np
import os 
from specify_experiments  import *

import matplotlib.pyplot as plt

def get_path_according_to_experiment(experiment:str, dataid:int) -> str:
    """
    gets the path of the results according to data id and the experiment type
    returns a path as string.
    """
    if experiment == ABLATION:
        path  = f'ablation_scores/{dataid}'
    else:
        raise KeyError
    
    return path


def get_cumulative_score(data:pd.DataFrame, norm_score:bool, min:pd.Series, max:pd.Series) -> tuple:

    max_column_indices = data.idxmax(axis=1)

    # Cumulative max per seed.
    cummax_per_seed = data.cummax(axis=1)


    if norm_score:
        denominator = max.subtract(min)
        nominator = cummax_per_seed.subtract(min,axis=0)
        
        resulting_mean_cumsum = nominator.divide(denominator, axis = 0)

        resulting_mean_cumsum = resulting_mean_cumsum.mean(axis=0)
        
    else:
        # The average cumulative across all seeds.
        resulting_mean_cumsum = cummax_per_seed.mean(axis=0)

    return resulting_mean_cumsum,max_column_indices



def get_cv_score_per_opt(path:str, opt:str, norm_score:bool, min:pd.Series, max:pd.Series)->pd.DataFrame:
    """
    Returns the average score per configuration for a dataset. (Across all seeds.)
    """
    path_to_data =  path+f'/CV/{opt}.csv'
    data = pd.read_csv(path_to_data,index_col=0)
    return get_cumulative_score(data, norm_score, min, max)


def get_holdout_score_per_opt(path:str, opt:str, index_of_max:pd.Series) -> float:
    """
    Returns the average score for the best_configuration for a dataset. 
    Across all seeds, in the holdout.
    """
    path_to_data =  path+f'/Holdout/{opt}.csv'
    data = pd.read_csv(path_to_data,index_col=0)
    holdout_scores = []
    for seed,index in enumerate(index_of_max):
        score = data.iloc[seed,int(index)]
        holdout_scores.append(score)
    return np.mean(holdout_scores)
        


def color_per_opt(opt):
    color = 'black'
    if opt == 'RF':
        color = 'blue'
    elif opt == 'GP':
        color = 'red'
    elif opt == 'RF_GRID':
        color = 'green'
    elif opt == 'RF_GRID_LOCAL':
        color = 'purple'
    elif opt == 'RF_GRID_LOCAL_TRANS':
        color = 'grey'

    return color


def get_min_max_per_dataset(path, opt_list):

    min_per_opt = []
    max_per_opt = []
    for opt in opt_list:
        path_to_data =  path+f'/CV/{opt}.csv'
        data = pd.read_csv(path_to_data,index_col=0)
        resulting_min_cumsum = data.min(axis=1)
        min_per_opt.append(resulting_min_cumsum)
        resulting_max_cumsum = data.max(axis=1)
        max_per_opt.append(resulting_max_cumsum)

    

    min = pd.concat(min_per_opt,axis=1).min(axis=1)
    max = pd.concat(max_per_opt,axis=1).max(axis=1)    

    return min,max

def plot_score_per_dataset(path:str ,opt_list: list):
    max_score = -1
    min_score = 100


    # min and max per seed.. for a dataset across all optimizers
    min,max  = get_min_max_per_dataset(path, opt_list)
    
    
    for opt in opt_list:
        # Get the CV cumulative score and the locations of maximums
        cumulative_score, index_of_max = get_cv_score_per_opt(path, opt)
        score = get_holdout_score_per_opt(path, opt, index_of_max)
        plt.plot(cumulative_score,label =f'CV Score{opt}',color = color_per_opt(opt))
        plt.axhline(y=score,label=f'Holdout {opt}')
        
        max_score = max_score if max_score > cumulative_score.max() else cumulative_score.max()
        max_score = max_score if max_score > score.max() else score.max()

        
        min_score = min_score if min_score < cumulative_score[50] else cumulative_score[50]

#
    #plt.ylim(bottom = max_score - 0.01,top= max_score+0.001)

def get_title(data_id):
    name_data = pd.read_csv('AutoML+CC18.csv',index_col=0)
    return name_data[name_data['tid'] == data_id]['name'].values[0]


def create_plot_per_dataset(experiment:str, datasets:list, opt_list:list ):
    for data_id in datasets:
        path = get_path_according_to_experiment(experiment, data_id)
        plot_score_per_dataset(path, opt_list)
        titled = get_title(data_id)
        # Create new labels starting from 1 to 350 with a step of 50
        plt.xticks(np.arange(0, 350, 50))
        plt.legend()
        plt.title(f'Comparison for {data_id} named: {titled}')
        plt.savefig(f'figures/{titled,data_id}.png')
        plt.clf()


SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


from scipy import stats

def do_Test(before,after):
    print(f'Means  : { np.round(np.mean(after),5),np.round( np.mean(before),5 ) }')
    print(f'Mean Diff : { np.round(np.mean(after) - np.mean(before),5 ) }')
    """print('Median Diff',np.median(before)-np.median(after))
    print('Min Diff',np.min(before)-np.min(after))
    print('Max Diff',np.max(before)-np.max(after))"""
    print('Percentage change % ', 100*(np.mean(after) - np.mean(before))/np.mean(before))
   
    _,p_val = stats.ttest_rel(after,before,alternative='greater')
    _,p_val2 = stats.wilcoxon(after,before,alternative='greater')
    #print('Pvalue, ',np.round(p_val,6), np.round(p_val2,6))



def count_of_wins(dataframe, col1, col2):
    n_wins = np.sum(dataframe[col2] > dataframe[col1])
    win_perce = n_wins*100 / dataframe.shape[0]

    tie_res = np.sum( dataframe[col2] == dataframe[col1] )
    tie_perce = tie_res*100 / dataframe.shape[0]

    loss_res = np.sum(dataframe[col2] < dataframe[col1])
    loss_perce = loss_res*100 / dataframe.shape[0]
    return np.round(win_perce,3), np.round(tie_perce,3), np.round(loss_perce,3)

def create_plot_average(experiment:str, datasets:list, opt_list:list, norm_score: bool):


    cv_scores_per_opt = {}
    holdout_scores_per_opt = {}
    for opt in opt_list:
        cv_scores_per_opt[opt] = []
        holdout_scores_per_opt[opt] = []
        for data_id in datasets:
            path = get_path_according_to_experiment(experiment, data_id)
            min, max  = get_min_max_per_dataset(path,opt_list=opt_list)
            # Get the CV cumulative score and the locations of maximums
            cumulative_score, index_of_max = get_cv_score_per_opt(path, opt, norm_score, min, max)
            score = get_holdout_score_per_opt(path, opt, index_of_max)
            cv_scores_per_opt[opt].append(cumulative_score[349])
            holdout_scores_per_opt[opt].append(score)


    dataframe = pd.DataFrame(cv_scores_per_opt)
        
    for idx,opt in enumerate(opt_list):

        #violin = plt.violinplot(cv_scores_per_opt[opt],positions=[idx], showextrema=True ,showmedians=False, vert=True)
        #violin['bodies'][0].set_facecolor(color_per_opt(opt))
        boxplot = plt.boxplot(cv_scores_per_opt[opt],positions=[idx],showmeans=False,patch_artist=True,labels=[opt],zorder=1)
        
        plt.scatter(y=cv_scores_per_opt[opt],x = [idx for _ in range(0,len(cv_scores_per_opt[opt]))],c='black',zorder=2)
        boxplot['boxes'][0].set_facecolor(color_per_opt(opt))
        boxplot['medians'][0].set_color('white')
        """violin2 = plt.violinplot(holdout_scores_per_opt[opt],positions=[idx+0.5], showmedians=True, vert=True)
        violin2['bodies'][0].set_facecolor(color_per_opt(opt))"""

    ticks = []
    for opt in opt_list:
        ticks.extend(['CV_'+opt])
    
    print('Test RF vs GP')
    do_Test(cv_scores_per_opt['GP'],cv_scores_per_opt['RF'])
    print(f"% Wins of RF over GP : {count_of_wins(dataframe,'GP','RF')}")

    print('Test RF SMALL GRID vs RF .')
    do_Test(cv_scores_per_opt['RF'],cv_scores_per_opt['RF_GRID'])
    print(f"% Wins of RF_GRID over RF : {count_of_wins(dataframe,'RF','RF_GRID')}")

    print('Test RF_GRID+local vs RF_GRID')
    do_Test(cv_scores_per_opt['RF_GRID'],cv_scores_per_opt['RF_GRID_LOCAL'])
    print(f"% Wins of RF_GRID_LOCAL over RF_GRID : {count_of_wins(dataframe,'RF_GRID','RF_GRID_LOCAL')}")

    print('Trans vs  no-Trans')
    do_Test(cv_scores_per_opt['RF_GRID_LOCAL'],cv_scores_per_opt['RF_GRID_LOCAL_TRANS'])
    print(f"% Wins of RF_Trans over RF_Local : {count_of_wins(dataframe,'RF_GRID_LOCAL','RF_GRID_LOCAL_TRANS')}")

    """print('Test GP vs RF_GRID With local search + small grid')
    do_Test(cv_scores_per_opt['GP'],cv_scores_per_opt['RF_GRID_LOCAL'])"""
    
    plt.xlabel('Optimizer')
    plt.ylabel('AUC Score')
    plt.title('Ablation Study')
    plt.xticks(ticks = range(0,len(opt_list)),labels=ticks)
    plt.legend()
    plt.show()



datasets  = ABLATION_DATASETS
experiment = ABLATION

#create_plot_per_dataset(experiment=experiment, datasets=datasets, opt_list=['RF','GP'])


create_plot_average(experiment=experiment, datasets=datasets, opt_list=['RF','GP','RF_GRID','RF_GRID_LOCAL','RF_GRID_LOCAL_TRANS'],norm_score=False)