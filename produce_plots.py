import pandas as pd
import numpy as np
import os 
from specify_experiments  import *
from matplotlib.lines import Line2D

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


    


def get_cumulative_score_box(data:pd.DataFrame, norm_score:bool, min:pd.Series, max:pd.Series) -> tuple:

    max_column_indices = data.idxmax(axis=1)

    # Cumulative max per seed.
    cummax_per_seed = data.cummax(axis=1)

    return cummax_per_seed,max_column_indices

def get_cv_adaptive(path:str, opt:str, norm_score:bool, min:pd.Series, max:pd.Series)->pd.DataFrame:
    path_to_data =  path+f'/CV/{opt}.csv'
    data = pd.read_csv(path_to_data,index_col=0)
    return data.mean(axis=0)


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
        


def get_cv_score_box(path:str, opt:str, norm_score:bool, min:pd.Series, max:pd.Series)->pd.DataFrame:
    """
    Returns the average score per configuration for a dataset. (Across all seeds.)
    """
    path_to_data =  path+f'/CV/{opt}.csv'
    data = pd.read_csv(path_to_data,index_col=0)
    return get_cumulative_score_box(data, norm_score, min, max)

def get_holdout_score_box(path:str, opt:str, index_of_max:pd.Series) -> float:
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
    return holdout_scores


def get_bbc_scores_box(path:str, opt:str) -> float:
    """
    Returns the average score for the best_configuration for a dataset. 
    Across all seeds, in the bbc.
    """
    path_to_data =  path+f'/BBC/{opt}.csv'
    data = pd.read_csv(path_to_data,index_col=0)

    return data


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
    elif opt == 'RF_GRID_LOCAL_INIT':
        color = 'black'
    elif opt == 'RF_GRID_LOCAL-Ensemble2':
        color = 'yellow'
    elif opt == 'RF_GRID_LOCAL_BIG_INIT':
        color = 'brown'
    
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
        cumulative_score, index_of_max = get_cv_score_per_opt(path, opt, False, min, max)
        score = get_holdout_score_per_opt(path, opt, index_of_max)
        plt.plot(cumulative_score,label =f'CV Score{opt}',color = color_per_opt(opt))
        #plt.axhline(y=score,label=f'Holdout {opt}')
        
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
        #plt.savefig(f'figures/{titled,data_id}.png')
        plt.show()
        plt.clf()


SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


from scipy import stats

def do_Test(name_before, name_after, df):


    print(f'Test {name_after} vs {name_before}')

    before = df[name_before]
    after = df[name_after]
    
    print(f'Score after { np.round(np.mean(after),5)} before {np.round( np.mean(before),5 ) }')
    print(f'Mean Diff AFter - Before : { np.round(np.mean(after) - np.mean(before),5 ) }')
    """print('Median Diff',np.median(before)-np.median(after))
    print('Min Diff',np.min(before)-np.min(after))
    print('Max Diff',np.max(before)-np.max(after))"""
    print('Percentage change % (>0\% improvement)', 100*(np.mean(after) - np.mean(before))/np.mean(before))

    print(f"% Wins of {name_after} vs {name_before} : {count_of_wins(df, name_before, name_after)}")
   
    #_,p_val = stats.ttest_rel(after,before,alternative='greater')
    #_,p_val2 = stats.wilcoxon(after,before,alternative='greater')
    #print('Pvalue, ',np.round(p_val,6), np.round(p_val2,6))



def count_of_wins(dataframe, col1, col2):
    n_wins = np.sum(dataframe[col2] > dataframe[col1])
    win_perce = n_wins*100 / dataframe.shape[0]

    tie_res = np.sum( dataframe[col2] == dataframe[col1] )
    tie_perce = tie_res*100 / dataframe.shape[0]

    loss_res = np.sum(dataframe[col2] < dataframe[col1])
    loss_perce = loss_res*100 / dataframe.shape[0]
    return np.round(win_perce,3), np.round(tie_perce,3), np.round(loss_perce,3)

def create_plot_average(experiment:str, datasets:list, opt_list:list, norm_score: bool, keep_high: bool):


    cv_scores_per_opt = {}
    total_cv_scores_per_opt = {}
    holdout_scores_per_opt = {}
    for opt in opt_list:
        cv_scores_per_opt[opt] = []
        total_cv_scores_per_opt[opt] = []
        holdout_scores_per_opt[opt] = []
        for data_id in datasets:
            path = get_path_according_to_experiment(experiment, data_id)
            min, max  = get_min_max_per_dataset(path,opt_list=opt_list)
            # Get the CV cumulative score and the locations of maximums
            cumulative_score, index_of_max = get_cv_score_per_opt(path, opt, norm_score, min, max)
            score = get_holdout_score_per_opt(path, opt, index_of_max)

            cv_scores_per_opt[opt].append(cumulative_score[349])
            total_cv_scores_per_opt[opt].append(cumulative_score)
            holdout_scores_per_opt[opt].append(score)

        total_cv_scores_per_opt[opt] = pd.DataFrame(total_cv_scores_per_opt[opt])
        


    dataframe = pd.DataFrame(cv_scores_per_opt)


    if keep_high:

        # Threshold value
        threshold = 0.99

        # Check which rows have all values over the threshold
        rows_to_remove = (dataframe > threshold).all(axis=1)

        # Remove rows where all values are over the threshold
        dataframe = dataframe[~rows_to_remove]


        for opt in opt_list:
            total_cv_scores_per_opt[opt] = total_cv_scores_per_opt[opt][~rows_to_remove]
      
    

    print(dataframe)

    for opt in opt_list:
        total_cv_scores_per_opt[opt] = total_cv_scores_per_opt[opt].mean(axis=0)

    for idx,opt in enumerate(opt_list):

        #violin = plt.violinplot(cv_scores_per_opt[opt],positions=[idx], showextrema=True ,showmedians=False, vert=True)
        #violin['bodies'][0].set_facecolor(color_per_opt(opt))
        boxplot = plt.boxplot(dataframe[opt],positions=[idx],showmeans=False,patch_artist=True,labels=[opt],zorder=1)
        
        plt.scatter(y=dataframe[opt],x = [idx for _ in range(0,len(dataframe[opt]))],c='black',zorder=2)
        boxplot['boxes'][0].set_facecolor(color_per_opt(opt))
        boxplot['medians'][0].set_color('white')
        """violin2 = plt.violinplot(holdout_scores_per_opt[opt],positions=[idx+0.5], showmedians=True, vert=True)
        violin2['bodies'][0].set_facecolor(color_per_opt(opt))"""

    ticks = []
    for opt in opt_list:
        ticks.extend(['CV_'+opt])

    

    do_Test('GP','RF', dataframe)
    do_Test('RF','RF_GRID', dataframe)
    do_Test('RF_GRID','RF_GRID_LOCAL', dataframe)
    do_Test('RF_GRID_LOCAL','RF_GRID_LOCAL_TRANS', dataframe)
    do_Test('RF_GRID_LOCAL','RF_GRID_LOCAL_INIT', dataframe)
    do_Test('RF_GRID_LOCAL','RF_GRID_LOCAL_BIG_INIT', dataframe)
    do_Test('RF_GRID_LOCAL','RF_GRID_LOCAL-Ensemble2', dataframe)
    do_Test('RF_GRID_LOCAL','RF_GRID_LOCAL-Pooled', dataframe)


    #do_Test('RF_GRID_LOCAL-Ensemble','RF_GRID_LOCAL-Ensemble2', dataframe)

    #do_Test('RF_GRID_LOCAL','RF_GRID_LOCAL-Pooled', dataframe)
    
    #do_Test('RF_GRID_LOCAL','RF_GRID_LOCAL_TRANS_INIT_ADAPTIVE',dataframe)
    
    #'RF_GRID_LOCAL','RF_GRID_LOCAL-Pooled'


    """print("+===========================+")
    print('Test RF vs GP')
    do_Test(cv_scores_per_opt['GP'],cv_scores_per_opt['RF'])
    print(f"% Wins of RF over GP : {count_of_wins(dataframe,'GP','RF')}")



    print("+===========================+")
    print('Test RF SMALL GRID vs RF .')
    do_Test(cv_scores_per_opt['RF'],cv_scores_per_opt['RF_GRID'])
    print(f"% Wins of RF_GRID over RF : {count_of_wins(dataframe,'RF','RF_GRID')}")


    print("+===========================+")
    print('Test RF_GRID+local vs RF_GRID')
    do_Test(cv_scores_per_opt['RF_GRID'],cv_scores_per_opt['RF_GRID_LOCAL'])
    print(f"% Wins of RF_GRID_LOCAL over RF_GRID : {count_of_wins(dataframe,'RF_GRID','RF_GRID_LOCAL')}")


    print("+===========================+")
    print('Trans vs  no-Trans')
    do_Test(cv_scores_per_opt['RF_GRID_LOCAL'],cv_scores_per_opt['RF_GRID_LOCAL_TRANS'])
    print(f"% Wins of RF_Trans over RF_Local : {count_of_wins(dataframe,'RF_GRID_LOCAL','RF_GRID_LOCAL_TRANS')}")


    print("+===========================+")
    print('20init vs  10 init')
    do_Test(cv_scores_per_opt['RF_GRID_LOCAL'],cv_scores_per_opt['RF_GRID_LOCAL_INIT'])
    print(f"% Wins of RF-simple 20 init over RF_Local 10: {count_of_wins(dataframe,'RF_GRID_LOCAL','RF_GRID_LOCAL_INIT')}")



    print("+===========================+")
    print('20 vs  30 init')
    do_Test(cv_scores_per_opt['RF_GRID_LOCAL'],cv_scores_per_opt['RF_GRID_LOCAL_BIG_INIT'])
    print(f"% Wins of RF-simple init 30 over RF_Local 20 init : {count_of_wins(dataframe,'RF_GRID_LOCAL','RF_GRID_LOCAL_BIG_INIT')}")


    print("+===========================+")
    print('Local vs Ensemble')
    do_Test(cv_scores_per_opt['RF_GRID_LOCAL'],cv_scores_per_opt['RF_GRID_LOCAL-Ensemble'])
    print(f"% Wins of RF-Ensemble over RF_Local 20 init : {count_of_wins(dataframe,'RF_GRID_LOCAL','RF_GRID_LOCAL-Ensemble')}")"""

    """do_Test(cv_scores_per_opt['RF_GRID_LOCAL'],cv_scores_per_opt['RF_GRID_LOCAL-Ensemble2'])
    print(f"% Wins of RF-Ensemble2 over RF_Local 20 init : {count_of_wins(dataframe,'RF_GRID_LOCAL','RF_GRID_LOCAL-Ensemble')}")"""

    """do_Test(cv_scores_per_opt['RF_GRID_LOCAL-Ensemble'],cv_scores_per_opt['RF_GRID_LOCAL-Ensemble2'])
    print(f"% Wins of RF-Ensemble2 over ensembl1  : {count_of_wins(dataframe,'RF_GRID_LOCAL-Ensemble','RF_GRID_LOCAL-Ensemble2')}")
    """
    """do_Test(cv_scores_per_opt['RF_GRID_LOCAL'],cv_scores_per_opt['RF_GRID_LOCAL-Pooled'])
    print(f"% Wins of RF-Pooled over RF_Local 20 init : {count_of_wins(dataframe,'RF_GRID_LOCAL','RF_GRID_LOCAL-Pooled')}")
    """
    """print('Test GP vs RF_GRID With local search + small grid')
    do_Test(cv_scores_per_opt['GP'],cv_scores_per_opt['RF_GRID_LOCAL'])"""

    """do_Test(cv_scores_per_opt['GP'],cv_scores_per_opt['GP_0_mean'])
    print(f"% Wins of GP-0-Mean over GP: {count_of_wins(dataframe,'GP_0_mean','GP')}")"""
    
    plt.xlabel('Optimizer')
    plt.ylabel('AUC Score')
    plt.title('Ablation Study')
    plt.xticks(ticks = range(0,len(opt_list)),labels=ticks,rotation=60)
    plt.legend()
    plt.show()



    plt.clf()


    for idx,opt in enumerate(['GP','RF','RF_GRID_LOCAL','RF_GRID_LOCAL-Ensemble2']):
        plt.plot(total_cv_scores_per_opt[opt],color = color_per_opt(opt),linestyle='-', label=opt)

    xticks_interval = 50
    plt.xticks(np.arange(1, 351, xticks_interval))
    plt.ylim(0.92,0.935)
    plt.axvline(x=100, color='red', linestyle=':', label='Random Initial Configs')
    plt.xlabel('Configurations')
    plt.ylabel('Avg. AUC Score')
    plt.title('Ablation Study')
    plt.legend()
    plt.show()

 


def create_adaptive_plot(experiment:str, datasets:list, opt_list:list, keep_high: bool):
    total_cv_scores_per_opt = {}
    cv_scores_per_opt = {}
    for opt in opt_list:
        total_cv_scores_per_opt[opt] = []
        cv_scores_per_opt[opt] = []
        for data_id in datasets:
            path = get_path_according_to_experiment(experiment, data_id)
            min, max  = get_min_max_per_dataset(path,opt_list=opt_list)
            # Get the CV cumulative score and the locations of maximums
            if opt == 'RF_GRID_LOCAL_TRANS_INIT_ADAPTIVE':
                # this score is not actually cumulative
                cumulative_score = get_cv_adaptive(path, opt, False, min, max)
            else:
                cumulative_score, index_of_max = get_cv_score_per_opt(path, opt, False, min, max)
            total_cv_scores_per_opt[opt].append(cumulative_score)
            cv_scores_per_opt[opt].append(cumulative_score[349])

        total_cv_scores_per_opt[opt] = pd.DataFrame(total_cv_scores_per_opt[opt])

    dataframe = pd.DataFrame(cv_scores_per_opt)


    if keep_high:

        # Threshold value
        threshold = 0.99

        # Check which rows have all values over the threshold
        rows_to_remove = (dataframe > threshold).all(axis=1)

        # Remove rows where all values are over the threshold
        dataframe = dataframe[~rows_to_remove]


        for opt in opt_list:
            total_cv_scores_per_opt[opt] = total_cv_scores_per_opt[opt][~rows_to_remove]
      
    

    for opt in opt_list:
        total_cv_scores_per_opt[opt] = total_cv_scores_per_opt[opt].mean(axis=0)

        print(total_cv_scores_per_opt[opt])

    for idx,opt in enumerate(opt_list):
        plt.plot(total_cv_scores_per_opt[opt],color = color_per_opt(opt),linestyle='-', label=opt)

    xticks_interval = 50
    plt.xticks(np.arange(1, 351, xticks_interval))
    #plt.ylim(0.86,0.88)
    plt.axvline(x=100, color='red', linestyle=':', label='Random Initial Configs')
    plt.xlabel('Configurations')
    plt.ylabel('Avg. AUC Score')
    plt.title('Ablation Study')
    plt.legend()
    plt.show()


def holdout(experiment:str, datasets:list, opt_list:list, norm_score: bool):
    

    bbc_scores_per_opt = {}
    cv_scores_per_opt = {}
    holdout_scores_per_opt = {}
    for opt in opt_list:
        cv_scores_per_opt[opt] = []
        holdout_scores_per_opt[opt] = []
        bbc_scores_per_opt[opt] = []
        for data_id in datasets:
            path = get_path_according_to_experiment(experiment, data_id)
            min, max  = get_min_max_per_dataset(path,opt_list=opt_list)
            # Get the CV cumulative score and the locations of maximums
            cumulative_score, index_of_max = get_cv_score_box(path, opt, norm_score, min, max)
            cumulative_score = cumulative_score.iloc[:,349]
            print(cumulative_score)
            score = get_holdout_score_box(path, opt, index_of_max)
            print(score)
            bbc_score = get_bbc_scores_box(path, opt)
            print(bbc_score)
            cv_scores_per_opt[opt].append(cumulative_score)
            holdout_scores_per_opt[opt].append(score)
            bbc_scores_per_opt[opt].append(bbc_score)


    dataframe = pd.DataFrame(cv_scores_per_opt)
        

    for idx,opt in enumerate(opt_list):
        for i,data_id in enumerate(datasets):

            #violin = plt.violinplot(cv_scores_per_opt[opt],positions=[idx], showextrema=True ,showmedians=False, vert=True)
            #violin['bodies'][0].set_facecolor(color_per_opt(opt))
            boxplot = plt.boxplot(cv_scores_per_opt[opt][i],positions=[idx+i],showmeans=False,patch_artist=True,labels=['CV'],zorder=1)
            plt.scatter(y=cv_scores_per_opt[opt][i],x = [idx+i for _ in range(0,len(cv_scores_per_opt[opt][i]))],c='black',zorder=2)
            boxplot['boxes'][0].set_facecolor('blue')
            boxplot['medians'][0].set_color('white')

            boxplot2 = plt.boxplot(holdout_scores_per_opt[opt][i],positions=[idx+i+0.2],showmeans=False,patch_artist=True,labels=['holdout'],zorder=1)
            plt.scatter(y=holdout_scores_per_opt[opt][i],x = [idx+i+0.2 for _ in range(0,len(cv_scores_per_opt[opt][i]))],c='black',zorder=2)
            boxplot2['boxes'][0].set_facecolor('red')
            boxplot2['medians'][0].set_color('white')

            boxplot3 = plt.boxplot(bbc_scores_per_opt[opt][i],positions=[idx+i+0.4],showmeans=False,patch_artist=True,labels=['bbc'],zorder=1)
            plt.scatter(y=bbc_scores_per_opt[opt][i],x = [idx+i+0.4 for _ in range(0,len(cv_scores_per_opt[opt][i]))],c='black',zorder=2)
            boxplot3['boxes'][0].set_facecolor('green')
            boxplot3['medians'][0].set_color('white')
            
            
            

        ticks = []
        for opt in opt_list:
            for id in datasets:
                ticks.extend([id])



        import matplotlib.patches as mpatches

        red_patch = mpatches.Patch(color='red', label='Holdout')
        gr_patch = mpatches.Patch(color='green', label='BBC-CV')
        blue_patch = mpatches.Patch(color='blue', label='CV')
        plt.legend(handles=[red_patch,gr_patch,blue_patch])

        """print('Test GP vs RF_GRID With local search + small grid')
        do_Test(cv_scores_per_opt['GP'],cv_scores_per_opt['RF_GRID_LOCAL'])"""
            
        plt.xlabel('Dataset + Estimation Protocol')
        plt.ylabel('AUC Score')
        plt.title('Ablation Study')
        plt.xticks(ticks = range(0,len(opt_list)*(len(datasets))),labels=ticks,rotation=60)
        plt.show()



def holdout_per_dataset(experiment:str, datasets:list, opt_list:list, norm_score: bool):
    

    bbc_scores_per_opt = {}
    cv_scores_per_opt = {}
    holdout_scores_per_opt = {}
    for opt in opt_list:
        cv_scores_per_opt[opt] = []
        holdout_scores_per_opt[opt] = []
        bbc_scores_per_opt[opt] = []
        for data_id in datasets:
            path = get_path_according_to_experiment(experiment, data_id)
            min, max  = get_min_max_per_dataset(path,opt_list=opt_list)
            # Get the CV cumulative score and the locations of maximums
            cumulative_score, index_of_max = get_cv_score_box(path, opt, norm_score, min, max)
            cumulative_score = cumulative_score.iloc[:,349]
            #print(cumulative_score)
            score = get_holdout_score_box(path, opt, index_of_max)
            #print(score)
            bbc_score = get_bbc_scores_box(path, opt)
            #print(bbc_score)
            cv_scores_per_opt[opt].append(cumulative_score)
            holdout_scores_per_opt[opt].append(score)
            bbc_scores_per_opt[opt].append(bbc_score)


    dataframe = pd.DataFrame(cv_scores_per_opt)
        

    for idx,opt in enumerate(opt_list):
        for i,data_id in enumerate(datasets):

            #violin = plt.violinplot(cv_scores_per_opt[opt],positions=[idx], showextrema=True ,showmedians=False, vert=True)
            #violin['bodies'][0].set_facecolor(color_per_opt(opt))

            plt.plot(cv_scores_per_opt[opt][i].values, marker='o', label = 'CV', linestyle = 'dashed', color = 'blue')
            
            plt.plot(holdout_scores_per_opt[opt][i],  marker='o', label = 'Holdout', linestyle = 'dashed', color = 'red')
            plt.plot(bbc_scores_per_opt[opt][i].values, marker='o', label = 'BBC', linestyle = 'dashed', color = 'green')
            plt.title(str(data_id))
            plt.legend()
            plt.xticks(ticks = list(range(0,5)),labels = list(range(0,5)))
            plt.xlabel('Seed')
            plt.ylabel('AUC')
            
            plt.savefig(f'figures/bbc_per_data_{data_id}.png',bbox_inches='tight')
            plt.clf()

    



datasets  = ABLATION_DATASETS
experiment = ABLATION



# ,'RF_GRID_LOCAL_TRANS','RF_GRID_LOCAL-Ensemble','RF_GRID_LOCAL-Ensemble2',
#create_plot_per_dataset(experiment=experiment, datasets=datasets, opt_list=['GP_0_mean','GP',])

#create_adaptive_plot(experiment=experiment, datasets=datasets,opt_list=['RF_GRID_LOCAL','RF_GRID_LOCAL_TRANS_INIT_ADAPTIVE'],keep_high=True)

create_plot_average(experiment=experiment, datasets=datasets,opt_list=['GP','RF','RF_GRID','RF_GRID_LOCAL','RF_GRID_LOCAL_TRANS','RF_GRID_LOCAL_INIT','RF_GRID_LOCAL_BIG_INIT','RF_GRID_LOCAL-Pooled','RF_GRID_LOCAL-Ensemble2','RF_GRID_LOCAL_TRANS_INIT_ADAPTIVE'],norm_score=False, keep_high=False)

#create_plot_average(experiment=experiment, datasets=datasets,opt_list=['RF_GRID_LOCAL-Ensemble','RF_GRID_LOCAL-Ensemble2'],norm_score=False, keep_high=True)


# create_plot_average(experiment=experiment, datasets=datasets,  opt_list=['RF_GRID_LOCAL','RF_GRID_LOCAL-Pooled'],norm_score=False,keep_high=True)


#create_plot_average(experiment=experiment, datasets=datasets,  opt_list=['RF_GRID_LOCAL','RF_GRID_LOCAL_TRANS_INIT_ADAPTIVE'],norm_score=False, keep_high=True)

# 'RF_GRID_LOCAL_INIT','RF_GRID_LOCAL_BIG_INIT', 'RF','GP','RF_GRID', ,'RF_GRID_LOCAL-Pooled' ,'RF_GRID_LOCAL_TRANS', 'RF_GRID_LOCAL','RF_GRID_LOCAL-Ensemble','RF_GRID_LOCAL-Ensemble2', 'GP_0_mean'

"""holdout(experiment=experiment, datasets=datasets, 
                    opt_list=['RF_GRID_LOCAL'],norm_score=False)


holdout_per_dataset(experiment=experiment, datasets=datasets, 
                    opt_list=['RF_GRID_LOCAL'],norm_score=False)"""