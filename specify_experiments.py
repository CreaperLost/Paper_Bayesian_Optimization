OPTUNA = 'Optuna'
HYPEROPT = 'HyperOpt'
SMAC = 'Smac'
RANDOM_SEARCH = 'Random_Search'
MANGO = 'Mango'
RF_LOCAL = 'RF_Local'

import random

OPTIMIZERS = [
    RF_LOCAL,
    MANGO,
    
]
"""
    MANGO,
    HYPEROPT,
    SMAC,
    OPTUNA,
    RANDOM_SEARCH, 
    
    
"""

"""
 {'name' : 'GP',
     'SURROGATE' : 'GP',
     'ACQ_GRID': 10000,
     'LOCAL_SEARCH':False,
     'OUTPUT_TRANSFORMATION' : None,
     'N_INIT'  :20,
     'ADAPTIVE': False,
    },
    {'name' : 'RF',
     'SURROGATE' : 'RF',
     'ACQ_GRID': 10000,
     'LOCAL_SEARCH':False,
     'OUTPUT_TRANSFORMATION' : None,
     'N_INIT'  :20,
     'ADAPTIVE': False,
    },
    
 {'name' : 'RF_GRID',
     'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':False,
     'OUTPUT_TRANSFORMATION' : None,
     'N_INIT'  :20,
     'ADAPTIVE': False,
    },
    {'name': 'RF_GRID_LOCAL',
     'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':True,
     'OUTPUT_TRANSFORMATION' : None,
     'N_INIT'  :20,
     'ADAPTIVE': False,
    },
 {'name' : 'RF_GRID_LOCAL_TRANS',
     'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':True,
     'OUTPUT_TRANSFORMATION' : True,
     'N_INIT'  :20,
     'ADAPTIVE': False,
    },
    {'name':'RF_GRID_LOCAL_INIT',
     'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':True,
     'OUTPUT_TRANSFORMATION' : False,
     'N_INIT'  :10,
     'ADAPTIVE': False,
    },
    {'name':'RF_GRID_LOCAL_BIG_INIT',
     'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':True,
     'OUTPUT_TRANSFORMATION' : False,
     'N_INIT'  :30,
     'ADAPTIVE': False,
    },
"""


ABLATION_CONFIG_LIST = [
    
   
]

"""

 
   
    
    
    {'name' :'RF_GRID_LOCAL_TRANS_INIT_ADAPTIVE',
     'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':True,
     'OUTPUT_TRANSFORMATION' : True,
     'N_INIT'  :20,
     'ADAPTIVE': True,
    }


"""



#   


FULL_CLASSIFICATION_AUTOML_LIST = [3,  11, 12,  14,  15, 16,
                 18, 22, 23, 28, 29,  31, 37, 43, 45, 49, 53, 2074, 2079, 
                 3021, 3022, 3481,3549, 3560, 3902, 
                3903, 3913, 3917, 3918, 9946, 9952, 9957, 
                9960, 9964, 9971, 9976, 9978, 9981, 9985, 
                10093, 10101, 14954, 14969, 125920, 125922, 
                146800, 146817, 146819, 146820, 146821, 146822, 
                146824, 167140, 167141, 146818, 168910, 168911, 168912]

N_SEEDS = [1,2,3,4,5]

# used for plotting mainly.
N_FOLDS = [0,1,2,3,4]

N_INIT = 10
N_MAXIMUM = 350


ABLATION = 'ABLATION'
COMPARISON = 'COMPARISON'


EXPERIMENT = ABLATION


ABLATION_DATASETS = [ 2079,  168912,  9960, 31,  125920, 14954, 49, 28, 29 , 3 ,11, 168911,10093,3902,146822] 

"""

These have run.
[ 2079,  168912,  9960, 31,  125920, 14954, 49, 28, 29 , 3 ,11 168911,10093,3902,146822] 

"""

# random.choices(FULL_CLASSIFICATION_AUTOML_LIST, k=15)

FULL_EXPERIMENTS_DATASETS = [12, 14, 15, 16, 18, 22, 23, 37, 43, 45, 53, 
                             2074, 3021, 3022, 3481, 3549, 3560, 3902,
                               3903, 3913, 3917, 3918, 9946, 9952, 9957, 
                               9964, 9971, 9976, 9978, 9981, 9985, 10093, 
                               10101, 14969, 125922, 146800, 146817, 146819, 
                               146820, 146821, 146822, 146824, 167140, 167141, 146818, 168910]
#list(filter(lambda i: i not in ABLATION_DATASETS, FULL_CLASSIFICATION_AUTOML_LIST))


