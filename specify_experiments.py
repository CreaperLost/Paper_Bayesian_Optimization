OPTUNA = 'Optuna'
HYPEROPT = 'HyperOpt'
SMAC = 'Smac'
RANDOM_SEARCH = 'Random_Search'
MANGO = 'Mango'
RF_LOCAL = 'RF_Local'



OPTIMIZERS = [
    MANGO,
]
"""
      RF_LOCAL,
    HYPEROPT,
    SMAC,
    OPTUNA,
    RANDOM_SEARCH,  
"""

"""
 
    
 

"""


ABLATION_CONFIG_LIST =[
    {'GP' : 
    {'SURROGATE' : 'GP',
     'ACQ_GRID': 10000,
     'LOCAL_SEARCH':False,
     'OUTPUT_TRANSFORMATION' : None,
     'N_INIT'  :20,
     'ADAPTIVE': False,
     }
    },
    {'RF' : 
    {'SURROGATE' : 'RF',
     'ACQ_GRID': 10000,
     'LOCAL_SEARCH':False,
     'OUTPUT_TRANSFORMATION' : None,
     'N_INIT'  :20,
     'ADAPTIVE': False,
     }
    },
    {'RF_GRID' : 
    {'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':False,
     'OUTPUT_TRANSFORMATION' : None,
     'N_INIT'  :20,
     'ADAPTIVE': False,
     }
    },
    {'RF_GRID_LOCAL' : 
    {'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':True,
     'OUTPUT_TRANSFORMATION' : None,
     'N_INIT'  :20,
     'ADAPTIVE': False,
     }
    },
    {'RF_GRID_LOCAL_TRANS' : 
    {'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':True,
     'OUTPUT_TRANSFORMATION' : True,
     'N_INIT'  :20,
     'ADAPTIVE': False,
     }
    },
    {'RF_GRID_LOCAL_TRANS_INIT' : 
    {'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':True,
     'OUTPUT_TRANSFORMATION' : True,
     'N_INIT'  :10,
     'ADAPTIVE': False,
     }
    },
    {'RF_GRID_LOCAL_TRANS_INIT_ADAPTIVE' : 
    {'SURROGATE' : 'RF',
     'ACQ_GRID': 900,
     'LOCAL_SEARCH':True,
     'OUTPUT_TRANSFORMATION' : True,
     'N_INIT'  :10,
     'ADAPTIVE': True,
     }
    }

]


FULL_CLASSIFICATION_AUTOML_LIST = [3, 11, 12, 14, 15, 16, 18, 22, 23, 
                28, 29, 31, 37, 43, 45, 49, 53, 2074, 
                2079, 3021, 3022, 3481, 3549, 3560, 3902, 
                3903, 3913, 3917, 3918, 9946, 9952, 9957, 
                9960, 9964, 9971, 9976, 9978, 9981, 9985, 
                10093, 10101, 14954, 14969, 125920, 125922, 
                146800, 146817, 146819, 146820, 146821, 146822, 
                146824, 167140, 167141, 146818, 168910, 168911, 168912]

HALF_CLASSIFICATION_AUTOML_LIST = [15, 23, 29, 43, 45, 2047, 
                                   2079, 3902, 3912, 3917, 9971 ,
                                   9952, 9957, 9976, 9978 ,14954, 
                                   146817, 14969, 167141, 10101, 146818, 3560]

N_SEEDS = [1,2,3,4,5]


N_INIT = 2
N_MAXIMUM = 30


ABLATION = 'ABLATION'
COMPARISON = 'COMPARISON'


EXPERIMENT = ABLATION