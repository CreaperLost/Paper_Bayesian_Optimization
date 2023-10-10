

"""
The names for each model - algorithm used.
"""
XGB_NAME = 'XGB'
RF_NAME = 'RF'
LINEAR_SVM_NAME = 'linearSVM'
RBF_SVM_NAME = 'rbfSVM'
DT_NAME = 'DT' 

"""
Upper, lower bound and transformation specification keys
"""
UPPER_BOUND = 'upper'
LOWER_BOUND = 'lower'


TRANSFORM_TYPE = 'transformation'
LOG_UNIFORM = 'loguniform'
UNIFORM = 'uniform'



HP_TYPE = 'HP_TYPE'

INTEGER_HP = 'INT_HP'
FLOAT_HP = 'FLOAT_HP'
CATEGORICAL_HP = 'CATEGORICAL_HP'

"""
Dictionary with the ranges and transformations for each hyper-parameter and algorithm

"""

hyper_parameters_dictionary = {
    RF_NAME : {
        "min_samples_leaf" : {
            LOWER_BOUND: 1 ,
            UPPER_BOUND:20,
            TRANSFORM_TYPE:None,
            HP_TYPE: INTEGER_HP, 
        },
        "min_samples_split" :{
            LOWER_BOUND: 1 ,
            UPPER_BOUND: 128,
            TRANSFORM_TYPE:None,
            HP_TYPE: INTEGER_HP,
        },
        "max_depth": {
            LOWER_BOUND: 1,
            UPPER_BOUND: 50,
            TRANSFORM_TYPE:None,
            HP_TYPE: INTEGER_HP,
        },
        "n_estimators": {
            LOWER_BOUND: 1,
            UPPER_BOUND: 1000,
            TRANSFORM_TYPE:None,
            HP_TYPE:INTEGER_HP,
        },
        "max_features":{
            LOWER_BOUND: 0,
            UPPER_BOUND: 1.0,
            TRANSFORM_TYPE:UNIFORM,
            HP_TYPE:FLOAT_HP,
        },
    },

    DT_NAME: {
        "dt_min_samples_leaf" : {
            LOWER_BOUND: 1 ,
            UPPER_BOUND:20,
            TRANSFORM_TYPE:None,
            HP_TYPE: INTEGER_HP,
        },
        "dt_min_samples_split" :{
            LOWER_BOUND: 1 ,
            UPPER_BOUND: 128,
            TRANSFORM_TYPE:None,
            HP_TYPE: INTEGER_HP,
        },
        "dt_max_depth": {
            LOWER_BOUND: 1,
            UPPER_BOUND: 50,
            TRANSFORM_TYPE:None,
            HP_TYPE: INTEGER_HP,
        },
        "dt_max_features":{
            LOWER_BOUND: 0,
            UPPER_BOUND: 1.0,
            TRANSFORM_TYPE:UNIFORM,
            HP_TYPE:FLOAT_HP,
        },
    
    },

    LINEAR_SVM_NAME: {
        "linear_C": {
            LOWER_BOUND : 2**-10,
            UPPER_BOUND : 2**10,
            TRANSFORM_TYPE:LOG_UNIFORM,
            HP_TYPE: FLOAT_HP,
        }, 
    },

    RBF_SVM_NAME: {
        "rbf_C": {
            LOWER_BOUND : 2**-10,
            UPPER_BOUND : 2**10,
            TRANSFORM_TYPE:LOG_UNIFORM,
            HP_TYPE: FLOAT_HP,
        }, 
        "rbf_gamma": {
            LOWER_BOUND : 2**-10,
            UPPER_BOUND : 2**10,
            TRANSFORM_TYPE:LOG_UNIFORM,
            HP_TYPE: FLOAT_HP,
        }, 
    },

    XGB_NAME: {
        "eta": {
            LOWER_BOUND: 2**-10,
            UPPER_BOUND: 1. ,
            TRANSFORM_TYPE:LOG_UNIFORM,
            HP_TYPE: FLOAT_HP,
        },
        "XGB_max_depth": {
            LOWER_BOUND : 1,
            UPPER_BOUND : 30,
            TRANSFORM_TYPE:None,
            HP_TYPE: INTEGER_HP,
        },
        "colsample_bytree" : {
            LOWER_BOUND : 0.1,
            UPPER_BOUND: 1. ,
            TRANSFORM_TYPE: UNIFORM,
            HP_TYPE: FLOAT_HP,
        },
        "reg_lambda": {
            LOWER_BOUND: 2**-10,
            UPPER_BOUND: 2**10,
            TRANSFORM_TYPE: LOG_UNIFORM,
            HP_TYPE: FLOAT_HP,
        },
        'subsample': {
            LOWER_BOUND: 0.1,
            UPPER_BOUND: 1.,
            TRANSFORM_TYPE: UNIFORM,
            HP_TYPE: FLOAT_HP,
        },
        "min_child_weight": {
            LOWER_BOUND: 1.,
            UPPER_BOUND: 2**7.,
            TRANSFORM_TYPE: LOG_UNIFORM,
            HP_TYPE: FLOAT_HP,
        },
        "colsample_bylevel": {
            LOWER_BOUND: 0.01 ,
            UPPER_BOUND: 1. ,
            TRANSFORM_TYPE: UNIFORM,
            HP_TYPE:FLOAT_HP,
        },
        "reg_alpha": {
            LOWER_BOUND: 2**-10 ,
            UPPER_BOUND: 2**10 ,
            TRANSFORM_TYPE: LOG_UNIFORM,
            HP_TYPE: FLOAT_HP,
        },
        "XGB_n_estimators": {
            LOWER_BOUND : 50,
            UPPER_BOUND: 500,
            TRANSFORM_TYPE: None,
            HP_TYPE: INTEGER_HP,
        },
        "max_delta_step":{
            LOWER_BOUND : 0,
            UPPER_BOUND : 10,
            TRANSFORM_TYPE: UNIFORM,
            HP_TYPE: FLOAT_HP,
        },
        "gamma":{
            LOWER_BOUND : 0,
            UPPER_BOUND : 5,
            TRANSFORM_TYPE: UNIFORM,
            HP_TYPE: FLOAT_HP,
        }

    },

}


model_list = [XGB_NAME,LINEAR_SVM_NAME,RF_NAME,DT_NAME,RBF_SVM_NAME]