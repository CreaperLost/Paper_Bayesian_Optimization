 # Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


#import sys
#from typing import Optional

import numpy  as np
#import pandas as pd
#import torch
from copy import deepcopy
#from torch.quasirandom import SobolEngine
from sklearn.preprocessing import power_transform
#import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from bo_algorithms.my_bo.surrogate.base_surrogate_model import BaseModel

class  Ensemble_RF2(BaseModel):
    def __init__(self, config_space, rng=None,n_estimators= 100,box_cox_enabled = None):
        
        self.config_space = config_space

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        print(f'My ensemble2 local random forest is deterministic by using seed : {rng,self.rng}')

        self.n_estimators =  n_estimators
        self.standardize_output = box_cox_enabled
        self.models = []
        super(Ensemble_RF2, self).__init__()


    def train(self, X, y_scores):
        assert X.ndim == 2
        
        self.models = []
        for fold in range(y_scores.shape[1]):
            m = RandomForestRegressor(n_estimators = 100,random_state=self.rng,n_jobs=-1)
            m.fit(X,y_scores[:,fold])
            self.models.append(m)

        

    def predict(self, X:np.ndarray):
        
        mean_list = []
        var_list = []


        for m in self.models:
            
            preds = []
            for estimator in m:
                preds.append(estimator.predict(X).reshape([-1,1]))


            mean = m.predict(X).reshape(-1, 1)
            var = np.var(np.concatenate(preds, axis=1), axis=1)

            mean_list.append(mean.reshape([-1,1]))
            
            var_list.append(var.reshape([-1,1]) )

        # variance is mean(variances)/n_folds
        return np.mean(mean_list,axis = 0).reshape([-1,1]), np.mean(var_list, axis =0 ).reshape([-1,1])/ len(var_list)
        
