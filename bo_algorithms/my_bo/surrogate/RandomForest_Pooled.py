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

class  Pooled_RF(BaseModel):
    def __init__(self, config_space, rng=None,n_estimators= 100,box_cox_enabled = None):
        
        self.config_space = config_space

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        print(f'My local random forest is deterministic by using seed : {rng,self.rng}')

        self.n_estimators =  n_estimators
        self.rf = RandomForestRegressor(n_estimators = 100,random_state=rng,n_jobs=-1)
        self.standardize_output = box_cox_enabled
        super(Pooled_RF, self).__init__()


    def train(self, X, y_scores):
        assert X.ndim == 2


        x_pooled = []
        y = []

        for fold in range(y_scores.shape[1]):
            x_pooled.extend(X)
            y.extend(y_scores[:,fold])

        
        #Not needed for non conditional space.
        #X = self._impute_inactive(X)
        if self.standardize_output == True:
            tmp_y = y.copy()

            try:
                if y.min() <= 0:
                    y = power_transform(y / y.std(), method = 'yeo-johnson',standardize=False)
                else:
                    y = power_transform(y / y.std(), method = 'box-cox',standardize=False)
                    if y.std() < 0.5:
                        y = power_transform(y / y.std(), method = 'yeo-johnson',standardize=False)
                if y.std() < 0.5:
                    print('It failed...')
                    raise RuntimeError('Power transformation failed')
            except:    
                #Reset y.
                y = tmp_y

            
        

            y = self._normalize_y(y)
        
        x_pooled = np.array(x_pooled)
        y = np.array(y)

        y = y.flatten()
        
        self.rf.fit(x_pooled,y)



    def predict(self, X: np.ndarray):
        """Predict means and variances for given X.
        Parameters
        ----------
        X : np.ndarray of shape = [n_samples,
                                   n_features (config + instance features)]
        cov_return_type: Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.
        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))
    
        mean = self.rf.predict(X).reshape(-1, 1)

        preds = []
        for estimator in self.rf.estimators_:
            preds.append(estimator.predict(X).reshape([-1,1]))
        
        var = np.var(np.concatenate(preds, axis=1), axis=1)


        if self.standardize_output == True:
            mean, var = self._untransform_y(mean, var)
        
        return mean.reshape([-1,1]), var.reshape([-1,1]) 

