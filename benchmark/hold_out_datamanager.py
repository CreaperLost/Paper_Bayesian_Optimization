import openml
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold
from oslo_concurrency import lockutils
from sklearn.preprocessing import LabelEncoder

from hpobench.util.data_manager import DataManager
from hpobench import config_file
from sklearn.model_selection import train_test_split
from specify_experiments import N_SEEDS

class Holdout_OpenMLDataManager(DataManager):

    def __init__(self, task_id: int,
                 data_path: Union[str, Path, None] = None,
                 seed: Union[int, None] = 1,
                 n_folds :int = 5):

        self.task_id = task_id
        assert seed != None
        self.seed = seed
        print(f'Data Manager Seed {seed}')

        # Using this split we can split all of the data.
        self.ncv_seed = 50

        self.train_X = []
        self.valid_X = []
        self.test_X = None
        self.train_y = []
        self.valid_y = []
        self.test_y = None
        self.train_idx = None
        self.test_idx = None
        self.task = None
        self.dataset = None
        self.preprocessor = None
        self.lower_bound_train_size = None
        self.n_classes = None
        self.n_folds = n_folds

        self.data_path = 'Holdout_Datasets/OpenML'
     
        #self.data_path = Path(data_path)
        openml.config.set_cache_directory(str(self.data_path))

        super(Holdout_OpenMLDataManager, self).__init__()

    # pylint: disable=arguments-differ
    @lockutils.synchronized('not_thread_process_safe', external=True, lock_path=f'{config_file.cache_dir}/openml_dm_lock', delay=0.2)
    def load(self, verbose=False):
        """Fetches data from OpenML and initializes the train-validation-test data splits

        The validation set is fixed till this function is called again or explicitly altered
        """
        # fetches task
        self.task = openml.tasks.get_task(self.task_id, download_data=False)
        self.n_classes = len(self.task.class_labels)

        # fetches dataset
        self.dataset = openml.datasets.get_dataset(self.task.dataset_id, download_data=False)
        if verbose:
            self.logger.debug(self.task)
            self.logger.debug(self.dataset)

        data_set_path = self.data_path + "/org/openml/www/datasets/" + f'Dataset{str(self.task.dataset_id)}_' 
        successfully_loaded = self.try_to_load_data(data_set_path)
        if successfully_loaded:
            self.logger.info(f'Successfully loaded the preprocessed splits from 'f'{data_set_path}')
            return

        # If the data is not available, download it.
        self.__download_data(data_set_path)

        successfully_loaded = self.try_to_load_data(data_set_path)

        if successfully_loaded:
            self.logger.info(f'Successfully loaded the preprocessed splits from 'f'{data_set_path}')

        return

    def try_to_load_data(self, data_path: str) -> bool:
        """
        Loads a dataset using a specified seed.
        The data are pre-splitted according to the n_seeds and the other stuff.
        returns boolean if can be loaded.
        """
        # Path for CV
        path_str = "Seed_{}_{}_{}_{}.parquet.gzip"

        #For test.
        path_str2 = "Seed_{}_{}_{}.parquet.gzip"
        try:
            for fold in range(self.n_folds) :
                self.train_X.append( pd.read_parquet(data_path + path_str.format(self.seed,"train", "x",str(fold))).to_numpy())
                self.train_y.append( pd.read_parquet(data_path + path_str.format(self.seed,"train", "y",str(fold))).squeeze(axis=1))
                self.valid_X.append( pd.read_parquet(data_path + path_str.format(self.seed,"valid", "x",str(fold))).to_numpy())
                self.valid_y.append( pd.read_parquet(data_path + path_str.format(self.seed,"valid", "y",str(fold))).squeeze(axis=1))
            
            
            self.test_X = pd.read_parquet(data_path + path_str2.format(self.seed,"test", "x")).to_numpy()
            self.test_y = pd.read_parquet(data_path + path_str2.format(self.seed,"test", "y")).squeeze(axis=1)
        except FileNotFoundError:
            print('File not found')
            return False
        return True

    def __download_data(self, data_path: str):
        self.logger.info('Start to download the OpenML dataset')

        # loads full data
        X, y, categorical_ind, feature_names = self.dataset.get_data(target=self.task.target_name,
                                                                     dataset_format="dataframe")
        #Label encode y.
        labelencoder = LabelEncoder()
        y = pd.Series(labelencoder.fit_transform(y))
        assert Path(self.dataset.data_file).exists(), f'The datafile {self.dataset.data_file} does not exists.'

        categorical_ind = np.array(categorical_ind)
        (cat_idx,) = np.where(categorical_ind)
        (cont_idx,) = np.where(~categorical_ind)


        # Stratified split of out_fold. Splits == ncv_seed.
        outer_fold = StratifiedKFold(n_splits=len(N_SEEDS),shuffle=True,random_state=self.ncv_seed)

        # Stratified split of inner fold. Splits == number of folds
        inner_fold = StratifiedKFold(n_splits=self.n_folds,shuffle=True,random_state=self.ncv_seed)

        X_train_per_seed, y_train_per_seed, X_valid_per_seed, y_valid_per_seed = [], [], [], []

        X_test_per_seed, y_test_per_seed = [], []

        # Hold-out split!
        for train_and_validation_idx, test_index in outer_fold.split(X,y):

            # tmp train split.
            X_train_and_validation,y_train_and_validation = X.iloc[train_and_validation_idx], y.iloc[train_and_validation_idx]
            
            # Test set.
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]

            X_test_per_seed.append(X_test)
            y_test_per_seed.append(y_test)

            # CV - Train, Valid splits.
            x_train_per_fold, y_train_per_fold, x_valid_per_fold, y_valid_per_fold = [],[],[],[]

            # 5- Fold inner CV.
            for train_idx, valid_idx in inner_fold.split(X_train_and_validation, y_train_and_validation):
                X_train = X_train_and_validation.iloc[train_idx]
                y_train = y_train_and_validation.iloc[train_idx]
                X_val  = X_train_and_validation.iloc[valid_idx]
                y_val  = y_train_and_validation.iloc[valid_idx]

                x_train_per_fold.append(X_train)
                y_train_per_fold.append(y_train)
                x_valid_per_fold.append(X_val)
                y_valid_per_fold.append(y_val)

            X_train_per_seed.append(x_train_per_fold)
            y_train_per_seed.append(y_train_per_fold)
            X_valid_per_seed.append(x_valid_per_fold)
            y_valid_per_seed.append(y_valid_per_fold)



        # preprocessor to handle missing values, categorical columns encodings,
        # and scaling numeric columns

        #
        self.preprocessor = make_pipeline(
            ColumnTransformer([
                (
                    "cat",
                    make_pipeline(SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(sparse=False, handle_unknown="ignore")
                    ),
                    cat_idx.tolist(),
                ),
                (
                    "cont",
                    make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler()),
                    cont_idx.tolist(),
                )
            ])
        )
        
        #Get back the training dataset, by combining the  training-validation.
        #Learn the preprocess and apply it to the test set. (FIT THE PROCEDURE TO WHOLE DATA)
        
        #Keep a training set before transformations
        for curr_seed in range(len(N_SEEDS)):
            
            # Start by transforming each of the X features of the test sets, using both train and validation data.
            train_and_validation_X  = np.vstack((X_train_per_seed[curr_seed][0],X_valid_per_seed[curr_seed][0]))
            self.preprocessor.fit(train_and_validation_X)
            
            print(f'Previous shape of X_test index: {curr_seed} , shape: {X_test_per_seed[curr_seed].shape}')
            X_test_per_seed[curr_seed] = self.preprocessor.transform(X_test_per_seed[curr_seed])
            print(f'New shape of X_test index: {curr_seed} , shape: {X_test_per_seed[curr_seed].shape}')

            y_test_per_seed[curr_seed] = self._convert_labels(y_test_per_seed[curr_seed])

            for curr_fold in range(self.n_folds):
                X_train_per_seed[curr_seed][curr_fold] = self.preprocessor.fit_transform(X_train_per_seed[curr_seed][curr_fold])
                print(f'X_train index: {curr_seed} , shape: {X_train_per_seed[curr_seed][curr_fold].shape}')
                X_valid_per_seed[curr_seed][curr_fold] = self.preprocessor.transform(X_valid_per_seed[curr_seed][curr_fold])    
                print(f'X_valid index: {curr_seed} , shape: {X_valid_per_seed[curr_seed][curr_fold].shape}')
                y_train_per_seed[curr_seed][curr_fold] = self._convert_labels(y_train_per_seed[curr_seed][curr_fold])
                y_valid_per_seed[curr_seed][curr_fold] = self._convert_labels(y_valid_per_seed[curr_seed][curr_fold])

        # Path for CV
        path_str = "Seed_{}_{}_{}_{}.parquet.gzip"

        #For test.
        path_str2 = "Seed_{}_{}_{}.parquet.gzip"
        

        label_name = str(self.task.target_name)

        #Store the needed information.
        #Each seed file
        #Each fold file.
        # Split as training, validation and test.
        for seed_idx,seed in enumerate(N_SEEDS):
            
            colnames = np.arange(X_test_per_seed[seed_idx].shape[1]).astype(str)
            pd.DataFrame(X_test_per_seed[seed_idx], columns=colnames).to_parquet(data_path + path_str2.format(str(seed),"test", "x"))
            y_test_per_seed[seed_idx].to_frame(label_name).to_parquet(data_path + path_str2.format(str(seed),"test", "y"))

            for curr_fold in range(self.n_folds):
                print(seed_idx,seed,curr_fold)
                colnames = np.arange(X_train_per_seed[seed_idx][curr_fold].shape[1]).astype(str)
                print(f'Length : {len(colnames), X_valid_per_seed[seed_idx][curr_fold].shape}')
                pd.DataFrame(X_train_per_seed[seed_idx][curr_fold], columns=colnames).to_parquet(data_path + path_str.format(str(seed),"train", "x",str(curr_fold)))
                y_train_per_seed[seed_idx][curr_fold].to_frame(label_name).to_parquet(data_path + path_str.format(str(seed),"train", "y",str(curr_fold)))
                pd.DataFrame(X_valid_per_seed[seed_idx][curr_fold], columns=colnames).to_parquet(data_path + path_str.format(str(seed),"valid", "x",str(curr_fold)))
                y_valid_per_seed[seed_idx][curr_fold].to_frame(label_name).to_parquet(data_path + path_str.format(str(seed),"valid", "y",str(curr_fold)))


    @staticmethod
    def _convert_labels(labels):
        """Converts boolean labels (if exists) to strings
        """
        label_types = list(map(lambda x: isinstance(x, bool), labels))
        if np.all(label_types):
            _labels = list(map(lambda x: str(x), labels))
            if isinstance(labels, pd.Series):
                labels = pd.Series(_labels, index=labels.index)
            elif isinstance(labels, np.array):
                labels = np.array(labels)
        return labels
