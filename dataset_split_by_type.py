import pandas as pd
from specify_experiments import ABLATION_DATASETS

data = pd.read_csv('AutoML+CC18.csv',index_col=0)
filtered_df = data[data['tid'].isin(ABLATION_DATASETS)]
filtered_df.to_csv('ablation_data.csv')