import numpy as np
import pandas as pd
from ParallelCalFactor import ParallelCalFactor_pro
Pro = ParallelCalFactor_pro()
from HighFreqFileSystem import HDFData
np.random.seed(22)
import bottleneck as bn
from calc_function import *
import matplotlib.pyplot as plt
from globalVariables import *
M = HDFData('1min')

df = M.get_data(start_time=start_time, end_time=end_time,
        factors=['min_1min/high', 'min_1min/low'],
        pandas=True, copy=False,form='model')

df['mid_price'] = (df['min_1min/high'] + df['min_1min/low']) / 2
df['mid_price'] = df.groupby('securityid')['mid_price'].fillna(method='ffill')
df.drop(['min_1min/high', 'min_1min/low'], axis=1, inplace=True)
df['mid_returns_1min'] = df.groupby(['securityid'])['mid_price'].transform(lambda x: np.log(x / x.shift(1)))
df['sigma_t'] = df.groupby('securityid').apply(calculate_sigma).reset_index(level=0, drop=True)
df['r_hat'] = df['mid_returns_1min'] / df['sigma_t']
df = df.groupby('securityid').apply(calc_jump_score).reset_index(level=0, drop=True)
print(df.isna().sum())
df.drop(['condition1', 'mid_price', 'r_hat_square1', 'condition2', 'r_hat_square2', 'mid_returns_1min', 'sigma_t', 'r_hat', 'W_i_1', 'f_i_1', 'W_i_2', 'f_i_2'], axis=1, inplace=True)
df['jump_score'] = df['jump_score'].apply(lambda x: 0 if (x < -100) or (x > 100) or np.isnan(x) else x)
df['hf_is_rise'] = df['jump_score'].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))

convolved_results = extract_and_convolve(df, j_values)[0]
jump_score_dict = extract_and_convolve(df, j_values)[1]
num = (T-1)/2
jump_score_hat_dict = {key: [-x for x in value[0]] if value[0][num] < 0 else value[0] for key, value in jump_score_dict.items()}

columns = [f'REf{i}(t)' for i in range(6)]+[f'IMf{i}(t)' for i in range(6)]+[f'REf{i}(t) * {j}(t)' for i in range(6) for j in range(6) if i < j] + [f'IMf{i}(t) * {j}(t)' for i in range(6) for j in range(6) if i < j]
col_selected = [f'IMf{i}(t) * {j}(t)' for i in range(6) for j in range(6) if i < j]
w_df = pd.DataFrame(np.array(list(convolved_results.values())).T).T
w_df.columns = columns
w_df.index = list(convolved_results.keys())

new_df = apply_kernel_pca_to_rolling(w_df, num_components, D1_window)
new_df['D1_score_abs'] = new_df['D1_score'].abs()
new_df['D2_score'] = w_df['IMf0(t)']
new_df['D3_score'] = w_df['REf0(t)']
new_df['securityid'] = new_df.index.str.split(' @ ').str[0].astype(int)
new_df['time'] = pd.to_datetime(new_df.index.str.split(' @ ').str[1])

merged_df = pd.merge(new_df, df[['securityid', 'time', 'hf_is_rise']], left_on=['securityid', 'time'], right_on=['securityid', 'time'], how='inner')
merged_df.index = new_df.index
filtered_df = merged_df[merged_df['hf_is_rise'] == 1]

D1_abs_largest = new_df.nlargest(int(len(new_df) * topRatio), 'D1_score_abs')
D1_largest = new_df.nlargest(int(len(new_df) * topRatio), 'D1_score')
D2_largest = new_df.nlargest(int(len(new_df) * topRatio), 'D2_score')
D3_largest = new_df.nlargest(int(len(new_df) * topRatio), 'D3_score')

D1_abs_smallest = new_df.nsmallest(int(len(new_df) * topRatio), 'D1_score_abs')
D1_smallest = new_df.nsmallest(int(len(new_df) * topRatio), 'D1_score')
D2_smallest = new_df.nsmallest(int(len(new_df) * topRatio), 'D2_score')
D3_smallest = new_df.nsmallest(int(len(new_df) * topRatio), 'D3_score')
