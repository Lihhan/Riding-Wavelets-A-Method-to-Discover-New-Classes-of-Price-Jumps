import numpy as np
import pandas as pd
from scipy.signal import morlet, convolve
from numpy.lib.stride_tricks import sliding_window_view
from globalVariables import *

def calculate_sigma(group, K=K):
    sigma_squared_t = np.sqrt((np.pi / (2 * K)) * (
        np.abs(group['mid_returns_1min']) * np.abs(group['mid_returns_1min']).shift(1)
    ).rolling(window=K).sum())
    return sigma_squared_t

def calculate_W_i(group, x):
    group['condition'] = (-group['r_hat']**2 + x > 0).astype(int)
    group['r_hat_square'] = np.where(group['condition'] == 1, 1.081 * group['r_hat'] ** 2, 0)
    numer = group['r_hat_square'].rolling(window=K).sum()
    denom = group['condition'].rolling(window=K).sum()
    return np.sqrt(numer / denom)

def calc_jump_score(group):
    group['condition1'] = (-group['r_hat']**2 + 4 ** 2 > 0).astype(int)
    group['r_hat_square1'] = np.where(group['condition1'] == 1, 1.081 * group['r_hat'] ** 2, 0)
    group['W_i_1'] = np.sqrt(group['r_hat_square1'].rolling(window=K).sum() / group['condition1'].rolling(window=K).sum())
    group['f_i_1'] = group['W_i_1'] / np.sqrt(1/K * ((group['W_i_1'] ** 2).rolling(window=K).sum()))
    
    group['condition2'] = (-group['r_hat']**2 + 6.635 > 0).astype(int)
    group['r_hat_square2'] = np.where(group['condition2'] == 1, 1.081 * group['r_hat'] ** 2, 0)
    group['W_i_2'] = np.sqrt(group['r_hat_square2'].rolling(window=K).sum() / group['condition2'].rolling(window=K).sum())
    group['f_i_2'] = group['W_i_2'] / np.sqrt(1/K * ((group['W_i_2'] ** 2).rolling(window=K).sum()))
    
    group['jump_score'] = group['mid_returns_1min'] / (group['sigma_t'] * group['f_i_1'] * group['f_i_2'])
    return group

def compute_convolutions(x, j_values, T, f=5, sigma=0.1):
    def base_wavelet(t, f=0.0, sigma=1.0):
        omega_0 = 2 * np.pi * f
        real_part = morlet(len(t), w=omega_0)
        return real_part * np.exp(1j * omega_0 * t)

    def scaled_wavelet(t, j, f=1.0, sigma=1.0):
        return base_wavelet(2**(-j) * t, f=f, sigma=sigma)
    
    def convolution(x, wavelet, T):
        result = np.zeros(T)  # Initialize the result array
        for t in range(T):
            for i in range(T):
                if 0 <= i - t <= T:
                    result[t] += x[i] * wavelet[i-t]
        return result

    convolved_results = []
    wavelet_length = T
    t = np.linspace(-0.5, 0.5, wavelet_length)
    
    for j in j_values:
        wavelet_j = scaled_wavelet(t, j, f=f, sigma=sigma)
        convolved_signal_real = convolve(x, np.real(wavelet_j), mode='valid')
        convolved_signal_imag = convolve(x, np.imag(wavelet_j), mode='valid')
        convolved_results.append(float(convolved_signal_real))
        convolved_results.append(float(convolved_signal_imag))
    
    combinations = [(j1, j2) for j1 in j_values for j2 in j_values if j1 < j2]
    
    for j1, j2 in combinations:
        wavelet_j1 = scaled_wavelet(t, j1, f=f, sigma=sigma)
        wavelet_j2 = scaled_wavelet(t, j2, f=f, sigma=sigma)
        
        convolved_signal_j1 = convolution(x, wavelet_j1, T)
        abs_convolved_signal_j1 = np.abs(convolved_signal_j1)
        
        final_convolved_signal_real = convolve(abs_convolved_signal_j1, np.real(wavelet_j2), mode='valid')
        final_convolved_signal_imag = convolve(abs_convolved_signal_j1, np.imag(wavelet_j2), mode='valid')
        
        convolved_results.append(float(np.real(final_convolved_signal_real)))
        convolved_results.append(float(np.imag(final_convolved_signal_imag)))
    
    convolved_results = np.array(convolved_results)
    normalized_results = (convolved_results - np.mean(convolved_results)) / np.std(convolved_results)
    
    return normalized_results

def process_group(group, T):
    half_T = 59
    jump_score_dict = {}
    valid_indices = group.index[group['hf_is_rise'].isin([1, -1])].to_numpy()
    
    start_idx = np.clip(valid_indices - half_T, group.index[0], group.index[-1])
    end_idx = np.clip(valid_indices + half_T, group.index[0], group.index[-1])
    for idx, start, end in zip(valid_indices, start_idx, end_idx):
        sig = group.loc[idx, 'hf_is_rise']
        relevant_rows = group.loc[start:end]
        if len(relevant_rows['jump_score'].values) == T:
            x1 = relevant_rows['jump_score'].values * sig
            x2 = relevant_rows['mid_returns_1min'].values
            x3 = relevant_rows['mid_price'].values
            jump_score_dict[group.loc[idx, 'key']] = [x1, x2, x3]
    return jump_score_dict

def extract_and_convolve(df, j_values, T):
    grouped_result = {}
    for name, group in df.groupby('securityid_str'):
        print(name)
        grouped_result[name] = process_group(group, T)
    jump_score_dict = {k: v for dict in grouped_result.values() for k, v in dict.items()}
    convolved_results_all = {}
    for key, x in jump_score_dict.items():
        print(key)
        convolved_results_all[key] = compute_convolutions(x[0], j_values, T)
    return convolved_results_all, jump_score_dict

def linear_kernel(X):
    return np.dot(X, X.T)

def center_kernel(K):
    n = K.shape[0]
    ones_n = np.ones((n, n)) / n
    K_centered = K - np.dot(ones_n, K) - np.dot(K, ones_n) + np.dot(ones_n, np.dot(K, ones_n))
    return K_centered

def get_eigens(K_centered):
    eigvals, eigvecs = np.linalg.eigh(K_centered)
    # 按照特征值从大到小排序
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

def select_components(eigvals, eigvecs, num_components):
    selected_eigvecs = eigvecs[:, :num_components]
    selected_eigvals = eigvals[:num_components]
    return selected_eigvals, selected_eigvecs

def project_data(K_centered, eigvecs):
    return np.dot(K_centered, eigvecs)

def calculate_weights(X, v, lambda_reg=1e-5):
    XtX = np.dot(X.T, X)
    XtX_reg = XtX + lambda_reg * np.eye(XtX.shape[0])
    w = np.dot(np.linalg.inv(XtX_reg), np.dot(X.T, v))
    return w

def kernel_pca(X, num_components):
    K = linear_kernel(X)
    K_centered = center_kernel(K)
    eigvals, eigvecs = get_eigens(K_centered)
    _, selected_eigvecs = select_components(eigvals, eigvecs, num_components)
    X_pc = project_data(K_centered, selected_eigvecs)
    return selected_eigvecs, X_pc

def compute_d1_scores(rolling_windows, num_components=num_components):
    d1_scores = []
    for window in rolling_windows:
        eigvecs, _ = kernel_pca(window, num_components)
        weights = calculate_weights(window, eigvecs[:, 0])
        D1_score = np.nansum((weights * window)[:, -15:], axis=1)
        d1_scores.append(D1_score.mean())
    return np.array(d1_scores)

def apply_kernel_pca_to_rolling(df, num_components=num_components, window=D1_window):
    rolling_windows = sliding_window_view(df.values, window_shape=(window, df.shape[1]))
    rolling_windows = rolling_windows.squeeze()
    d1_scores = compute_d1_scores(rolling_windows, num_components=num_components)
    new_df = pd.DataFrame(d1_scores, columns=['D1_score'], index=df.index[window-1:])
    return new_df

def get_middle_percent_indices(df, col, lower_ratio=lower_ratio, upper_ratio=upper_ratio):
    lower_bound_index = int(len(df) * lower_ratio)
    upper_bound_index = int(len(df) * upper_ratio)
    sorted_df = df.sort_values(by=col)
    middle_indices = sorted_df.iloc[lower_bound_index:upper_bound_index].index
    return middle_indices
