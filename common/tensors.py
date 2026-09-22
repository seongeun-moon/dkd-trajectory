"""Loading the preprocessed tensors and padding them into model input.

``preprocessing/05``, ``het_trans/`` and ``baselines/`` all read the same
``reduced24_*.pkl`` files and pad them the same way, so the loader lives here
rather than in each stage.

``data_path`` must already be resolved and end with a separator; callers get it
from their own ``resolve_data_path`` / ``paths`` helper.
"""

import joblib
import numpy as np
import pandas as pd


def obtain_category_token(category_data, token_info):
    category_token = {}
    for category in category_data.keys():
        importance = []
        for key in category_data[category]:
            try:
                importance.append(token_info['lab'][key])
            except KeyError:
                importance.append(0.)
        category_token[category] = importance
    return category_token


def obtain_prs_token(prs_data, token_info):
    prs_token = []
    for data in prs_data:
        importance = []
        for ingrs in data[:, 2]: # basis to determine importance is changed from code to ingredients due to there are duplicated codes for different ingredients while the original importance is based on ingredients
            ingr_importance = []

            num_ingrs = len(ingrs)
            if pd.isnull(ingrs).mean == 1:
                ingr_importance = [1.]*num_ingrs
            else:
                for ingr in ingrs:
                    try:
                        ingr_importance.append(token_info['drug'][ingr])
                    except KeyError:
                        ingr_importance.append(1.)
            importance.append(ingr_importance)
        prs_token.append(importance)
    return prs_token


def pad_category_data(org_data, obsn_mask, max_lengths, eps=-1e-5):
    padded_inputs = {}
    padded_times = {}
    padded_obsn_masks = {}
    for key in org_data.keys():
        max_length = max_lengths[key]
        num_data = len(org_data[key])
        num_feature = org_data[key][0].shape[-1]
        
        padded_category = np.zeros((num_data, max_length, num_feature)) + eps
        padded_time = np.zeros((num_data, max_length))
        padded_obsn_mask = np.zeros_like(padded_category)

        for i in range(num_data):
            current_seq_len, feature_dim = org_data[key][i].shape
            padded_category[i, :current_seq_len] = np.concatenate((np.ones(current_seq_len).reshape(-1, 1), org_data[key][i][:, 1:]), axis=1)
            padded_time[i, :current_seq_len] = org_data[key][i][:, 0]
            padded_obsn_mask[i, :current_seq_len] = obsn_mask[key][i]
            
        padded_inputs[key] = padded_category
        padded_times[key] = padded_time
        padded_obsn_masks[key] = padded_obsn_mask

    return {'inputs': padded_inputs,
            'time': padded_times, 
            'obsn_mask': padded_obsn_masks}


def pad_prs_data(org_data, length_info, tokens):
    num_data = len(org_data)
    padded_codes = np.zeros((num_data, length_info['prs'], length_info['prs_num']+1))
    padded_dosage = np.zeros_like(padded_codes)
    padded_units = np.zeros_like(padded_codes)
    padded_time = np.zeros((num_data, length_info['prs']))
    padded_tokens = np.ones((num_data, length_info['prs'], length_info['prs_num']))

    tokens = np.array(tokens, dtype='object')
    for data_i, data in enumerate(org_data):
        current_seq_len, feature_dim = data.shape
        for seq in range(current_seq_len):
            if data[seq, 2] == 'nan':
                data[seq, 1] = [0]
                data[seq, 3] = [0]
                data[seq, 4] = [0]
                tokens[data_i][seq] = [1]
            current_num = len(data[seq, 1])

            try:
                padded_codes[data_i, seq, :current_num+1] = [1] + data[seq, 1] # CLS
                padded_dosage[data_i, seq, :current_num+1] = [1] + [0 if x=='nan' else x for x in data[seq, 3]]
                padded_units[data_i, seq, :current_num+1] = [1] + data[seq, 4]
                padded_time[data_i, seq] = data[seq, 0]
            except TypeError:
                raise
            try:
                padded_tokens[data_i, seq, :current_num] = tokens[data_i][seq]
            except RuntimeError:
                raise
    
    padded_data = {
        'code': padded_codes,
        'dosage': padded_dosage,
        'unit': padded_units,
        'time': padded_time,
        'token': padded_tokens
        }
    return padded_data


def load_data(
    data_path=None,
    split_type='frequency',
    data_type='train'):
    """ To load preprocessed data.
    """
    category_data = joblib.load(data_path+'reduced24_'+split_type+'_' + data_type+'.pkl')
    obsn_mask = joblib.load(data_path+'reduced24_obsn_mask_'+split_type+'.pkl')
    length_info = joblib.load(data_path+'reduced24_length_info_'+split_type+'.pkl')
    key_info = joblib.load(data_path+'key_info.pkl')

    padded_inputs = pad_category_data(category_data['inputs'], obsn_mask[data_type], length_info)

    padded_inputs['token'] = obtain_category_token({'usual': key_info['usual_categories'], 'unusual': key_info['unusual_categories']}, key_info['importance_token'])

    category_data['inputs'] = padded_inputs
    
    targets = category_data['targets']
    times = category_data['times']
    category_data = category_data['inputs']

    prs_data = joblib.load(data_path+'reduced24_prescription_'+data_type+'.pkl')['inputs']
    prs_token = obtain_prs_token(prs_data, key_info['importance_token'])
    prs_data = pad_prs_data(prs_data, length_info, prs_token)

    return {
        'category_data': category_data, 
        'prs_data': prs_data, 
        'targets': targets, 
        'conditions': times, 
        'length_info': length_info
    }
