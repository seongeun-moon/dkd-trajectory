"""Tokenisation helpers for the step-04 pipeline.

Extracted from ``04_tokenization_unified.py`` so that the tokenisation stage is
declared once, separately from the script that drives it.
"""

import ast
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

def obtain_obsn_mask(inputs):
    categories = list(inputs.keys())
    obsn_masks = {}
    for category in categories:
        print(category)
        obsn_mask = []
        for data in inputs[category]:
            obsn_mask.append(~pd.isnull(data))
        obsn_masks[category] = np.array(obsn_mask)
    return obsn_masks


def scale_data(inputs, obsn_mask, target_idx=None, scalers=None):
    categories = list(inputs.keys())
    if scalers is None:
        scalers = {}
        scale_flag = True
    else:
        scale_flag = False

    scaled_inputs = {}
    for category in categories:
        print(category)
        scaled_input = []
        
        if target_idx is None:
            target = np.arange(inputs[category][0].shape[-1])
        else:
            target = target_idx[category]

        try:
            integ_data = np.concatenate(tuple(inputs[category]), axis=0)
            integ_data = integ_data[:, target]
            integ_data = integ_data.astype(float)
        except ValueError:
            raise
        
        if scale_flag:
            f_min = np.nanmin(integ_data, axis=0)
            f_max = np.nanmax(integ_data, axis=0)
            f_avg = np.nanmean(integ_data, axis=0)
            scalers[category] = [f_min, f_avg, f_max]
        else:
            f_min, f_avg, f_max = scalers[category]
        
        for data, obsn_idx in zip(inputs[category], obsn_mask[category]):
            scaled_data = data.copy()
            
            for i, f in enumerate(target):
                scaled_data[:, f][obsn_idx[:,f]] = (data[:, f][obsn_idx[:,f]] - f_min[i]) / (f_max[i] - f_min[i])
                scaled_data[:, f][~obsn_idx[:,f]] = (f_avg[i] - f_min[i]) / (f_max[i] - f_min[i])
            scaled_input.append(scaled_data)
        scaled_inputs[category] = np.array(scaled_input)
    return scaled_inputs, scalers


def convert_sex(data, key_info):
    for i, dd in tqdm(enumerate(data), total=len(data)):
        target_idx = (list(key_info['meta_keys'])+list(key_info['usual_categories'])).index('sex')
        if len(np.where(dd[:, target_idx] == 'M')[0]) > 0:
            data[i][:, target_idx] = np.where(dd[:, target_idx] == 'M', 1, dd[:, target_idx])
        if len(np.where(dd[:, target_idx] == 'F')[0]) > 0:
            data[i][:, target_idx] = np.where(dd[:, target_idx] == 'F', 0, dd[:, target_idx])
    return data


def literal_eval_lists(data, target_column=(-4, -3, -2, -1)):
    new_data = []
    for i in tqdm(range(len(data))):
        d = data[i]
        new_d = d.copy()
        for i, seq in enumerate(d):
            seq[pd.isnull(seq)] = "['nan']"
            for column in target_column:
                try:
                    new_d[i, column] = ast.literal_eval(seq[column].replace('[nan', "['nan'").replace(', nan', ", 'nan'"))
                except ValueError:#AttributeError:
                    raise
        new_data.append(new_d)
    return new_data


def embed(data, embed_indices, save_indices=None, data_dicts=None, special_token=('<UNK>',), update_dict=False):
    embedded_data = copy.deepcopy(data)
    
    def embed_data(element, embed_dict, update_dict, dict_count):
        if type(element) == list:
            embedded_element = []
            for elem in element:
                embedded_elem, embed_dict, dict_count = embed_data(elem, embed_dict, update_dict, dict_count)
                embedded_element.append(embedded_elem)
        else:
            if not element in embed_dict:
                if update_dict:
                    embed_dict[element] = dict_count
                    dict_count += 1
                    embedded_element = embed_dict[element]
                else:
                    embedded_element = embed_dict['<UNK>']
            else:
                embedded_element = embed_dict[element]
        return embedded_element, embed_dict, dict_count
        
    if type(embed_indices) == int:
        embed_indices = [embed_indices]
    if save_indices is None:
        save_indices = embed_indices
    if type(save_indices) == int:
        save_indices = [save_indices]
    
    dict_is = [len(special_token) for _ in range(len(embed_indices))]
    if data_dicts is None:
        special_token_dict = {k: v for v, k in enumerate(special_token)}
        data_dicts = [copy.deepcopy(special_token_dict) for _ in range(len(embed_indices))]
        update_dict = True
    
    for di, dd in tqdm(enumerate(data), total=len(data)):
        for i, (embed_idx, save_idx, data_dict, dict_i) in enumerate(zip(
            embed_indices, save_indices, data_dicts, dict_is
        )):
            target_list = dd[:, embed_idx]
            for ti, tts in enumerate(target_list):
                new_targets, data_dict, dict_i = embed_data(tts, data_dict, update_dict, dict_i)
                embedded_data[di][ti, save_idx] = new_targets
            data_dicts[i] = data_dict
            dict_is[i] = dict_i
            
    return embedded_data, data_dicts


def reduce_prs(inputs, max_len=24):
    reduced_inputs = []
    selected_idxs = []
    for data in tqdm(inputs):
        if data.shape[0] <= max_len:
            reduced_inputs.append(data)
            selected_idxs.append([])
            continue
        
        num_drug = np.array([len(data[i, 2]) for i in range(data.shape[0])])
        p = num_drug / num_drug.sum()
        idx = sorted(np.random.choice(np.arange(data.shape[0]), size=max_len, p=p, replace=False))
        
        reduced_inputs.append(data[idx])
        selected_idxs.append(idx)
    return np.array(reduced_inputs, dtype='object'), np.array(selected_idxs, dtype='object')


def reject_data(category_data, prs, times):
    num_data = times.shape[0]
    reject_idx = np.zeros(num_data, dtype=bool)
    
    for i in tqdm(range(num_data)):
        current_max_seq_len = 0
        for key in category_data.keys():
            if category_data[key][i].shape[0] > current_max_seq_len:
                current_max_seq_len = category_data[key][i].shape[0]
        if prs[i].shape[0] > current_max_seq_len:
            current_max_seq_len =  prs[i].shape[0]
        
        if times[i] > current_max_seq_len:
            reject_idx[i] = True
    return reject_idx


def reduce_seq(inputs, obsn, len_info, egfr_idx, max_len=24):    
    reduced_inputs = {}
    reduced_obsn = {}
    selected_idxs = {}
    for key in inputs.keys():
        new_data = []
        new_obsn = []
        idxs = []
        for data, mask in tqdm(zip(inputs[key], obsn[key]), total=len(inputs[key])):
            _, unique_idx = np.unique(data.astype(float), return_index=True, axis=0)
            unique_idx = sorted(unique_idx) 
            unique_data = data[unique_idx]
            unique_mask = mask[unique_idx]
            
            if unique_data.shape[0] <= max_len:
                new_data.append(unique_data)
                new_obsn.append(unique_mask)
                idxs.append([unique_idx, []])
                continue
                
            num_obsn = unique_mask.sum(-1)
            prob = num_obsn / num_obsn.sum()
            
            if key == 'usual':
                prob[unique_mask[:, egfr_idx] == True] = 1
                prob = prob / prob.sum()
            
            idx = sorted(np.random.choice(np.arange(unique_data.shape[0]), size=max_len, p=prob, replace=False))
            
            new_data.append(unique_data[idx])
            new_obsn.append(unique_mask[idx])
            idxs.append([unique_idx, idx])
        
        reduced_inputs[key] = np.array(new_data, dtype='object')
        reduced_obsn[key] = np.array(new_obsn, dtype='object')
        selected_idxs[key] = np.array(idxs, dtype='object')
    return reduced_inputs, reduced_obsn, selected_idxs
