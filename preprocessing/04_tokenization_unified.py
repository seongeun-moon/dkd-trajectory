import os

import joblib
import numpy as np
from numpy.random import shuffle
from tqdm import tqdm

from config import DATA_TYPE
from paths import RAW_ROOT
from tokenization import (convert_sex, embed, literal_eval_lists,
                          obtain_obsn_mask, reduce_prs, reduce_seq,
                          reject_data, scale_data)

def obtain_reserve_idx(inputs):
    num_data = len(inputs[list(inputs.keys())[0]])
    idx = np.ones(num_data, dtype=bool)
    for i in range(num_data):
        flag = True
        for k in inputs.keys():
            flag = flag & (len(inputs[k][i]) <= 1)

        if flag:
            idx[i] = False                
    return idx

def reject_data_by_length(data, prs):
    reserve_idx = obtain_reserve_idx(data['inputs'])
    for k in data['inputs'].keys():
        data['inputs'][k] = [d for d, cdt in zip(data['inputs'][k], reserve_idx) if cdt]
    data['targets'] = data['targets'][reserve_idx]
    data['times'] = data['times'][reserve_idx]
    
    prs['inputs'] = [d for d, cdt in zip(prs['inputs'], reserve_idx) if cdt]
    prs['targets'] = prs['targets'][reserve_idx]
    prs['times'] = prs['times'][reserve_idx]
    return data

def concat_data(data1, data2):
    data = data1.copy()
    if type(data1['inputs']) == dict:
        for k in data['inputs'].keys():
            data['inputs'][k] = data1['inputs'][k] + data2['inputs'][k]
    elif type(data1['inputs']) == np.ndarray:
        data['inputs'] = np.concatenate((data1['inputs'], data2['inputs']), axis=0)
    elif type(data1['inputs']) == list:
        data['inputs'] = data1['inputs'] + data2['inputs']
    else:
        raise ValueError
        
    data['targets'] = np.concatenate((data1['targets'], data2['targets']), axis=0)
    data['times'] = np.concatenate((data1['times'], data2['times']), axis=0)
    return data


def main():
    SPLITS = ('train', 'valid', 'test')
    # ## Integrate data
    HELDOUT = SPLITS[1:]
    data_path = RAW_ROOT
    hospital = 'snubh'
    data_type = DATA_TYPE
    df_segmented_paths = [os.path.join(data_path, hospital, 'DM_df_segmented', data_type)]
    split_type = 'frequency'
    segmented_df_path = os.path.join(data_path, hospital, 'DM_df_segmented', data_type)
    os.makedirs(segmented_df_path, exist_ok=True)
    final_path = os.path.join(data_path, hospital, 'DM_df_final_pos_encode', data_type)
    os.makedirs(final_path, exist_ok=True)
    key_info = joblib.load(os.path.join(segmented_df_path, 'key_info.pkl'))
    df_segmented_path = df_segmented_paths[0]
    data = joblib.load(os.path.join(df_segmented_path, f'segmented_{split_type}.pkl'))
    prs = joblib.load(os.path.join(df_segmented_path, 'segmented_prescription.pkl'))
    # reject data with length of 1
    data = reject_data_by_length(data, prs)
    for df_segmented_path in df_segmented_paths[1:]:
        tmp_data = joblib.load(os.path.join(df_segmented_path, f'segmented_{split_type}.pkl'))
        tmp_prs = joblib.load(os.path.join(df_segmented_path, 'segmented_prescription.pkl'))

        # reject data with length of 1
        tmp_data = reject_data_by_length(tmp_data, tmp_prs)

        # concat data
        data = concat_data(data, tmp_data)    
        prs = concat_data(prs, tmp_prs)
    joblib.dump(data, os.path.join(segmented_df_path, f'segmented_{split_type}.pkl'))
    joblib.dump(prs, os.path.join(segmented_df_path, 'segmented_prescription.pkl'))
    # divide into training, validation, and test set
    patients = list(set([i[0][0] for i in data['inputs']['usual']]))
    # update key info with patient IDs
    # ## Observation mask
    # ## Processing
    # ## Scaling
    # convert categorical data into scalars
    patients = np.array(patients)
    num_patients = len(patients)
    idx = np.arange(num_patients)
    shuffle(idx)
    split_patients = {}
    split_patients['train'] = patients[idx[ : int(num_patients*0.7)]]
    split_patients['valid'] = patients[idx[int(num_patients*0.7) : int(num_patients*0.85)]]
    split_patients['test'] = patients[idx[int(num_patients*0.85) : ]]
    joblib.dump(split_patients, os.path.join(segmented_df_path, 'patients_div_info.pkl'))
    inputs = {split: {k: [] for k in data['inputs'].keys()} for split in SPLITS}
    targets = {split: [] for split in SPLITS}
    times = {split: [] for split in SPLITS}
    input_keys = list(data['inputs'].keys())
    for i, d in tqdm(enumerate(data['inputs'][input_keys[0]]), total=len(data['targets'])):
        subject_id = str(d[0, 0])
        for split in SPLITS:
            if subject_id in split_patients[split]:
                for k in input_keys:
                    inputs[split][k].append(data['inputs'][k][i])
                targets[split].append(data['targets'][i])
                times[split].append(data['times'][i])
                break
        else:
            raise ValueError
    del data
    for split in SPLITS:
        joblib.dump({
            'inputs': inputs[split],
            'targets': np.array(targets[split]),
            'times': np.array(times[split]),
            }, os.path.join(segmented_df_path, f'segmented_{split_type}_{split}.pkl'))
    split_prs = {}
    for split in SPLITS:
        split_prs[split] = {'inputs': [], 'targets': [], 'times': []}
    for i, d in tqdm(enumerate(prs['inputs']), total=len(prs['targets'])):
        subject_id = str(d[0, 0])
        for split in SPLITS:
            if subject_id in split_patients[split]:
                for field in ('inputs', 'targets', 'times'):
                    split_prs[split][field].append(prs[field][i])
                break
        else:
            raise ValueError
    del prs
    for split in SPLITS:
        joblib.dump(split_prs[split], os.path.join(segmented_df_path, f'segmented_prescription_{split}.pkl'))
    for split in SPLITS:
        key_info[f'{split}_patient'] = split_patients[split]
    joblib.dump(key_info, os.path.join(segmented_df_path, 'key_info.pkl'))
    joblib.dump(key_info, os.path.join(final_path, 'key_info.pkl'))
    obsn_mask = {}
    for split in SPLITS:
        obsn_mask[split] = obtain_obsn_mask(inputs[split])
    for split in SPLITS:
        inputs[split]['usual'] = convert_sex(inputs[split]['usual'], key_info)
    # extract target columns for scaling
    len_meta_keys = len(key_info['meta_keys'])
    # scaling
    target_idx = {}
    target_idx['usual'] = np.arange(len_meta_keys, len_meta_keys + len(key_info['usual_categories']))
    target_idx['unusual'] = np.arange(len_meta_keys, len_meta_keys + len(key_info['unusual_categories']))
    scaled_inputs = {}
    scaled_inputs['train'], scalers = scale_data(inputs['train'], obsn_mask['train'], target_idx=target_idx)
    for split in HELDOUT:
        scaled_inputs[split], scalers = scale_data(inputs[split], obsn_mask[split], target_idx=target_idx, scalers=scalers)
    del inputs
    for split in SPLITS:
        joblib.dump({
            'inputs': scaled_inputs[split],
            'targets': np.array(targets[split]),
            'times': np.array(times[split]),
            }, os.path.join(segmented_df_path, f'scaled_{split_type}_{split}.pkl'))
    # ### Obtain max lengths
    # #### exam
    max_lengths = {}
    # #### prescription
    # convert list-like strings to lists
    # ## Embedding
    # ### Embedding medicine information
    # 6: code, 7: ingredient, 8: dosage, 9: unit
    for k in scaled_inputs['train'].keys():
        max_length = 0
        for split in SPLITS:
            for data in scaled_inputs[split][k]:
                if data.shape[0] > max_length:
                    max_length = data.shape[0]

        max_lengths[k] = max_length
    prs_inputs = {}
    for split in SPLITS:
        prs_inputs[split] = literal_eval_lists(split_prs[split]['inputs'])
    max_length = 0
    max_num = 0
    for split in SPLITS:
        for data in prs_inputs[split]:
            if data.shape[0] > max_length:
                max_length = data.shape[0]
            current_num = [len(seq) for seq in data[:, -1] if seq != 'nan']
            if len(current_num) > 0:
                if max(current_num) > max_num:
                    max_num = max(current_num)
    max_lengths['prs'] = max_length
    max_lengths['prs_num'] = max_num
    del split_prs
    embedded_prs_inputs = {}
    embedded_prs_inputs['train'], [ingredient_embed_dict, unit_embed_dict] = embed(prs_inputs['train'], [7, 9], [6, 9])
    for split in HELDOUT:
        embedded_prs_inputs[split], _ = embed(prs_inputs[split], [7, 9], [6, 9], [ingredient_embed_dict, unit_embed_dict])
    del prs_inputs
    # ### Embedding times
    # make dictionary using training data
    time_idx = key_info['meta_keys'].index('relative_date_in_month')
    # ### Embedding conditions
    embedded_inputs = {}
    embedded_inputs['train'] = {}
    embedded_inputs['train']['usual'], [time_embed_dict] = embed(scaled_inputs['train']['usual'], [time_idx], [time_idx], special_token=['<UNK>', '<CLS>'])
    embedded_inputs['train']['unusual'], [time_embed_dict] = embed(scaled_inputs['train']['unusual'], [time_idx], [time_idx], [time_embed_dict], update_dict=True)
    embedded_prs_inputs['train'], [time_embed_dict] = embed(embedded_prs_inputs['train'], [time_idx], [time_idx], [time_embed_dict], update_dict=True)
    for split in HELDOUT:
        embedded_inputs[split] = {}
        embedded_inputs[split]['usual'], _ = embed(scaled_inputs[split]['usual'], [time_idx], [time_idx], [time_embed_dict])
        embedded_inputs[split]['unusual'], _ = embed(scaled_inputs[split]['unusual'], [time_idx], [time_idx], [time_embed_dict])
        embedded_prs_inputs[split], _ = embed(embedded_prs_inputs[split], [time_idx], [time_idx], [time_embed_dict])
    del scaled_inputs
    embedded_times = {}
    embedded_times['train'], [condition_embed_dict] = embed(np.expand_dims(times['train'], 1), [1], [1])
    for split in HELDOUT:
        embedded_times[split], _ = embed(np.expand_dims(times[split], 1), [1], [1], [condition_embed_dict])
    for split in SPLITS:
        embedded_times[split] = np.squeeze(embedded_times[split])
    # ### Saving information
    joblib.dump(max_lengths, os.path.join(segmented_df_path, f'length_info_{split_type}.pkl'))
    joblib.dump(max_lengths, os.path.join(final_path, f'length_info_{split_type}.pkl'))
    joblib.dump({
        'code_embed': ingredient_embed_dict,
        'unit_embed': unit_embed_dict
        }, os.path.join(segmented_df_path, 'prs_embed_dict.pkl'))
    joblib.dump(time_embed_dict, os.path.join(segmented_df_path, 'time_embed_dict.pkl'))
    joblib.dump(condition_embed_dict, os.path.join(segmented_df_path, 'condition_embed_dict.pkl'))
    joblib.dump({
        'code_embed': ingredient_embed_dict,
        'unit_embed': unit_embed_dict
        }, os.path.join(final_path, 'prs_embed_dict.pkl'))
    joblib.dump(time_embed_dict, os.path.join(final_path, 'time_embed_dict.pkl'))
    joblib.dump(condition_embed_dict, os.path.join(final_path, 'condition_embed_dict.pkl'))
    # scalers
    joblib.dump(scalers, os.path.join(segmented_df_path, f'scalers_{split_type}.pkl'))
    joblib.dump(scalers, os.path.join(final_path, f'scalers_{split_type}.pkl'))
    for split in SPLITS:
    # ### Saving data
        joblib.dump({
            'inputs': embedded_inputs[split],
            'targets': targets[split],
            'times': embedded_times[split],
            }, os.path.join(segmented_df_path, f'embedded_{split_type}_{split}.pkl'))
    for split in SPLITS:
        joblib.dump({
            'inputs': embedded_prs_inputs[split],
            'targets': targets[split],
            'times': embedded_times[split],
            }, os.path.join(segmented_df_path, f'embedded_prescription_{split}.pkl'))
    # ### Dropping meta info
    len_meta_key = len(key_info['meta_keys'])# - 1 # to preserve date info
    # ## Reducing data
    # ### Make reduced version - max_seq_len = 24 - sampling information will be saved
    final_inputs = {}
    for split in SPLITS:
        final_inputs[split] = {}
    for key in embedded_inputs['train'].keys():
        print(key)
        reserve_idx = [key_info['meta_keys'].index('relative_date_in_month')] + list(np.arange(len(key_info[key + '_categories'])) + len_meta_key)

        for split in SPLITS:
            final_inputs[split][key] = np.array(
                [data[:, reserve_idx] for data in embedded_inputs[split][key]])
    reserve_idx = [key_info['meta_keys'].index('relative_date_in_month')] + list(np.arange(len(key_info['prs_keys'])) + len_meta_key)
    final_prs_inputs = {}
    for split in SPLITS:
        final_prs_inputs[split] = np.array(
            [data[:, reserve_idx] for data in embedded_prs_inputs[split]])
    del embedded_inputs
    del embedded_prs_inputs
    final_obsn_masks = {}
    for data_type in obsn_mask.keys():
        final_obsn_mask = {}
        for category in obsn_mask[data_type].keys():
            reserve_idx = [key_info['meta_keys'].index('relative_date_in_month')] + list(np.arange(len(key_info[category + '_categories'])) + len_meta_key)
            final_data = []
            for data in obsn_mask[data_type][category]:
                final_data.append(data[:, reserve_idx])
            final_obsn_mask[category] = np.array(final_data)
        final_obsn_masks[data_type] = final_obsn_mask
    egfr_idx = list(key_info['usual_categories']).index('eGFR')
    reduced_inputs = {}
    reduced_obsn = {}
    seq_idx = {}
    for split in SPLITS:
        reduced_inputs[split], reduced_obsn[split], seq_idx[split] = reduce_seq(final_inputs[split], final_obsn_masks[split], max_lengths, egfr_idx=egfr_idx+1)
    del final_inputs
    del final_obsn_masks, obsn_mask
    prs_seq_idx = {}
    reduced_prs_inputs = {}
    for split in SPLITS:
        reduced_prs_inputs[split], prs_seq_idx[split] = reduce_prs(final_prs_inputs[split])
    del final_prs_inputs
    joblib.dump(seq_idx, os.path.join(final_path, f'reduced24_{split_type}_idx.pkl'))
    joblib.dump(prs_seq_idx, os.path.join(final_path, 'reduced24_prescription_idx.pkl'))
    reject_idx = {}
    for split in SPLITS:
        reject_idx[split] = reject_data(reduced_inputs[split], reduced_prs_inputs[split], embedded_times[split][:,0])
    for key in reduced_inputs['train'].keys():
        for split in SPLITS:
            reduced_inputs[split][key] = reduced_inputs[split][key][~reject_idx[split]]
    for split in SPLITS:
        reduced_prs_inputs[split] = reduced_prs_inputs[split][~reject_idx[split]]
    for split in SPLITS:
        targets[split] = np.array(targets[split])
    for split in SPLITS:
    # save sampling information
        joblib.dump({
    # ### Make reduced version2 - min_seq_len >= observation duration in month - the rejection is a constant operation based on the length; therefore, no need for saving the selected index information
            'inputs': reduced_inputs[split],
            'times': embedded_times[split][~reject_idx[split]],
            'targets': targets[split][~reject_idx[split]]
            }, os.path.join(final_path, f'reduced24_{split_type}_{split}.pkl'))
    for split in SPLITS:
        joblib.dump({'inputs': reduced_prs_inputs[split]},
                    os.path.join(final_path, f'reduced24_prescription_{split}.pkl'))
    for key in reduced_obsn['train'].keys():
        for split in SPLITS:
            reduced_obsn[split][key] = reduced_obsn[split][key][~reject_idx[split]]
    joblib.dump(reduced_obsn, os.path.join(final_path, f'reduced24_obsn_mask_{split_type}.pkl'))
    # ### new max lengths
    max_lengths = {}
    for k in reduced_inputs['train'].keys():
        max_length = 0
        for split in SPLITS:
            for data in reduced_inputs[split][k]:
                if data.shape[0] > max_length:
                    max_length = data.shape[0]

        max_lengths[k] = max_length
    max_length = 0
    max_num = 0
    for split in SPLITS:
        for data in reduced_prs_inputs[split]:
            if data.shape[0] > max_length:
                max_length = data.shape[0]
            current_num = [len(seq) for seq in data[:, -1] if seq != 'nan']
            if len(current_num) > 0:
                if max(current_num) > max_num:
                    max_num = max(current_num)
    max_lengths['prs'] = max_length
    max_lengths['prs_num'] = max_num
    joblib.dump(max_lengths, os.path.join(final_path, f'reduced24_length_info_{split_type}.pkl'))

if __name__ == '__main__':
    main()
