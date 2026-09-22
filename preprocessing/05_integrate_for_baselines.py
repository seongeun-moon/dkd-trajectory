import os
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.tensors import load_data
from paths import DATA_ROOT


def extract_unique_time(times, target_len=24):
    num_data = times.shape[0]
    unique_times = np.zeros((num_data, target_len))
    for i, t in enumerate(times):
        t = np.unique(t)
        t = np.delete(t, 0)
        if len(t) > target_len:
            unique_times = np.concatenate(
                (unique_times, 
                np.zeros((num_data, len(t) - target_len))), 
                axis=-1)
            target_len = len(t)
        else:
            unique_times[i, :len(t)] = t
            
    return unique_times.astype('int32')


def align_data(data, mask, time, ref_time, token=None, include_special_token=False):
    num_data, seq_len = ref_time.shape
    feature_dim = data.shape[-1]
    if not include_special_token:
        feature_dim -= 1
        data = data[:, :, 1:]
        mask = mask[:, :, 1:]
        
    aligned_data = np.zeros((num_data, seq_len, feature_dim), dtype='float32')
    aligned_mask = np.zeros_like(aligned_data, dtype='int32')
    if token is not None:
        if not include_special_token:
            aligned_token = np.zeros_like(aligned_data, dtype='int32')
        else: 
            aligned_token = np.zeros((num_data, seq_len, feature_dim-1), dtype='int32')
    else:
        aligned_token = None
    
    for i, d in enumerate(data):
        current_time = time[i]
        current_time = current_time [ np.nonzero(current_time) ]
        
        for ii, t in enumerate(current_time):
            idx = np.where(ref_time[i] == t)[0]
            aligned_data[i, idx] = d[ii]
            aligned_mask[i, idx] = mask[i][ii]
            
            if token is not None:
                aligned_token[i, idx] = token[i][ii]

    return aligned_data, aligned_mask, aligned_token


def main():
    data_path = os.path.join(DATA_ROOT, '')
    data_type = 'test'
    split_type = 'frequency'
    loaded = load_data(data_path, split_type, data_type)
    category_data, prs_data = loaded['category_data'], loaded['prs_data']
    integ_time = np.concatenate((category_data['time']['usual'], 
                                 category_data['time']['unusual'],
                                 prs_data['time']), axis=-1)
    unique_time = extract_unique_time(integ_time)
    integ_data, integ_obsn_mask, _ = align_data(
        data=category_data['inputs']['usual'],
        mask=category_data['obsn_mask']['usual'],
        time=category_data['time']['usual'],
        ref_time=unique_time,
        include_special_token=True
    )
    tmp_data, tmp_obsn_mask, _ = align_data(
        data=category_data['inputs']['unusual'],
        mask=category_data['obsn_mask']['unusual'],
        time=category_data['time']['unusual'],
        ref_time=unique_time,
        )
    integ_data = np.concatenate((integ_data, tmp_data), axis=-1)
    integ_obsn_mask = np.concatenate((integ_obsn_mask, tmp_obsn_mask), axis=-1)
    tmp_data, tmp_obsn_mask, prs_token = align_data(
        data=prs_data['code'],
        mask=np.where(prs_data['code'] == 0, 0, 1),
        time=prs_data['time'],
        ref_time=unique_time,
        token=prs_data['token']
        )
    integ_data = np.concatenate((integ_data, tmp_data), axis=-1)
    integ_obsn_mask = np.concatenate((integ_obsn_mask, tmp_obsn_mask), axis=-1)
    tmp_data, tmp_obsn_mask, _ = align_data(
        data=prs_data['dosage'],
        mask=np.where(prs_data['dosage'] == 0, 0, 1),
        time=prs_data['time'],
        ref_time=unique_time,
        )
    integ_data = np.concatenate((integ_data, tmp_data), axis=-1)
    integ_obsn_mask = np.concatenate((integ_obsn_mask, tmp_obsn_mask), axis=-1)
    tmp_data, tmp_obsn_mask, _ = align_data(
        data=prs_data['unit'],
        mask=np.where(prs_data['unit'] == 0, 0, 1),
        time=prs_data['time'],
        ref_time=unique_time,
        )
    integ_data = np.concatenate((integ_data, tmp_data), axis=-1)
    integ_obsn_mask = np.concatenate((integ_obsn_mask, tmp_obsn_mask), axis=-1)
    token_usual = np.tile(
        np.expand_dims(
            np.array(
                category_data['token']['usual']
            ), 
            [0, 1]
        ), 
        [integ_data.shape[0], integ_data.shape[1], 1]
    )
    token_unusual = np.tile(
        np.expand_dims(
            np.array(
                category_data['token']['unusual']
            ), 
            [0, 1]
        ), 
        [integ_data.shape[0], integ_data.shape[1], 1]
    )
    integ_token = np.concatenate((token_usual, token_unusual, prs_token, prs_token, prs_token), axis=-1)
    joblib.dump({
        'inputs': integ_data,
        'obsn_mask': integ_obsn_mask,
        'token': integ_token.astype('int32'),
        'times': unique_time,
    }, data_path + 'homogeneous_'+data_type+'.pkl')


if __name__ == '__main__':
    main()
