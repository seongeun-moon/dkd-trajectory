import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.tensors import load_data

def resolve_data_path(data_path=None):
    """Locate the preprocessed dataset directory.

    Resolution order: the argument > ``$CKD_DATA_ROOT``. Site-specific absolute
    paths are deliberately not hard-coded: the SNUBH EHR extract is governed by
    an IRB data-use agreement and is never shipped with this code.
    """
    path = data_path or os.environ.get('CKD_DATA_ROOT')
    if not path:
        raise SystemExit(
            'No dataset directory configured. Pass --data_path <dir> or set '
            'CKD_DATA_ROOT to the directory holding the preprocessed '
            'reduced24_*.pkl / key_info.pkl files.')
    if not os.path.isdir(path):
        raise SystemExit(f'Dataset directory does not exist: {path}')
    return os.path.join(path, '')

        


def integ_data(
    data_path=None,
    split_type='frequency',
    data_type='train',
    prs_include=['code', 'dosage', 'unit'],
    ):
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
                aligned_token = np.zeros((num_data, seq_len, feature_dim+1), dtype='int32')
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

    data = load_data(
        data_path=data_path,
        split_type=split_type,
        data_type=data_type,
    )
    category_data = data['category_data']
    prs_data = data['prs_data']

    integ_time = np.concatenate((
        category_data['time']['usual'], 
        category_data['time']['unusual'],
        prs_data['time']
        ), axis=-1)

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

    for prs_type in prs_include:
        tmp_data, tmp_obsn_mask, tmp_token = align_data(
            data=prs_data[prs_type],
            mask=np.where(prs_data[prs_type] == 0, 0, 1),
            time=prs_data['time'],
            ref_time=unique_time,
            token=prs_data['token'] if prs_type == 'code' else None
            )
        if prs_type == 'code':
            prs_token = tmp_token
            

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
    return {
        'inputs': integ_data,
        'obsn_mask': integ_obsn_mask,
        'token': integ_token.astype('int32'),
        'times': unique_time,
        'targets': data['targets'], 
        'conditions': data['conditions'], 
        'length_info': data['length_info']
    }

class dataset(Dataset):
    def __init__(self, data, device, agg='plain', concat_time=False, concat_cdt=False, missing_na=False, return_tensor=True):
        
        self.target = data['targets'][:, -1]
        self.device = device
        self.return_tensor = return_tensor
        if missing_na:
            data['inputs'][data['obsn_mask'] == 0] = np.nan
        
        num_data, len_seq, num_feature = data['inputs'].shape
        if agg == 'average':
            self.input = np.zeros((num_data, num_feature))
            last_time = np.zeros((num_data, 1))
            for i, (d, m, t) in enumerate(zip(data['inputs'], data['obsn_mask'], data['times'])):
                last_time[i] = t[t > 0][-1]
                for ii in range(num_feature):
                    v = d[:, ii][m[:, ii] == 1]
                    if len(v) > 0:
                        self.input[i, ii] = np.mean(v)

            if concat_time:
                self.input = np.concatenate((last_time, self.input), axis=-1)
        elif agg == 'last':
            self.input = np.zeros((num_data, num_feature))
            last_time = np.zeros((num_data, 1))
            for i, (d, m, t) in enumerate(zip(data['inputs'], data['obsn_mask'], data['times'])):
                last_time[i] = t[t > 0][-1]
                for ii in range(num_feature):
                    v = d[:, ii][m[:, ii] == 1]
                    if len(v) > 0:
                        self.input[i, ii] = v[-1]

            if concat_time:
                self.input = np.concatenate((last_time, self.input), axis=-1)
        else:
            if concat_time:
                data['inputs'] = np.concatenate((np.expand_dims(data['times'], -1), data['inputs']), axis=-1)
            self.input = data['inputs'].reshape(num_data, -1)
        if concat_cdt:
            self.input = np.concatenate((self.input, data['conditions'][:, -1].reshape(-1, 1)), axis=-1)
    
    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        batch_input = self.input[idx]
        batch_target = self.target[idx]

        if self.return_tensor:
            batch_input = torch.tensor(batch_input).float().to(self.device)
            batch_target = torch.tensor(batch_target).long().to(self.device)
        return batch_input, batch_target


    


