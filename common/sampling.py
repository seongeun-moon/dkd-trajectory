"""Class balancing shared by the training drivers."""

import numpy as np


def balance_data(data, idx=None, run=0):
    if idx is None:
        targets = data['targets'][:, -1]
        pos_idx = np.where(targets > 0)[0]
        num_pos = len(pos_idx)

        neg_idx = np.where(targets == 0)[0]
        num_run = int(len(neg_idx) / num_pos)
        neg_idxs = np.random.choice(neg_idx, size=[num_run, num_pos], replace=False)

        idx = {}
        for i, neg_idx in enumerate(neg_idxs):
            idx[i] = np.append(pos_idx, neg_idx)

    balanced_data = {}
    for k, v in data.items():
        if k == 'length_info':
            continue
        if type(v) == dict:
            balanced_data[k] = {}
            for kk, vv in v.items():
                if kk == 'token':
                    balanced_data[k][kk] = vv
                    continue
                if type(vv) == dict:
                    balanced_data[k][kk] = {}
                    for kkk, vvv in vv.items():
                        try:
                            balanced_data[k][kk][kkk] = vvv[idx[run]]
                        except:
                            raise
                else:
                    balanced_data[k][kk] = vv[idx[run]]
        else:
            try:
                balanced_data[k] = v[idx[run]]
            except IndexError:
                raise
    balanced_data['length_info'] = data['length_info']
    return balanced_data, idx
