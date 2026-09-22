"""Conventional baselines (LR, SVM, LightGBM, XGBoost) via PyCaret.

The per-slice sequences are reduced to one row per slice (``--agg average``),
optionally with the observation-window length and the horizon condition
appended (``--concat_time --concat_cdt``), and handed to PyCaret, which does
the hyperparameter search on the validation split.
"""
import argparse
import os
import sys
from pathlib import Path

import joblib
import json
import numpy as np

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.endpoints import binarise, resolve_target_type
from common.evaluation import multiclass_score
from common.sampling import balance_data
from utils import dataset, integ_data, resolve_data_path


from torch.utils.data import DataLoader

from pycaret.classification import *

def prepare_data(args, path_to_data, sample_data):
    dim_info = {}
    num_data, seq_len, num_feature = sample_data['inputs'].shape
    if args.concat_time: num_feature += 1
    if args.agg == 'plain': 
        dim_info['input_dim'] = seq_len * num_feature
    else:
        dim_info['input_dim'] = num_feature
    if args.concat_cdt: dim_info['input_dim'] += 1
    dim_info['output_dim'] = len(np.unique(sample_data['targets'][:,-1]))

    time_info = joblib.load(path_to_data + 'time_embed_dict.pkl')
    dim_info['time_embed'] = len(time_info)
    return dim_info

class weighted_evaluation():
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __name__(self):
        return 'weighted_acc'
    
    def __call__(self, y, y_pred, **kwargs):
        target_classes = np.unique(y)
        acc = []
        for target_class in target_classes:
            acc.append(
                np.sum(y_pred[y==target_class] == y[y==target_class]) / np.sum(y==target_class)
            )
        return np.mean([a * w for a, w in zip(acc, weight)])


def main():
    """ A main function of the module.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=['lr', 'svm', 'lightgbm', 'xgb'], required=True)
    
    parser.add_argument('--agg', choices=['plain', 'average', 'last'], default='plain')
    parser.add_argument('--concat_time', action='store_true', dest='concat_time', default=False)
    parser.add_argument('--concat_cdt', action='store_true', dest='concat_cdt', default=False)


    ### environmental parameters
    parser.add_argument('--session_no', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Directory holding the preprocessed reduced24_* pickles. '
             'Defaults to $CKD_DATA_ROOT. The EHR extract is not distributed '
             'with this code.')

    ### data-related parameters
    parser.add_argument('--prs_features', type=str, nargs='+', default=['code'])
    parser.add_argument('--binary', action='store_true', dest='binary', default=False)
    parser.add_argument(
        '--target_type', choices=['ckd-stage', 'egfr-decline'], default=None,
        help='Which endpoint --binary collapses the label to. '
             'ckd-stage: CKD stage at the horizon is above baseline. '
             'egfr-decline: horizon eGFR <= 70%% of baseline. '
             'Required whenever --binary is set -- the two are not '
             'interchangeable and give different models.')
    parser.add_argument('--balanced', action='store_true', dest='balanced', default=False)
    parser.add_argument('--balance_run', type=int, default=0)

    args = parser.parse_args()
    if args.binary:
        args.target_type = resolve_target_type(args)
    print(args)

    data_path = resolve_data_path(args.data_path)
    result_path = args.result_path

    if (torch.cuda.is_available()):
        print('Using GPU')
        print(torch.cuda.device_count())
        device = torch.device('cuda')
        torch.cuda.set_device(0)
    else:
        print('Using CPU')
        device = torch.device('cpu')
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### prepare result folder
    if args.session_no is None:
        session_no = args.model

        if not args.binary:
            session_no += '_stage'
        session_no += '_' + args.agg
        if args.concat_time: session_no += '_time'
        if args.concat_cdt: session_no += '_cdt'
        if args.balanced: session_no += '_balanced' + str(args.balance_run)
        session_no += '_' + str(args.seed)
        print('Session no: {}'.format(session_no))

        result_path = result_path + str(session_no) + '/'
        os.makedirs(result_path)
    else:
        session_no = args.session_no
        if os.path.exists(os.path.join(result_path, str(session_no), 'best_valid_loss_model.pt')):
            print(f'Continue training from session {session_no}')
        else:
            os.makedirs(os.path.join(result_path, str(session_no)), exist_ok=True)
            print('Session no: {}'.format(session_no))
        
        result_path = result_path + str(session_no) + '/'
    
    ### save arguments
    with open(result_path+'args.txt', 'w') as f:
        json.dump(args.__dict__, f)

    ### load data
    train_data = integ_data(data_path=data_path, data_type='train', prs_include=args.prs_features)
    valid_data = integ_data(data_path=data_path, data_type='valid', prs_include=args.prs_features) 
    test_data = integ_data(data_path=data_path, data_type='test', prs_include=args.prs_features)
    if args.binary:
        for split in (train_data, valid_data, test_data):
            binarise(split, args.target_type)
    
    if args.balanced:
        try:
            idx = joblib.load(os.path.join(data_path, 'balancing_idx.pkl'))
        except FileNotFoundError:
            idx = {'train': None, 'valid': None}
        train_data, balancing_idx_train = balance_data(train_data, idx=idx['train'], run=args.balance_run)
        valid_data, balancing_idx_valid = balance_data(valid_data, idx=idx['valid'], run=args.balance_run)
        if idx['train'] is None:
            joblib.dump({
                'train': balancing_idx_train, 
                'valid': balancing_idx_valid,
                }, os.path.join(data_path, 'balancing_idx.pkl'))
    
    train_targets = train_data['targets'][:, -1]
    num_class = len(np.unique(train_targets))
    num_data = len(train_targets)
    class_weight = []
    for i in range(num_class):
        class_ratio = np.where(train_targets == i)[0].shape[0] / num_data
        class_weight.append(1/class_ratio*0.5)
    class_weight = torch.tensor(class_weight).float().to(device)
    print(class_weight)
    weighted_acc = weighted_evaluation(class_weight)
    
    ### set up model
    
    train_dataloader = DataLoader(
        dataset(train_data, device, args.agg, args.concat_time, args.concat_cdt, missing_na=True, return_tensor=False), 
        batch_size=train_data['inputs'].shape[0],
        shuffle=True
        )
    test_dataloader = DataLoader(
        dataset(test_data, device, args.agg, args.concat_time, args.concat_cdt, missing_na=True, return_tensor=False), 
        batch_size=test_data['inputs'].shape[0],
        shuffle=False
        )
    dtrain = next(iter(train_dataloader))
    dtest = next(iter(test_dataloader))
    
    train_df = pd.DataFrame(dtrain[0])
    train_df['target'] = dtrain[1]

    test_df = pd.DataFrame(dtest[0])
    test_df['target'] = dtest[1]
    
    if args.model == 'lightgbm':
        n_jobs = 1
    else:
        n_jobs = -1

    setup(data=train_df, target='target', use_gpu=(device.type == 'cuda'),
                      n_features_to_select=1, n_jobs=n_jobs)
    add_metric('weighted_acc', 'weighted_evaluation', weighted_acc)
    if args.model == 'xgb':
        model = create_model('xgboost')
    elif args.model == 'lr':
        model = create_model('lr')
    elif args.model == 'svm':
        model = create_model('rbfsvm')
    elif args.model == 'lightgbm':
        model = create_model('lightgbm')
    else:
        raise NotImplementedError

    tuned_model = tune_model(model, optimize='weighted_acc', n_iter=10, fold=5, choose_better=True, verbose=True)

    prediction = predict_model(tuned_model, data=test_df)
    prediction_class = prediction['prediction_label']
    prediction = np.where(prediction['prediction_label'] == 1, prediction['prediction_score'], 1 - prediction['prediction_score'])
    

    targets = test_df['target']
    acc = np.equal(targets, prediction_class).sum() / targets.shape[0]
    auroc = multiclass_score(targets, prediction, 'auroc')
    auprc = multiclass_score(targets, prediction, 'auprc')
    recall = multiclass_score(targets, prediction_class, 'recall')
    
    with open(os.path.join(result_path, 'result.json'), 'w') as f:
        json.dump({
            'prediction': list(map(float, list(prediction))),
            'target': list(map(float, list(targets))),
            'acc': float(acc),
            'auroc': float(auroc),
            'recall': float(recall),
            'auprc': float(auprc)
        }, f)

    print(f'ACC: {acc :.3f}\tAUROC: {auroc :.3f}\tRecall: {recall :.3f}\tAUPRC: {auprc :.3f}')
        
    print('Session no: {}'.format(session_no))
    print(args)


if __name__ == '__main__':
    main()
