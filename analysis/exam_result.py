import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import multiclass_score
from paths import OUT_ROOT, RESULT_ROOT


def add_recall(result):
    """ To add recall score to the result dictionary."""
    if 'recall' in result.keys():
        pass
    else:
        result['recall'] = multiclass_score(result['target'], result['prediction'], 'recall')
    return result

if __name__ == '__main__':
    rows = []
    for session in sorted(os.listdir(RESULT_ROOT)):
        session_dir = os.path.join(RESULT_ROOT, session)
        if not os.path.isdir(session_dir):
            continue
        print(session)
        model = session.split('_')[0]
        if model == 'snubh':
            model = 'ours'
            result = joblib.load(os.path.join(session_dir, 'best_valid_loss_result.pkl'))
            result = add_recall(result)
        else:
            with open(os.path.join(session_dir, 'result.json'), 'r') as f:
                result = json.load(f)
        auroc = result['auroc']
        auprc = result['auprc']
        recall = result['recall']
        print(f'\tAUROC: {auroc :.3f}\tRecall: {recall :.3f}\tAUPRC: {auprc :.3f}')

        try:
            seed = int(session.split('_')[-1])
            balance_run = session.split('_')[-2].replace('balanced', '')
        except ValueError:
            seed = session.split('_')[-2]
            balance_run = session.split('_')[-1].replace('balanced', '')
        rows.append({'model': model, 'seed': seed, 'balance_run': balance_run,
                     'session': session, 'auroc': auroc, 'recall': recall,
                     'auprc': auprc})

    df = pd.DataFrame(rows, columns=['model', 'seed', 'balance_run', 'session',
                                     'auroc', 'recall', 'auprc'])
    os.makedirs(OUT_ROOT, exist_ok=True)
    df.to_csv(os.path.join(OUT_ROOT, 'result.csv'))