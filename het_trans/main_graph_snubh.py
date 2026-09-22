"""Training driver for the heterogeneous Performer model.

Three parallel encoders (frequently recorded variables, low-frequency
variables, prescriptions) feed a Performer encoder/decoder whose [CLS] token
carries the classification head, with an auxiliary eGFR regression head.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.endpoints import binarise, resolve_target_type
from common.evaluation import multiclass_score
from common.sampling import balance_data
from common.tensors import load_data
from utils import AdamP, iterate_minibatches, obtain_session_no, save_data


def test(model,
         data,
         batch_size,
         device,
         loss_function=None,
         recon_loss_function=None,
         mlm_loss_function=None,
         reg_weight=0.0001,
         mlm_weight=0.1,
         return_attention_score=False):
    """ This function evalutes the given model using the given input and target data. If the loss functions are given, the lists of obtained loss values are also returned.
    """
    loss_trj = []
    pred_loss_trj = []
    reg_loss_trj = []
    mlm_loss_trj = []
    stacked_target = []
    stacked_prediction = []
    stacked_attention_score = {}
    

    model.eval()

    with torch.no_grad():
        for batch in iterate_minibatches(category=data['category_data'],
                                         prs=data['prs_data'],
                                         conditions=data['conditions'],
                                         targets=data['targets'],
                                         batch_size=batch_size,
                                         shuffle=False, 
                                         device=device):
            inputs, targets, times, obsn_mask, tokens = batch

            prediction, regression, mlms, attention_score = model(inputs, tokens, times, targets['condition'], obsn_mask, mode='eval', return_attn_score=return_attention_score)

            if loss_function is None:
                pred_loss = torch.zeros(1)
                reg_loss = torch.zeros(1)
                mlm_loss = torch.zeros(1)
                loss = torch.zeros(1)
            else:
                pred_loss = loss_function(prediction, targets['CKD_targets'])
                reg_loss = recon_loss_function(regression, targets['eGFR_targets'].view(-1, 1))
                
                mlm_loss = torch.zeros(1).to(device)
                for key in mlms.keys():
                    if key == 'prs':
                        mlm_loss += mlm_loss_function(mlms[key][0], mlms[key][1])
                    else:
                        mlm_loss += recon_loss_function(torch.squeeze(mlms[key][0]), mlms[key][1])

                loss = pred_loss + reg_weight * reg_loss + mlm_weight * mlm_loss

            loss_trj.append(loss.item())
            pred_loss_trj.append(pred_loss.item())
            reg_loss_trj.append(reg_loss.item())
            mlm_loss_trj.append(mlm_loss.item())

            stacked_target.append(torch.squeeze(targets['CKD_targets']).data.cpu().numpy())
            stacked_prediction.append(torch.squeeze(prediction).data.cpu().numpy())
            

            for key in attention_score.keys():
                if attention_score[key] is not None:
                    try:
                        stacked_attention_score[key] = np.concatenate((stacked_attention_score[key], attention_score[key].data.cpu().numpy()), axis=-1)
                    except (KeyError, AttributeError):
                        stacked_attention_score[key] = attention_score[key].data.cpu().numpy()
    return loss_trj, pred_loss_trj, reg_loss_trj, mlm_loss_trj, stacked_prediction, stacked_target, stacked_attention_score#, {'input': stacked_inputs, 'time': stacked_times, 'mask': stacked_masks, 'token': stacked_tokens}


def train(args,
          model,
          train_data,
          valid_data,
          test_data,
          device,
          result_path='./',
          previous_model=None,
          save_log=True):
    """ This function implements the entire training process including the validation and test with the best model. Class weights for imbalanced dataset and early stopping schemes are adopted. 
    """

    if save_log:
        ### set a logger
        logging.basicConfig(filename=result_path+"log.txt",
                            format='%(asctime)s %(message)s',
                            filemode='a')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    writer = SummaryWriter(result_path) if save_log else None

    ### calculate weights for each target class based on the number of occurences
    num_class = len(np.unique(train_data['targets'][:, -1]))
    num_data = train_data['targets'][:, -1].shape[0]
    class_weight = []
    for i in range(num_class):
        class_ratio = np.where(train_data['targets'][:, -1] == i)[0].shape[0] / num_data
        class_weight.append(1/class_ratio*0.5)
    class_weight = torch.tensor(class_weight).float().to(device)

    print(class_weight)
    loss_function = nn.CrossEntropyLoss(weight=class_weight)
    if save_log:
        logging.info('Class weight: {}'.format(class_weight))

    ### define learning objects
    optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    recon_loss_function = nn.MSELoss(reduction='mean')
    mlm_loss_function = nn.CrossEntropyLoss()

    ### define initial parameters for early stopping
    best_valid_loss = 10000
    best_valid_epoch = 0
    patience_count = 0
    test_auroc = 0

    ### recover information from previous model
    if previous_model is not None:
        model.eval()
        model.load_state_dict(previous_model['model'])
        optimizer.load_state_dict(previous_model['optimizer'])
        start_epoch = previous_model['epoch'] + 1
        if save_log:
            logging.info(f'Continue training from {result_path}')
    else:
        start_epoch = 0
        if save_log:
            logging.info('Start new training')

    for epoch in range(start_epoch, args.epoch):
        ### train model
        model.train()

        train_loss = []
        train_pred_loss = []
        train_reg_loss = []
        train_mlm_loss = []

        train_target = []
        train_prediction = []

        for batch in iterate_minibatches(category=train_data['category_data'],
                                        prs=train_data['prs_data'],
                                        conditions=train_data['conditions'],
                                        targets=train_data['targets'],
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        device=device):

            # with profiler.profile(with_stack=True, profile_memory=True, use_cuda=True) as prof:
            inputs, targets, times, obsn_mask, tokens = batch

            prediction, regression, mlms, _ = model(inputs, tokens, times, targets['condition'], obsn_mask, mode='train')
            pred_loss = loss_function(prediction, targets['CKD_targets'])
            reg_loss = recon_loss_function(regression, targets['eGFR_targets'].view(-1, 1))
            
            mlm_loss = torch.zeros(1).to(device)
            for key in mlms.keys():
                if key == 'prs':
                    mlm_loss += mlm_loss_function(mlms[key][0], mlms[key][1])
                else:
                    mlm_loss += recon_loss_function(torch.squeeze(mlms[key][0]), mlms[key][1])

            loss = pred_loss + args.reg_weight * reg_loss + args.mlm_weight * mlm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            train_pred_loss.append(pred_loss.item())
            train_reg_loss.append(reg_loss.item())
            train_mlm_loss.append(mlm_loss.item())

            train_prediction.append(torch.squeeze(prediction).data.cpu().numpy())
            train_target.append(targets['CKD_targets'].data.cpu().numpy())
            
            # print(prof.key_averages().table())
            # import pdb; pdb.set_trace()

        scheduler.step()

        train_prediction = np.concatenate(tuple(train_prediction), axis=0)
        train_target = np.concatenate(tuple(train_target), axis=0)    
        train_prediction_class = np.argmax(train_prediction, axis=-1)

        train_acc = np.equal(train_target, train_prediction_class).sum() / train_target.shape[0]
        train_auroc = multiclass_score(train_target, train_prediction, 'auroc')

        ### obtain validation performance
        valid_loss, valid_pred_loss, valid_reg_loss, valid_mlm_loss, _, _, _ = test(
            model=model,
            data=valid_data,
            batch_size=args.batch_size,
            device=device,
            loss_function=loss_function,
            recon_loss_function=recon_loss_function,
            mlm_loss_function=mlm_loss_function,
            reg_weight=args.reg_weight,
            mlm_weight=args.mlm_weight)

        ### check validation performance
        if np.mean(valid_pred_loss) < best_valid_loss:
            best_valid_epoch = epoch
            best_valid_loss = np.mean(valid_pred_loss)
            
            model.eval()
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, result_path + '/best_valid_loss_model.pt')

            ### obtain test performance
            _, _, _, _, test_prediction, test_target, _ = test(
                model=model,
                data=test_data,
                batch_size=args.batch_size,
                device=device,
                loss_function=loss_function,
                recon_loss_function=recon_loss_function,
                mlm_loss_function=mlm_loss_function,
                reg_weight=args.reg_weight,
                mlm_weight=args.mlm_weight
                )

            ### calculate test performance           
            test_prediction = np.concatenate(tuple(test_prediction), axis=0)
            test_target = np.concatenate(tuple(test_target), axis=0)
            test_prediction_class = np.argmax(test_prediction, axis=-1)

            test_acc = np.equal(test_target, test_prediction_class).sum() / test_target.shape[0]
            test_auroc = multiclass_score(test_target, test_prediction, 'auroc')
            test_auprc = multiclass_score(test_target, test_prediction, 'auprc')
            test_recall = multiclass_score(test_target, test_prediction_class, 'recall')

            save_data('best_valid_loss_result.pkl',
                    data={
                        'prediction': test_prediction,
                        'target': test_target,
                        'acc': test_acc,
                        'auroc': test_auroc,
                        'auprc': test_auprc
                    },
                    path=result_path)

            
            if writer is not None:
                writer.add_scalar('auroc/test', test_auroc, best_valid_epoch)

            patience_count = 0
        else:
            patience_count += 1

        if writer is not None:
            writer.add_scalar('loss/train', np.mean(train_loss), epoch)
            writer.add_scalar('pred_loss/train', np.mean(train_pred_loss), epoch)
            writer.add_scalar('mlm_loss/train', np.mean(train_mlm_loss), epoch)
            writer.add_scalar('reg_loss/train', np.mean(train_reg_loss), epoch)
            writer.add_scalar('auroc/train', train_auroc, epoch)
            writer.add_scalar('loss/valid', np.mean(valid_loss), epoch)
            writer.add_scalar('pred_loss/valid', np.mean(valid_pred_loss), epoch)
            writer.add_scalar('mlm_loss/valid', np.mean(valid_mlm_loss), epoch)
            writer.add_scalar('reg_loss/valid', np.mean(valid_reg_loss), epoch)
            writer.add_scalar('patience', patience_count, epoch)
        

        print(
            'Train Epoch: {}\n\tLoss: {:.3f}\tPrediction loss: {:.3f}\tRegression loss: {:.3f}\tMLM loss: {:.3f}\tACC: {:.3f}\tAUROC: {:.3f}'
            .format(epoch, np.mean(train_loss), np.mean(train_pred_loss), np.mean(train_reg_loss), np.mean(train_mlm_loss), train_acc, train_auroc)
        )
        print(
            '\tValid loss: {:.3f}\tPrediction loss: {:.3f}\tRegression loss: {:.3f}\tMLM loss: {:.3f}\n\tTest AUROC {:.3f}\tBest valid loss: {:.3f} of epoch {}'
            .format(np.mean(valid_loss), np.mean(valid_pred_loss), np.mean(valid_reg_loss), np.mean(valid_mlm_loss), test_auroc, best_valid_loss, best_valid_epoch)
        )

        if save_log:
            logging.info(
                'Train Epoch: {}\n\tLoss: {:.3f}\tPrediction loss: {:.3f}\tRegression loss: {:.3f}\tMLM loss: {:.3f}\tACC: {:.3f}\tAUROC: {:.3f}'
                .format(epoch, np.mean(train_loss), np.mean(train_pred_loss), np.mean(train_reg_loss), np.mean(train_mlm_loss), train_acc, train_auroc)
            )
            logging.info(
                '\tValid loss: {:.3f}\tPrediction loss: {:.3f}\tRegression loss: {:.3f}\tMLM loss: {:.3f}\n\tTest AUROC {:.3f}\tBest valid loss: {:.3f} of epoch {}'
                .format(np.mean(valid_loss), np.mean(valid_pred_loss), np.mean(valid_reg_loss), np.mean(valid_mlm_loss), test_auroc, best_valid_loss, best_valid_epoch)
            )

        if patience_count == args.patience:
            break

    print('Test Epoch: {}\n\tAccuracy: {:.3f}\tAUROC: {:.3f}\tAUPRC: {:.3f}\tRecall: {:.3f}'.format(best_valid_epoch, test_acc, test_auroc, test_auprc, test_recall))
    if save_log:
        logging.info(
            'Test Epoch: {}\tAccuracy: {:.3f}\tAUROC: {:.3f}\tAUPRC: {:.3f}\tRecall: {:.3f}'.format(best_valid_epoch, test_acc, test_auroc, test_auprc, test_recall)
        )

def prepare_data(args, path_to_data, sample_data):
    ### data preparation
    key_info = joblib.load(path_to_data + 'key_info.pkl')
    prs_info = joblib.load(path_to_data + 'prs_embed_dict.pkl')
    categories = list(sample_data['category_data']['inputs'].keys())

    dim_info = {}
    for category in categories:
        dim_info[category+'_input_dim'] = len(key_info[category+'_categories'])
        dim_info[category+'_len'] = sample_data['length_info'][category]
    dim_info['prs_input_dim'] = sample_data['length_info']['prs_num']
    dim_info['prs_len'] = sample_data['length_info']['prs']
    dim_info['code_embed'] = len(prs_info['code_embed'])
    dim_info['unit_embed'] = len(prs_info['unit_embed'])
    dim_info['importance_embed'] = 3
    dim_info['obsn_embed'] = 2
    dim_info['num_condition'] = max(3, len(np.unique(sample_data['conditions'][:,1]))+1)
    dim_info['input_dim'] = args.embed_dim * len(args.feature_set)
    if 'prs' in args.feature_set: dim_info['input_dim'] += args.embed_dim
    dim_info['output_dim'] = len(np.unique(sample_data['targets'][:,-1]))

    if args.pos_encode:
        dim_info['time_embed'] = 2
    else:
        time_info = joblib.load(path_to_data + 'time_embed_dict.pkl')
        dim_info['time_embed'] = len(time_info)
    return key_info, prs_info, categories, dim_info


def main():
    """ A main function of the module.
    """
    parser = argparse.ArgumentParser()

    ### model structure parameters
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--cls_hidden_dim', type=int, nargs='+', default=[256])
    parser.add_argument('--mlm_hidden_dim', type=int, nargs='+', default=[256])
    parser.add_argument('--trans_pos_encode', action='store_true', dest='pos_encode', default=False)

    ### transformer parameters
    parser.add_argument('--num_head', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--num_layers', type=int, default=1)

    ### learning scheme parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--reg_weight', type=float, default=0.0001)
    parser.add_argument('--mlm_weight', type=float, default=0.1)

    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--p_m',
                        type=float,
                        default=0.3,
                        help='Masking ratio to conduct pretraining task')
    parser.add_argument('--patience', type=int, default=20)

    ### environmental parameters
    parser.add_argument('--test',
                        action='store_true',
                        dest='test',
                        default=False)
    parser.add_argument('--session_no', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--result_path', type=str, default='./result/')

    ### data-related parameters
    parser.add_argument('--feature_set', type=str, nargs='+', default=['usual', 'unusual', 'prs'])
    parser.add_argument('--binary', action='store_true', dest='binary_classification', default=False)
    parser.add_argument(
        '--target_type', choices=['ckd-stage', 'egfr-decline'], default=None,
        help='Which endpoint --binary collapses the label to. '
             'ckd-stage: CKD stage at the horizon is above baseline '
             '(what the archived runs used). '
             'egfr-decline: horizon eGFR <= 70%% of baseline. '
             'Required whenever --binary is set -- the two are not '
             'interchangeable and give different models. May also come from '
             '--config.')
    parser.add_argument(
        '--config', type=str, default=None,
        help="A run's config.yaml; exp_setting.target_type is read from it. "
             'Point this at an archived run directory to train the same '
             'endpoint it used. An explicit --target_type wins over it.')
    parser.add_argument('--balanced', action='store_true', dest='balanced', default=False)
    parser.add_argument('--balance_run', type=int, default=0)
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Directory holding the preprocessed reduced24_* pickles. '
             'Defaults to $CKD_DATA_ROOT. The EHR extract is not distributed '
             'with this code.')

    args = parser.parse_args()
    if args.binary_classification:
        args.target_type = resolve_target_type(args)
    print(args)

    from model.het_perf_obsn_mask_snubh import trans_model

    data_path = resolve_data_path(args)
    result_path = args.result_path

    if torch.cuda.is_available():
        print('Using GPU')
        print(torch.cuda.device_count())
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        device = torch.device('cuda')#('cuda:{}'.format(0))
        torch.cuda.set_device(0)
    elif torch.backends.mps.is_available():
        print('Using MPS')
        device = torch.device('mps')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_data = load_data(data_path=data_path, data_type='test')

    if args.test:
        model_file = result_path + str(args.session_no) + '/best_valid_loss_model.pt'
        if args.binary_classification:
            binarise(test_data, args.target_type)

        key_info, prs_info, categories, dim_info = prepare_data(
            args,
            data_path,
            test_data
        )

        model = trans_model(
            args,
            categories, 
            dim_info
            ).to(device)
        
        model.eval()
        model.load_state_dict(torch.load(model_file)['model'])
        _, _, _, _, test_prediction, test_target, attention_score = test(
                model=model,
                data=test_data,
                batch_size=args.batch_size,
                device=device,
                return_attention_score=True
                )

        test_prediction = np.concatenate(tuple(test_prediction), axis=0)
        test_target = np.concatenate(tuple(test_target), axis=0)

        save_data(
            'test_result.pkl',
            {
                'prediction': test_prediction,
                'target': test_target,
                'attention_score': attention_score
            },
            os.path.join(result_path, args.session_no)
        )
        
        test_prediction_class = np.argmax(test_prediction, axis=-1)

        test_acc = np.equal(test_target, test_prediction_class).sum() / test_target.shape[0]
        test_auroc = multiclass_score(test_target, test_prediction, 'auroc')
        test_auprc = multiclass_score(test_target, test_prediction, 'auprc')
        test_recall = multiclass_score(test_target, test_prediction_class, 'recall')
        
        session_no = args.session_no
        print('Test session: {}\tAccuracy: {:.3f}\tAUROC: {:.3f}\tRecall: {:.3f}\tAUPRC: {:.3f}'.format(session_no, test_acc, test_auroc, test_recall, test_auprc))

    else:
        ### define the path for results
        if args.session_no is None:
            session_no = obtain_session_no(result_path)
            load_model_flag = False
            print('Session no: {}'.format(session_no))

            result_path = result_path + str(session_no) + '/'
            os.makedirs(result_path)
        else:
            session_no = args.session_no
            if os.path.exists(os.path.join(result_path, str(session_no), 'best_valid_loss_model.pt')):
                load_model_flag = True
                print(f'Continue training from session {session_no}')
            else:
                os.makedirs(os.path.join(result_path, str(session_no)), exist_ok=True)
                load_model_flag = False
                print('Session no: {}'.format(session_no))
            
            result_path = result_path + str(session_no) + '/'
        
        ### save arguments
        with open(result_path+'args.txt', 'w') as f:
            json.dump(args.__dict__, f)

        #### temp ####
        #####
        train_data = load_data(data_path=data_path, data_type='train')
        valid_data = load_data(data_path=data_path, data_type='valid')

        if args.binary_classification:
            for split in (train_data, valid_data, test_data):
                binarise(split, args.target_type)

        if args.balanced:
            try:
                idx = joblib.load(os.path.join(data_path, 'balancing_idx.pkl'))
            except FileNotFoundError:
                idx = {'train': None, 'valid': None}
            train_data, train_balancing_idx = balance_data(train_data, idx=idx['train'], run=args.balance_run)
            valid_data, valid_balancing_idx = balance_data(valid_data, idx=idx['valid'], run=args.balance_run)
            if idx['train'] is None:
                joblib.dump({
                    'train': train_balancing_idx,
                    'valid': valid_balancing_idx
                    }, os.path.join(data_path, 'balancing_idx.pkl'))

        key_info, prs_info, categories, dim_info = prepare_data(
            args,
            data_path,
            train_data
        )

        model = trans_model(
            args,
            categories, 
            dim_info
            ).to(device)

        if load_model_flag:
            model.eval()
            previous_model =  torch.load(os.path.join(result_path, 'best_valid_loss_model.pt'))
        else:
            previous_model = None

        train(args,
              model,
              train_data,
              valid_data,
              test_data,
              device=device,
              result_path=result_path,
              previous_model=previous_model)
        
        print('Session no: {}'.format(session_no))

    print(args)


def resolve_data_path(args):
    """Locate the preprocessed dataset directory.

    Resolution order: ``--data_path`` > ``$CKD_DATA_ROOT``. Site-specific
    absolute paths are deliberately not hard-coded here; the SNUBH EHR extract
    is governed by an IRB data-use agreement and is never shipped with the code.
    """
    path = args.data_path or os.environ.get('CKD_DATA_ROOT')
    if not path:
        raise SystemExit(
            'No dataset directory configured. Pass --data_path <dir> or set '
            'CKD_DATA_ROOT to the directory holding the preprocessed '
            'reduced24_*.pkl / key_info.pkl files.')
    if not os.path.isdir(path):
        raise SystemExit(f'Dataset directory does not exist: {path}')
    return os.path.join(path, '')

if __name__ == '__main__':
    main()
