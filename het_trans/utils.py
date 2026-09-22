import math
import os

import joblib
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(
            map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]

class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / \
                    view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * \
                    view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (
                    exp_avg_sq.sqrt() /
                    math.sqrt(bias_correction2)).add_(
                    group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(
                        p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(
                        1 -
                        group['lr'] *
                        group['weight_decay'] *
                        wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss

def obtain_session_no(result_folder):
    """ To obtain the numeric ID of the current session
    """
    session_list = os.listdir(result_folder)
    if len(session_list) == 0:
        last_session = -1
    else:
        session_list = list(map(int, session_list))
        last_session = max(session_list)
    return last_session + 1

        


def iterate_minibatches(category, prs, conditions, targets, batch_size, shuffle, device):
    num_data = targets.shape[0]
    if shuffle:
        indices = np.arange(num_data)
        np.random.shuffle(indices)
    for start_idx in range(0, num_data, batch_size):
        if shuffle:
            if start_idx + batch_size >= num_data:
                excerpt = indices[start_idx:]
            else:
                excerpt = indices[start_idx:start_idx + batch_size]
        else:
            if start_idx + batch_size >= num_data:
                excerpt = slice(start_idx, num_data)
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
    
        batch = {}
        times = {}
        obsn_mask = {}
        tokens = {}
        tokens['category'] = {}

        for key in category['inputs'].keys():
            batch[key] = torch.from_numpy(category['inputs'][key][excerpt]).float().to(device)
            times[key] = torch.from_numpy(category['time'][key][excerpt]).long().to(device)
            obsn_mask[key] = torch.from_numpy(category['obsn_mask'][key][excerpt]).long().to(device)
            tokens['category'][key] = torch.tensor(category['token'][key]).long().to(device)

        batch['code'] = torch.from_numpy(prs['code'][excerpt]).long().to(device)
        batch['dosage'] = torch.from_numpy(prs['dosage'][excerpt]).float().to(device)
        batch['unit'] = torch.from_numpy(prs['unit'][excerpt]).long().to(device)
        times['prs'] = torch.from_numpy(prs['time'][excerpt]).long().to(device)
        tokens['prs'] = torch.from_numpy(prs['token'][excerpt]).long().to(device)

        yield batch, {
            'eGFR_targets': torch.from_numpy(targets[excerpt, -2].astype(float)).float().to(device), 
            'CKD_targets': torch.from_numpy(targets[excerpt, -1].astype(int)).long().to(device),
            'condition': torch.from_numpy(conditions[excerpt, 1].astype(int)).long().to(device)
            }, times, obsn_mask, {
            'lab': tokens['category'],
            'drug': tokens['prs']
            } 

def save_data(filename, data, path='./'):
    os.makedirs(path, exist_ok=True)
    joblib.dump(data, os.path.join(path, filename))


    

def Bernoulli_mask(x):
    b, l, d = x.shape
    u = torch.empty(b, l, d).uniform_(0,1)
    mask = torch.bernoulli(u).float().to(x.device)
    return x * mask


def append_dict(org, add):
    if len(org.keys()) == 0:
        for k, v in add.items():
            if type(v) == torch.Tensor:
                org[k] = v.data.cpu().numpy()
            elif type(v) == dict:
                org[k] = append_dict({}, v)
            else:
                raise ValueError(f'Values of {k} has an unadaptable type {type(v)}.')
    else:
        assert list(org.keys()) == list(add.keys())
        for k, v in add.items():
            if type(org[k]) == np.ndarray:
                org[k] = np.concatenate((org[k], v.data.cpu().numpy()), axis=0)
            elif type(org[k]) == list:
                org[k] += list(v.data.cpu().numpy())
            elif type(org[k]) == dict:
                org[k] = append_dict(org[k], add[k])
            else:
                raise ValueError(f'Values of {k} has an unadaptable type {type(v)}.')
    return org
