import copy

import numpy as np
import torch
import torch.nn as nn

from model.layers import fc
from model.performer_attention import SelfAttention


class meta_embeddings(nn.Module):
    def __init__(self, input_dims, embed_dim, pos_encode=False):
        super().__init__()

        num_importance_token = input_dims['num_importance_token']
        num_time_token = input_dims['num_time_token']
        num_obsn_token = input_dims['num_obsn_token']
        
        self.pos_encode = pos_encode
        self.embed_dim = embed_dim

        self.importance_embed = nn.Embedding(
            num_embeddings=num_importance_token,
            embedding_dim=embed_dim
        )
        self.time_embed = nn.Embedding( ## including <CLS> tokens
            num_embeddings=num_time_token,
            embedding_dim=embed_dim

        )
        self.obsn_embed = nn.Embedding(
            num_embeddings=num_obsn_token,
            embedding_dim=embed_dim
        )

        if self.pos_encode:
            self.sinusoid_table = self.positional_encode()

        # value * importance + time + obsn_mask 
        # mask on value

    def positional_encode(self):
        def cal_angle(position, i_hidn):
            return position / np.power(10000, 2*(i_hidn // 2) / d_hidn)
        def get_position_angle_vec(position, d_hidn):
            return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

        d_hidn = self.embed_dim
        sinusoid_table = np.array([get_position_angle_vec(i_seq, d_hidn) for i_seq in range(7300)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.tensor(sinusoid_table).to(torch.float32)

        # import pdb; pdb.set_trace()
    
    def forward(self, importance, times, special_token, obsn_mask=None):
        embedded_importance = self.importance_embed(importance)
        if self.pos_encode:
            embedded_times = self.sinusoid_table.to(times.device)[times]
        else:
            embedded_times = self.time_embed(times)
        embedded_special_token = self.time_embed(special_token)
        if obsn_mask is not None:
            embedded_obsn_mask = self.obsn_embed(obsn_mask)
        else:
            embedded_obsn_mask = None
        return embedded_importance, embedded_times, embedded_special_token, embedded_obsn_mask


class embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embed = nn.DataParallel(fc(
                input_dim=input_dim,
                output_dim=input_dim*embed_dim,
                out_act='ReLU'
                ))

    def vime_mask(self, m, x):
        b, l, f = x.size()
        reshaped_x = x.reshape([b*l, f])
        no, feature = reshaped_x.shape
        m = m.reshape([b*l, f])
        
        x_bar = torch.zeros_like(reshaped_x)
        for i in range(feature):
            idx = np.random.permutation(no)
            x_bar[:, i] = reshaped_x[idx, i]
        
        ### corrupt samples
        x_tilde = reshaped_x * (1-m) + x_bar * m  
        x_tilde = x_tilde.reshape([b, l, f])
        #### define new mask matrix
        m_new = 1 * (x != x_tilde)
        return m_new, x_tilde

    def mask_data(self, p_m, x):
        mask = torch.zeros(x.size()) + p_m
        mask = torch.bernoulli(mask).to(x.device)
        mask_label, x_tilde = self.vime_mask(mask, x)
        return x_tilde.float().to(x.device), mask_label.bool().to(x.device)

    def forward(self, inputs, meta_embed, importance, times, obsn_mask, mode='train', p_m=0):
        batch_size, seq_len, num_feature = inputs.size()
        embedded_importance, embedded_times, embedded_special_token, embedded_obsn_mask = meta_embed(importance, times, inputs[:, :, 0].long(), obsn_mask)
        
        if mode == 'train':
            masked_inputs, mask_label = self.mask_data(p_m, inputs[:, :, 1:])
            embedded_inputs = self.embed(masked_inputs)
            mlm_targets = inputs[:, :, 1:][mask_label]
        elif mode == 'eval':
            embedded_inputs = self.embed(inputs[:, :, 1:])
            mask_label = None
            mlm_targets = None
        
        embedded_inputs = embedded_inputs.reshape(batch_size, seq_len, num_feature-1, -1)

        embedded_inputs = embedded_inputs * embedded_importance.repeat([batch_size, seq_len, 1, 1])
        embedded_inputs = embedded_inputs + embedded_times.unsqueeze(-2).repeat_interleave(num_feature-1, dim=-2)
        embedded_inputs = embedded_inputs + embedded_obsn_mask[:, :, 1:, :]
        embedded_inputs = torch.cat((embedded_special_token.unsqueeze(-2), embedded_inputs), dim=-2)

        return embedded_inputs, (mask_label, mlm_targets)


class prs_embeddings(embeddings):
    def __init__(self, input_dim, embed_dim, num_medicine_embeddings, num_unit_embeddings):
        super().__init__(input_dim, embed_dim)

        self.medicine_embed = nn.Embedding(
            num_embeddings=num_medicine_embeddings,
            embedding_dim=embed_dim
            )
        self.unit_embed = nn.Embedding(
            num_embeddings=num_unit_embeddings,
            embedding_dim=embed_dim
            )
    
    def vime_mask(self, m, x):
        b, l, f, d = x.shape
        reshaped_x = x.reshape([b*l, f, d])
        no, feature, dim = reshaped_x.shape
        m = m.reshape([b*l, f, d])
        
        x_bar = torch.zeros_like(reshaped_x)
        for i in range(feature):
            idx = np.random.permutation(no)
            x_bar[:, i] = reshaped_x[idx, i]
        
        ### corrupt samples
        x_tilde = reshaped_x * (1-m) + x_bar * m  
        x_tilde = x_tilde.reshape([b, l, f, d])
        #### define new mask matrix
        m_new = 1 * (x != x_tilde)
        return m_new, x_tilde

    def forward(self, medicine_inputs, dosage_inputs, unit_inputs, meta_embed, importance, times, mode='train', p_m=0):
        batch_size, seq_len, num_feature = medicine_inputs.shape

        embedded_importance, embedded_times, embedded_special_token, _ = meta_embed(importance, times, medicine_inputs[:, :, 0])

        embedded_medicine_inputs = self.medicine_embed(medicine_inputs[:, :, 1:])
        if mode == 'train':
            masked_medicine_inputs, mask_label = self.mask_data(p_m, embedded_medicine_inputs)
            mask_label = mask_label[:, :, :, 0]
            mlm_targets = medicine_inputs[:, :, 1:][mask_label]
            embedded_medicine_inputs = masked_medicine_inputs * embedded_importance
        elif mode == 'eval':
            embedded_medicine_inputs = embedded_medicine_inputs * embedded_importance
            mask_label = None
            mlm_targets = None
        
        embedded_unit_inputs = self.unit_embed(unit_inputs[:, :, 1:])
        embedded_dosage_inputs = self.embed(dosage_inputs[:, :, 1:])
        embedded_dosage_inputs = embedded_dosage_inputs.reshape(batch_size, seq_len, num_feature-1, -1)
    
        embedded_dosage_inputs = embedded_dosage_inputs * embedded_unit_inputs

        embedded_prs_inputs = torch.cat((embedded_medicine_inputs, embedded_dosage_inputs), axis=-1)
        
        embedded_prs_inputs += embedded_times.unsqueeze(-2).repeat_interleave(num_feature-1, dim=-2).repeat_interleave(2, dim=-1)
        embedded_prs_inputs = torch.cat((embedded_special_token.repeat_interleave(2, dim=-1).unsqueeze(-2), embedded_prs_inputs), dim=2)
        
        return embedded_prs_inputs, (mask_label, mlm_targets)


class tabular_transformer_encoder_layer(nn.Module):
    def __init__(
        self, 
        time_d_model,
        feature_d_model,
        embed_dim,
        nhead,
        dim_feedforward=2048, 
        dropout=0.1, 
        activation=nn.functional.relu,
        layer_norm_eps=1e-5,

        ):

        super().__init__()
        
        self.embed_dim = embed_dim
        self.time_attn = SelfAttention(
            dim=time_d_model,
            heads=nhead
            )
        self.feature_attn = SelfAttention(
            dim=feature_d_model,
            heads=nhead
            )

        self.linear1 = nn.Linear(time_d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, time_d_model)
        
        self.norm_ta = nn.LayerNorm(time_d_model, eps=layer_norm_eps)
        self.norm_fa = nn.LayerNorm(feature_d_model, eps=layer_norm_eps)
        self.norm_ff = nn.LayerNorm(time_d_model, eps=layer_norm_eps)

        self.dropout_ta = nn.Dropout(dropout)
        self.dropout_fa = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.activation = activation
    
    def _ta_block(self, x):
        x, _ = self.time_attn(x, x, x)
        x = self.dropout_ta(x)
        return x

    def _fa_block(self, x, return_attn_score):
        seq_len, batch_size, _ = x.size()
        x = x.view(seq_len, batch_size, -1, self.embed_dim)
        feature_dim = x.size(2)

        x = x.permute(2, 1, 0, 3) # feature_dim batch_size seq_len embed_dim
        x = x.reshape(feature_dim, batch_size, -1)
        x, feature_attn_score= self.feature_attn(x, x, x, return_attn_score=return_attn_score)
        x = self.dropout_fa(x)

        x = x.reshape(feature_dim, batch_size, seq_len, self.embed_dim)
        x = x.permute(2, 1, 0, 3) 
        return x.reshape(seq_len, batch_size, -1), feature_attn_score

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout_ff(x)

    def forward(self, src, return_attn_score):
        x = src
        tx = self._ta_block(x)
        fx, feature_attn_score = self._fa_block(x, return_attn_score)
        x = self.norm_ta(x + tx + fx)
        x = self.norm_ff(x + self._ff_block(x))
        return x, feature_attn_score

class transformer_encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    def forward(self, src, return_attn_score):
        output = src
        output, feature_attention = self.layers[0](output, return_attn_score)
        for mod in self.layers[1:]:
            output, _ = mod(output, False)
        if self.norm is not None:
            output = self.norm(output)

        return output, feature_attention


class uni_trans(nn.Module):
    def __init__(self, input_dims, dropout, trans_config, mlm_config):
        super(uni_trans, self).__init__()
        
        self.homo_transformer = self.Transformer(
            d_models=input_dims,
            nhead=trans_config['nhead'],
            num_encoder_layers=trans_config['num_layers'],
            dim_feedforward=trans_config['d_ff']
            )
        self.mlm_predictor = fc(
            input_dim=mlm_config['input_dim'],
            output_dim=mlm_config['output_dim'],
            hidden_dim=mlm_config['hidden_dim'],
            dropout=dropout
            )
        
    def Transformer(self, d_models, nhead, num_encoder_layers, dim_feedforward):
        encoder_layer = tabular_transformer_encoder_layer(
                            time_d_model=d_models[0],
                            feature_d_model=d_models[1],
                            embed_dim=d_models[2],
                            nhead=nhead, 
                            dim_feedforward=dim_feedforward, 
                            dropout=0.1, 
                            activation=nn.functional.relu,
                            layer_norm_eps=1e-5,
                            )
        return transformer_encoder(
                            encoder_layer,
                            num_layers=num_encoder_layers,
                            norm=nn.LayerNorm(d_models[0])
                            )

    def forward(self, inputs, mode, mask_label=None, return_attn_score=False):
        batch_size, seq_len, feature_dim, embed_dim = inputs.size()
        z, attn_score = self.homo_transformer(inputs.permute(1,0,2,3).reshape(seq_len, batch_size, feature_dim*embed_dim), return_attn_score)
        z = z.permute(1,0,2).reshape(batch_size, seq_len, feature_dim, embed_dim)
        if mode == 'train':
            z_masked = z[:, :, 1:, :][mask_label]
            reconstructed = self.mlm_predictor(z_masked)
        elif mode == 'eval':
            reconstructed = None
        
        if not return_attn_score: attn_score = None
        return z, reconstructed, attn_score
 

class trans_model(nn.Module):
    def __init__(self, args, categories, dim_info):
        super().__init__()
        self.args = args

        layers = {}
        layers['embed_meta'] = meta_embeddings(
            input_dims={'num_importance_token': dim_info['importance_embed'],
                        'num_time_token': dim_info['time_embed'],
                        'num_obsn_token': dim_info['obsn_embed']}, 
            embed_dim=args.embed_dim,
            pos_encode=args.pos_encode
            )
        for category in categories:
            layers['embed_' + category] = embeddings(
                input_dim=dim_info[category + '_input_dim'],
                embed_dim=args.embed_dim
            )
            layers['trans_' + category] = nn.DataParallel(uni_trans(
                input_dims=(
                    (dim_info[category + '_input_dim'] + 1) * args.embed_dim,
                    dim_info[category + '_len'] * args.embed_dim,
                    args.embed_dim
                    ),
                dropout=args.dropout,
                trans_config={'nhead': args.num_head,
                              'num_layers': args.num_layers,
                              'd_ff': args.d_ff},
                mlm_config={'input_dim': args.embed_dim,
                            'hidden_dim': args.mlm_hidden_dim,
                            'output_dim': 1}
            ))
        
        layers['embed_prs'] = prs_embeddings(
            input_dim=dim_info['prs_input_dim'], 
            embed_dim=args.embed_dim, 
            num_medicine_embeddings=dim_info['code_embed'], 
            num_unit_embeddings=dim_info['unit_embed']
            )
        layers['trans_prs'] = nn.DataParallel(uni_trans(
            input_dims=(
                (dim_info['prs_input_dim'] + 1) * args.embed_dim * 2,
                dim_info['prs_len'] * args.embed_dim * 2,
                args.embed_dim * 2
                ),
            dropout=args.dropout,
            trans_config={'nhead': args.num_head,
                          'num_layers': args.num_layers,
                          'd_ff': args.d_ff},
            mlm_config={'input_dim': args.embed_dim*2,
                        'hidden_dim': args.mlm_hidden_dim,
                        'output_dim': dim_info['code_embed']}
            ))
        self.layers = nn.ModuleDict(layers)
        
        self.condition_embed = nn.Embedding(
            num_embeddings=dim_info['num_condition'],
            embedding_dim=dim_info['input_dim']
            )
        self.regressor = nn.DataParallel(fc(
            input_dim=dim_info['input_dim'],
            output_dim=1,
            hidden_dim=args.cls_hidden_dim,
            dropout=args.dropout,
            out_act='ReLU'
            ))
        self.predictor = nn.DataParallel(fc(
            input_dim=dim_info['input_dim'],
            output_dim=dim_info['output_dim'],
            hidden_dim=args.cls_hidden_dim,
            dropout=args.dropout,
            out_act='softmax'
            ))

    def forward(self, inputs, importance, times, condition, obsn_mask, mode='train', return_attn_score=False):
        zs = {}
        mlms = {}
        attn_scores = {}
        for key in inputs.keys():
            if key in ['code', 'dosage', 'unit']:
                continue
            embedded_data, mlm = self.layers['embed_' + key](
                inputs[key],
                self.layers['embed_meta'],
                importance['lab'][key],
                times[key],
                obsn_mask[key],
                p_m=self.args.p_m, 
                mode=mode
                )
            z_key, reconstructed, attn_score = self.layers['trans_' + key](
                embedded_data,
                mode=mode,
                mask_label=mlm[0],
                return_attn_score=return_attn_score
                )
            mlms[key] = (reconstructed, mlm[1])
            zs[key] = z_key[:, :, 0].sum(1)
            attn_scores[key] = attn_score
        embedded_data, mlm = self.layers['embed_prs'](
            inputs['code'], 
            inputs['dosage'],
            inputs['unit'],
            self.layers['embed_meta'],
            importance['drug'],
            times['prs'],
            p_m=self.args.p_m,
            mode=mode
        )
        z_prs, reconstructed, attn_score = self.layers['trans_prs'](
            embedded_data,
            mode=mode,
            mask_label=mlm[0],
            return_attn_score=return_attn_score
            )
        mlms['prs'] = (reconstructed, mlm[1])
        zs['prs'] = z_prs[:, :, 0].sum(1)
        attn_scores['prs'] = attn_score
        z = torch.cat([*zs.values()], axis=-1)
        z += self.condition_embed(condition)
        predicted = self.predictor(z)
        regressed = self.regressor(z)

        return predicted, regressed, mlms, attn_scores