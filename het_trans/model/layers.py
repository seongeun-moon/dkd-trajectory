import torch.nn as nn

class fc(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[], dropout=0, out_act=None):
        super(fc, self).__init__()

        layers = []    
        if len(hidden_dim) > 0:
            for h_dim in hidden_dim:
                layers += [nn.Linear(input_dim, h_dim)]
                layers += [nn.BatchNorm1d(h_dim)]
                layers += [nn.ReLU()]
                layers += [nn.Dropout(dropout)]
                input_dim = h_dim
            layers += [nn.Linear(h_dim, output_dim)]
        else:
            layers += [nn.Linear(input_dim, output_dim)]
        if out_act:
            layers += [self.obtain_act_function(out_act)]
        self.layers = nn.Sequential(*layers)

    def obtain_act_function(self, act):
        if act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'ReLU':
            return nn.ReLU()
        elif act == 'LeakyReLU':
            return nn.LeakyReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'softmax':
            return nn.Softmax(dim=-1)

    def forward(self, x):
        return self.layers(x)

class embedding_layer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(embedding_layer, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim
        )

    def forward(self, x):
        return self.embed(x)