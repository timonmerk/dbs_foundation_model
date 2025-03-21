import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(RotaryPositionalEncoding, self).__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb

class InputEmbedding(nn.Module):
    def __init__(self, in_dim, seq_len, d_model, num_cls_token, use_rotary_encoding=True, add_hour_emb=False):
        super(InputEmbedding, self).__init__()

        self.cls_tokens = nn.Parameter(torch.randn(num_cls_token, 1, d_model), requires_grad=True)  # classification tokens
        model_dim_ = d_model +1 if add_hour_emb else d_model
        if use_rotary_encoding:
            self.positional_encoding = RotaryPositionalEncoding(model_dim_)
        else:
            self.positional_encoding = nn.Parameter(torch.randn(seq_len + num_cls_token, model_dim_), requires_grad=True)  # learnable positional encoding
        self.use_rotary_encoding = use_rotary_encoding
        self.add_hour_emb = add_hour_emb
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_cls_token = num_cls_token
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
        )
        self.apply(_weights_init)

    def forward(self, data, hour=None):
        bat_size, seq_len, seg_len, ch_num = data.shape
        data = data.view(bat_size * ch_num, seq_len, seg_len)
        input_emb = self.proj(data)
        
        cls_tokens = self.cls_tokens.expand(-1, bat_size * ch_num, -1).transpose(0, 1)
        if self.add_hour_emb:
            input_emb = torch.cat((cls_tokens, input_emb), dim=1)
            hour = hour.repeat(ch_num)
            hour = hour.unsqueeze(1).unsqueeze(2) 
            
            hour = hour.expand(bat_size * ch_num, self.num_cls_token+self.seq_len, 1)
            input_emb = torch.cat((input_emb, hour), dim=2)

        else:
            input_emb = torch.cat((cls_tokens, input_emb), dim=1)
        
        if self.use_rotary_encoding:
            pos_enc = self.positional_encoding(torch.arange(seq_len + self.num_cls_token, device=input_emb.device).float())
        else:
            pos_enc = self.positional_encoding
        input_emb += pos_enc

        return input_emb

def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class TimeEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dim_feedforward,
                 seq_len, n_layer, nhead, num_cls_token,
                 use_rotary_encoding=True,
                 add_hour_emb=False):
        super(TimeEncoder, self).__init__()

        self.input_embedding = InputEmbedding(in_dim=in_dim, seq_len=seq_len, d_model=d_model,
                                              num_cls_token=num_cls_token,
                                              use_rotary_encoding=use_rotary_encoding,
                                              add_hour_emb=add_hour_emb)
        self.num_cls_token = num_cls_token

        d_model_ = d_model + 1 if add_hour_emb else d_model
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model_, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layer)

        self.apply(_weights_init)

    def forward(self, data, hour=None):
        input_emb = self.input_embedding(data, hour)
        trans_out = self.trans_enc(input_emb)
        cls_token_outputs = trans_out[:, :self.num_cls_token, :]  # extract the classification token outputs

        return cls_token_outputs, trans_out[:, self.num_cls_token:, :]
