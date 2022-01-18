from torch.nn import functional as F
from models.meshed_memory_transformer.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.meshed_memory_transformer.transformer.attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask


class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, num_proposals=256, **kwargs): # TODO: d_in should be num_proposals * num_features
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in + 27, int(self.d_model)) # d_in + bbox_center (3) + corners (24)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(int(self.d_model))
        self.num_proposals = num_proposals

    def forward(self, data_dict, attention_weights=None):
        """[summary]

        Args:
            object_proposals ([type]): [batch_size, num_proposals, features]
            object_masks ([type]): [batch_size, num_proposals]
            attention_weights ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        B, N, _ = data_dict["bbox_feature"].shape
        
        object_proposals = torch.cat([data_dict["bbox_feature"], data_dict["center"], data_dict["bbox_corner"].view(B, N, -1)], -1).type(torch.float).to(data_dict["bbox_feature"].device)
        object_masks = data_dict["bbox_mask"]
        
        object_proposals = object_proposals.view(B * N, -1) # [batch_size * num_proposals, feature_size + 3]
        
        out = F.relu(self.fc(object_proposals)) #128 features and center coordinates (3) per object_proposal -> d_model features (default=512)
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.view(B, N, -1) # [batch_size, num_proposals, d_model]
        out[object_masks == 0] = 0
        data_dict["encoder_input"] = out
        return super(MemoryAugmentedEncoder, self).forward(out, attention_weights=attention_weights)
