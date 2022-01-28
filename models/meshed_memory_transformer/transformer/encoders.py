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
    def __init__(self, N, padding_idx, no_encoder, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.padding_idx = padding_idx
        if not no_encoder:
            self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                    identity_map_reordering=identity_map_reordering,
                                                    attention_module=attention_module,
                                                    attention_module_kwargs=attention_module_kwargs)
                                        for _ in range(N)])

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
    def __init__(self, N, padding_idx, d_in=2048, num_proposals=256, no_encoder=False, **kwargs): # TODO: d_in should be num_proposals * num_features
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, no_encoder, **kwargs)
        self.no_encoder = no_encoder
        self.fc = nn.Linear(d_in, self.d_model - 1) # 128 -> 191
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model - 1)
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
        B, N, _ = data_dict["target_edge_feature"].shape
        
        encoder_input = torch.cat([
            data_dict["target_object_proposal"].unsqueeze(1),
            data_dict["target_edge_feature"]
        ], 1).type(torch.float).to(data_dict["bbox_feature"].device)

        B, N, _ = encoder_input.shape

        #object_masks = data_dict["bbox_mask"] TODO: Object masks necessary with graph?
        
        object_proposals = encoder_input.view(B * N, -1) # [batch_size * (n_locals+1), feature_size]

        target_indicator = torch.tensor([1] + data_dict["target_edge_feature"].shape[1]*[0], device=data_dict["bbox_feature"].device).unsqueeze(0).unsqueeze(-1).repeat_interleave(B, dim=0)
        out = F.relu(self.fc(object_proposals)) #128 features per object_proposal -> d_model features (default=512)
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.view(B, N, -1) # [batch_size, n_locals+1, d_model-1]
        out = torch.cat([out, target_indicator], dim=-1).to(out.device)
        data_dict["encoder_input"] = out
        
        if not self.no_encoder:
            return super(MemoryAugmentedEncoder, self).forward(out, attention_weights=attention_weights)
        else:
            attention_mask = (torch.sum(out, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
            return out.unsqueeze(1), attention_mask