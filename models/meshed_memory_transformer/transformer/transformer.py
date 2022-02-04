import torch
from torch import nn
import copy
from models.meshed_memory_transformer.containers import ModuleList
from ..captioning_model import CaptioningModel
from caption_module import select_target
from lib.config import CONF
from ..beam_search.iterative import IterativeGeneration
from ..beam_search.beam_search import BeamSearch


class Transformer(CaptioningModel):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = 3
        self.encoder = encoder
        self.decoder = decoder
        # self.register_state('enc_output', None)
        # self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data_dict, is_eval=False, *args):
        return self.train_forward(data_dict, *args)
    
    def train_forward(self, data_dict, *args):
        data_dict["lang_ids_model"] = data_dict["lang_ids"][:,0:-1]
        
        data_dict = self.get_best_object_proposal(data_dict) # get best object proposal for training
        
        enc_output, mask_enc = self.encoder(data_dict)

        data_dict = self.decoder(data_dict, enc_output, mask_enc)
        return data_dict

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def encode_for_search(self, data_dict):
        B, N, L, F =  data_dict["edge_feature"].shape
        data_dict["target_object_proposal"] = data_dict["bbox_feature"].view(B * N, F)
        data_dict["target_edge_feature"] = data_dict["edge_feature"].view(B * N, L, F)

        enc_output, mask_enc = self.encoder(data_dict)
        return enc_output, mask_enc

    def step(self, t, data_dict, mode='teacher_forcing', **kwargs):
        if mode == 'teacher_forcing':
            raise NotImplementedError

        data_dict = self.decoder(data_dict, data_dict["enc_output"], data_dict["enc_mask"])
        return data_dict
    
    def get_best_object_proposal(self, data_dict):
        target_ids, target_ious = select_target(data_dict)
        B, N, L, F = data_dict["edge_feature"].shape
        
        # select object features
        target_object_proposal = torch.gather(
            data_dict["bbox_feature"], 1, target_ids.view(B, 1, 1).repeat(1, 1, F)).squeeze(1) # batch_size, feature_size
        target_edge_feat = torch.gather(
            data_dict["edge_feature"], 1, target_ids.view(B, 1, 1, 1).repeat(1, 1, L, F)).squeeze(1)
        
        
        good_bbox_masks = target_ious > CONF.TRAIN.MIN_IOU_THRESHOLD # batch_size
        num_good_bboxes = good_bbox_masks.sum()
        
        data_dict["target_object_proposal"] = target_object_proposal
        data_dict["target_edge_feature"] = target_edge_feat
        data_dict["pred_ious"] =  target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()
        data_dict["good_bbox_masks"] = good_bbox_masks
        
        return data_dict
    
    def iterative(self, data_dict, max_len=32, eos_idx=3, is_eval=False):
        iterative = IterativeGeneration(self, max_len, eos_idx)
        return iterative.apply(data_dict, is_eval=is_eval)

    def beam_search(self, data_dict, max_len=32, eos_idx=3, is_eval=False, beam_size=5):
        beam_search = BeamSearch(self, max_len, eos_idx, beam_size)
        return beam_search.apply(data_dict, is_eval=is_eval)
