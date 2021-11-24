import torch
from torch import nn
import copy
from models.meshed_memory_transformer.containers import ModuleList
from ..captioning_model import CaptioningModel
from caption_module import select_target
from lib.config import CONF


class Transformer(CaptioningModel):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = -1 # TODO: remove
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data_dict, is_eval=False, *args):
        """[summary]

        Args:
            data_dict

        Returns:
            [type]: [description]
        """
        
        enc_output, mask_enc = self.encoder(data_dict)
        
        if is_eval:
            return self.eval_forward(data_dict, enc_output, mask_enc, *args)
        else:
            return self.train_forward(data_dict, enc_output, mask_enc, *args)
    
    def train_forward(self, data_dict, enc_output, mask_enc, *args):
        B, N = data_dict["bbox_feature"].shape[0], data_dict["bbox_feature"].shape[1]
        data_dict["lang_ids_model"] = data_dict["lang_ids"][:,1:]
        
        data_dict = self._get_best_object_proposal(data_dict) # [batch_size, object_proposal_features]
        
        data_dict = self.decoder(data_dict, enc_output, mask_enc)
        return data_dict
        
    def eval_forward(self, data_dict, enc_output, mask_enc, *args):
        B, N = data_dict["bbox_feature"].shape[0], data_dict["bbox_feature"].shape[1]
        
        data_dict, enc_output, mask_enc = self._batchify(data_dict, enc_output, mask_enc)
        data_dict = self.decoder(data_dict, enc_output, mask_enc)
        data_dict = self._unbatchify(data_dict, B, N)
        return data_dict

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)
    
    def _get_best_object_proposal(self, data_dict):
        target_ids, target_ious = select_target(data_dict)
        B, N, F = data_dict["bbox_feature"].shape[0], data_dict["bbox_feature"].shape[1],  data_dict["bbox_feature"].shape[2]
        # select object features
        target_object_proposal = torch.gather(
            data_dict["bbox_feature"], 1, target_ids.view(B, 1, 1).repeat(1, 1, F)).squeeze(1) # batch_size, feature_size
        
        good_bbox_masks = target_ious > CONF.TRAIN.MIN_IOU_THRESHOLD # batch_size
        num_good_bboxes = good_bbox_masks.sum()
        
        data_dict["target_object_proposal"] = target_object_proposal
        data_dict["pred_ious"] =  target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()
        data_dict["good_bbox_masks"] = good_bbox_masks
        
        return data_dict
    
    def _batchify(self, data_dict, enc_output, mask_enc):
        B, N = data_dict["bbox_feature"].shape[0], data_dict["bbox_feature"].shape[1]
        
        data_dict["lang_ids_model"] = torch.zeros([B * N, data_dict["lang_ids"].shape[1] - 1]).cuda().int()
        data_dict["target_object_proposal"] = data_dict["bbox_feature"].reshape(B*N, -1)
        enc_output = enc_output.repeat_interleave(N, dim = 0)
        mask_enc = mask_enc.repeat_interleave(N, dim = 0)
        
        return data_dict, enc_output, mask_enc
    
    def _unbatchify(self, data_dict, B, N):
        data_dict["lang_cap"] = data_dict["lang_cap"].reshape(B,N,data_dict["lang_cap"].shape[-2], data_dict["lang_cap"].shape[-1])
        return data_dict


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
