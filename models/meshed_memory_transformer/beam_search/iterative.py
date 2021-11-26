import torch
import utils_meshed_memory_transformer as utils


class IterativeGeneration(object):
    def __init__(self, model, max_len: int, eos_idx: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def apply(self, data_dict, out_size=1, **kwargs):
        # Bathified
        # data_dict = self._batchify(data_dict) # num_proposals are also seen as batches
        # lang_ids_model = torch.ones([data_dict["target_object_proposal"].shape[0], 1]).to(data_dict["target_object_proposal"].device).int() * 2
        # data_dict["lang_ids_model"] = lang_ids_model
        # lang_ids_model = torch.ones([data_dict["target_object_proposal"].shape[0], 1], dtype=int).to(data_dict["target_object_proposal"].device) * 2
        # data_dict["lang_ids_model"] = lang_ids_model
        
        self.b_s = utils.get_batch_size(data_dict["bbox_feature"])
        self.device = utils.get_device(data_dict["bbox_feature"])
        
        enc_output, enc_mask = self.model.encode_for_beam_search(data_dict)
        # enc_output = enc_output.repeat_interleave(data_dict["bbox_feature"].shape[1], dim=0)
        # enc_mask = enc_mask.repeat_interleave(data_dict["bbox_feature"].shape[1], dim=0)
        lang_caps = []

        for batch_idx in range(data_dict["bbox_feature"].shape[0]):
            data_dict["target_object_proposal"] = data_dict["bbox_feature"][batch_idx, :]
            # with self.model.statefulness(self.b_s):
            lang_ids_model = torch.ones([data_dict["target_object_proposal"].shape[0], 1]).to(data_dict["target_object_proposal"].device).int() * 2
            data_dict["lang_ids_model"] = lang_ids_model
            self.model.enc_output = enc_output[batch_idx].unsqueeze(0).repeat_interleave(data_dict["target_object_proposal"].shape[0], dim=0)
            self.model.enc_mask = enc_mask[batch_idx].unsqueeze(0).repeat_interleave(data_dict["target_object_proposal"].shape[0], dim=0)
            for t in range(self.max_len):
                data_dict = self.iter(t, data_dict, **kwargs)
            lang_caps.append(data_dict["lang_cap"].unsqueeze(0))
                
        data_dict["lang_cap"] = torch.cat(lang_caps, dim=0)
        return data_dict

    def iter(self, t: int, data_dict, **kwargs):
        data_dict = self.model.step(t, data_dict, mode='feedback', **kwargs)
        words = data_dict["lang_cap"].argmax(-1)
        data_dict["lang_ids_model"] = torch.cat([torch.ones([words.shape[0],1], dtype=int).to(words.device)*2, words], dim=1)
        return data_dict
        
    def _batchify(self, data_dict):
        B, N = data_dict["bbox_feature"].shape[0], data_dict["bbox_feature"].shape[1]
        
        data_dict["lang_ids_model"] = torch.zeros([B * N, data_dict["lang_ids"].shape[1] - 1]).cuda().int()
        data_dict["target_object_proposal"] = data_dict["bbox_feature"].reshape(B*N, -1)
        
        return data_dict
    
    def _unbatchify(self, data_dict, B, N):
        data_dict["lang_cap"] = data_dict["lang_cap"].reshape(B,N,data_dict["lang_cap"].shape[-2], data_dict["lang_cap"].shape[-1])
        return data_dict
