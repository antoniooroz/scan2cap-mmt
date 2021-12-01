import torch
import utils_meshed_memory_transformer as utils
import time


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

    def apply(self, data_dict, out_size=1, is_eval=False, **kwargs):
        # Bathified
        # data_dict = self._batchify(data_dict) # num_proposals are also seen as batches
        # lang_ids_model = torch.ones([data_dict["target_object_proposal"].shape[0], 1]).to(data_dict["target_object_proposal"].device).int() * 2
        # data_dict["lang_ids_model"] = lang_ids_model
        # lang_ids_model = torch.ones([data_dict["target_object_proposal"].shape[0], 1], dtype=int).to(data_dict["target_object_proposal"].device) * 2
        # data_dict["lang_ids_model"] = lang_ids_model
        if is_eval:
            return self.apply_eval(data_dict, out_size, **kwargs)
        else:
            return self.apply_train(data_dict, out_size, **kwargs)
    
    def apply_train(self, data_dict, out_size=1, **kwargs):
        self.b_s = utils.get_batch_size(data_dict["bbox_feature"])
        self.device = utils.get_device(data_dict["bbox_feature"])
        
        enc_output, enc_mask = self.model.encode_for_beam_search(data_dict)
        # enc_output = enc_output.repeat_interleave(data_dict["bbox_feature"].shape[1], dim=0)
        # enc_mask = enc_mask.repeat_interleave(data_dict["bbox_feature"].shape[1], dim=0)
        data_dict = self.model.get_best_object_proposal(data_dict)
        device = data_dict["bbox_feature"].device

        lang_ids_model = torch.ones([self.b_s, 1]).to(device).int() * 2
        data_dict["lang_ids_model"] = lang_ids_model
        
        #with self.model.statefulness(self.b_s):
        data_dict["enc_output"] = enc_output
        data_dict["enc_mask"] = enc_mask
        for t in range(self.max_len):
            data_dict = self.iter(t, data_dict, **kwargs)
                
        return data_dict
    
    def apply_eval(self, data_dict, out_size=1, **kwargs):
        self.b_s = utils.get_batch_size(data_dict["bbox_feature"])
        self.device = utils.get_device(data_dict["bbox_feature"])
        
        enc_output, enc_mask = self.model.encode_for_beam_search(data_dict)
        # enc_output = enc_output.repeat_interleave(data_dict["bbox_feature"].shape[1], dim=0)
        # enc_mask = enc_mask.repeat_interleave(data_dict["bbox_feature"].shape[1], dim=0)
        num_proposals = data_dict["bbox_feature"].shape[1]
        device = data_dict["bbox_feature"].device
        
        data_dict["target_object_proposal"] = data_dict["bbox_feature"].reshape(self.b_s * num_proposals, -1)
        lang_ids_model = torch.ones([self.b_s * num_proposals, 1]).to(device).int() * 2
        data_dict["lang_ids_model"] = lang_ids_model
        #with self.model.statefulness(num_proposals):
        data_dict["enc_output"] = enc_output.repeat_interleave(num_proposals, dim=0)
        data_dict["enc_mask"] = enc_mask.repeat_interleave(num_proposals, dim=0)
        
        for t in range(self.max_len):
            data_dict = self.iter(t, data_dict, **kwargs)
       
        data_dict["lang_cap"] = data_dict["lang_cap"].reshape(self.b_s, num_proposals, data_dict["lang_cap"].shape[-2], data_dict["lang_cap"].shape[-1])
        return data_dict
    
    def apply_eval_old(self, data_dict, out_size=1, **kwargs):
        self.b_s = utils.get_batch_size(data_dict["bbox_feature"])
        self.device = utils.get_device(data_dict["bbox_feature"])
        
        enc_output, enc_mask = self.model.encode_for_beam_search(data_dict)
        # enc_output = enc_output.repeat_interleave(data_dict["bbox_feature"].shape[1], dim=0)
        # enc_mask = enc_mask.repeat_interleave(data_dict["bbox_feature"].shape[1], dim=0)
        lang_caps = []
        num_proposals = data_dict["bbox_feature"].shape[1]
        device = data_dict["bbox_feature"].device
        
        for batch_idx in range(data_dict["bbox_feature"].shape[0]):
            data_dict["target_object_proposal"] = data_dict["bbox_feature"][batch_idx, :]
            lang_ids_model = torch.ones([num_proposals, 1]).to(device).int() * 2
            data_dict["lang_ids_model"] = lang_ids_model
            #with self.model.statefulness(num_proposals):
            data_dict["enc_output"] = enc_output[batch_idx].unsqueeze(0).repeat_interleave(num_proposals, dim=0)
            data_dict["enc_mask"] = enc_mask[batch_idx].unsqueeze(0).repeat_interleave(num_proposals, dim=0)
            start_time = time.time()
            for t in range(self.max_len):
                data_dict = self.iter(t, data_dict, **kwargs)
            print("all iter: %s" % (time.time() - start_time))
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
