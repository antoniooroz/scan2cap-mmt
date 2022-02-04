import torch
import utils_meshed_memory_transformer as utils
import time


class IterativeGeneration(object):
    def __init__(self, model, max_len: int, eos_idx: int):
        self.model = model
        self.max_len = max_len
        self.b_s = None
        self.device = None

    def apply(self, data_dict, out_size=1, is_eval=False, **kwargs):
        if is_eval:
            return self.apply_eval(data_dict, out_size, **kwargs)
        else:
            return self.apply_train(data_dict, out_size, **kwargs)
        
    def apply_eval(self, data_dict, out_size=1, **kwargs):
        data_dict["enc_output"], data_dict["enc_mask"] = self.model.encode_for_search(data_dict) # Prepare decoder inputs
        
        self.b_s = utils.get_batch_size(data_dict["bbox_feature"])
        self.device = utils.get_device(data_dict["encoder_input"])
        
        num_proposals = data_dict["bbox_feature"].shape[1]
        device = data_dict["encoder_input"].device
        
        # Sentence starts
        lang_ids_model = torch.ones([self.b_s * num_proposals, 1]).to(device).int() * 2
        data_dict["lang_ids_model"] = lang_ids_model
        
        # Generate sentences
        with self.model.statefulness(data_dict["target_object_proposal"].shape[0]):
            for t in range(self.max_len):
                data_dict = self.iter(t, data_dict, **kwargs)
       
        # Prepare for eval
        data_dict["lang_cap"] = data_dict["lang_cap_iterative"].view(self.b_s, num_proposals, data_dict["lang_cap_iterative"].shape[-2], data_dict["lang_cap_iterative"].shape[-1])
        data_dict["lang_pred_sentences"] = data_dict["lang_cap"].argmax(-1)
        
        return data_dict
                
    def iter(self, t: int, data_dict, **kwargs):
        # Call model
        data_dict = self.model.step(t, data_dict, mode='feedback', **kwargs)
        
        # Get predicted words and add them for next step to lang_ids_model
        words = data_dict["lang_cap"].argmax(-1)
        data_dict["lang_ids_model"] = words[:,-1].unsqueeze(1)
        
        # Add to lang_cap_iterative
        if "lang_cap_iterative" in data_dict.keys():
            data_dict["lang_cap_iterative"] = torch.cat([data_dict["lang_cap_iterative"], data_dict["lang_cap"][:,-1,:].unsqueeze(1)], dim=1)
        else:
            data_dict["lang_cap_iterative"] = data_dict["lang_cap"][:,-1,:].unsqueeze(1)
        
        return data_dict

    def apply_train(self, data_dict, out_size=1, **kwargs):
        self.b_s = utils.get_batch_size(data_dict["bbox_feature"])
        self.device = utils.get_device(data_dict["bbox_feature"])
        
        data_dict = self.model.get_best_object_proposal(data_dict) # get best object proposal for training
        
        data_dict["enc_output"], data_dict["enc_mask"] = self.model.encoder(data_dict) # Prepare decoder inputs
        
        # Start of sentences
        lang_ids_model = torch.ones([self.b_s, 1]).to(self.device).int() * 2
        data_dict["lang_ids_model"] = lang_ids_model
        
        with self.model.statefulness(data_dict["target_object_proposal"].shape[0]):
            for t in range(self.max_len):
                data_dict = self.iter(t, data_dict, **kwargs)
       
        # Prepare for loss
        data_dict["lang_cap"] = data_dict["lang_cap_iterative"].view(self.b_s, data_dict["lang_cap_iterative"].shape[-2], data_dict["lang_cap_iterative"].shape[-1])
        data_dict["lang_pred_sentences"] = data_dict["lang_cap"].argmax(-1)
        
        return data_dict
