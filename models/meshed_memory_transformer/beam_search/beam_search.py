import torch
import utils_meshed_memory_transformer as utils


class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx # TODO: Check if given
        self.beam_size = beam_size
        self.b_s = None
        self.device = None

    def apply(self, data_dict, out_size=1, is_eval=False, **kwargs):
        if is_eval:
            return self.apply_eval(data_dict, out_size, **kwargs)
        else:
            return self.apply_train(data_dict, out_size, **kwargs)

    def apply_train(self, data_dict, out_size=1, **kwargs):
        data_dict = self.model.get_best_object_proposal(data_dict) # into target_object_proposals
        
        data_dict["enc_output"], data_dict["enc_mask"] = self.model.encoder(data_dict)
        
        data_dict = self.run_search(data_dict, out_size, **kwargs)

        data_dict["lang_cap"] = data_dict["lang_cap"].view(self.b_s * data_dict["lang_cap"].shape[-3], data_dict["lang_cap"].shape[-2], data_dict["lang_cap"].shape[-1])
        
        return data_dict

    def apply_eval(self, data_dict, out_size=1, **kwargs):
        enc_output, enc_mask = self.model.encode_for_beam_search(data_dict) # [12, 256, 3, 128]
        encoder_input = data_dict["encoder_input"]
        #bbox_features = data_dict["bbox_feature"]
        lang_caps = []
        lang_pred_sentences = []
        
        for i in range(encoder_input.shape[0]):
            data_dict["encoder_input"] = encoder_input[i].unsqueeze(0)
            data_dict["enc_output"] = enc_output[i].unsqueeze(0)
            data_dict["enc_output"] = enc_output[i].unsqueeze(0)
            data_dict["enc_mask"] = enc_mask[i].unsqueeze(0)
            
            data_dict = self.run_search(data_dict, out_size, **kwargs)
            #lang_caps.append(data_dict["lang_cap"])
            lang_pred_sentences.append(data_dict["lang_pred_sentences"])
            
        #data_dict["bbox_feature"] = encoder_input
        #data_dict["lang_cap"] = torch.cat(lang_caps, dim=0).cuda()
        data_dict["lang_pred_sentences"] = torch.cat(lang_pred_sentences, dim=0).cuda()
        
        return data_dict

    def run_search(self, data_dict, out_size=1, **kwargs):
        self.b_s = utils.get_batch_size(data_dict["encoder_input"])
        self.device = utils.get_device(data_dict["encoder_input"])
        
        device = data_dict["encoder_input"].device

        lang_ids_model = torch.ones([self.b_s, 1]).to(device).int() * 2
        data_dict["lang_ids_model"] = lang_ids_model

        data_dict["bs_seq_mask"] = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
        data_dict["bs_seq_logprob"] = torch.zeros((self.b_s, 1, 1), device=self.device)
        data_dict["bs_log_probs"] = []
        data_dict["bs_selected_words"] = None
        data_dict["bs_all_log_probs"] = []
        
        data_dict["bs_outputs"] = []
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                data_dict = self.iter(t, data_dict, **kwargs)

        # Sort result
        seq_logprob, sort_idxs = torch.sort(data_dict["bs_seq_logprob"], 1, descending=True)
        outputs = data_dict["bs_outputs"]
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        log_probs = torch.cat(data_dict["bs_log_probs"], -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))

        all_log_probs = torch.cat(data_dict["bs_all_log_probs"], 2)
        all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size, self.max_len, all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)
       
        data_dict["lang_pred_sentences"] = outputs
        data_dict["lang_cap"] = all_log_probs
        
        return data_dict

    def iter(self, t: int, data_dict, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size

        data_dict = self.model.step(t, data_dict, mode='feedback', **kwargs)
        words = data_dict["lang_cap"].argmax(-1) # TODO: from iterative
        
        word_logprob = data_dict["lang_cap"]
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)
        candidate_logprob = data_dict["bs_seq_logprob"] + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (data_dict["bs_selected_words"].view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
            data_dict["bs_seq_mask"] = data_dict["bs_seq_mask"] * mask
            word_logprob = word_logprob * data_dict["bs_seq_mask"].expand_as(word_logprob)
            old_seq_logprob = data_dict["bs_seq_logprob"].expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = data_dict["bs_seq_mask"] * candidate_logprob + old_seq_logprob * (1 - data_dict["bs_seq_mask"])

        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs)
        selected_beam = (selected_idx / candidate_logprob.shape[-1]).type(torch.LongTensor).cuda()
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))

        # Expand encoder output
        if t == 0:
            data_dict["enc_output"] = data_dict["enc_output"].repeat_interleave(self.beam_size, dim=0)
            data_dict["enc_mask"] = data_dict["enc_mask"].repeat_interleave(self.beam_size, dim=0)
            data_dict["encoder_input"] = data_dict["encoder_input"].repeat_interleave(self.beam_size, dim=0)

        data_dict["bs_seq_logprob"] = selected_logprob.unsqueeze(-1)
        data_dict["bs_seq_mask"] = torch.gather(data_dict["bs_seq_mask"], 1, selected_beam.unsqueeze(-1))
        data_dict["bs_outputs"] = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in data_dict["bs_outputs"])
        data_dict["bs_outputs"].append(selected_words.unsqueeze(-1))
        
        # Return all probs
        if t == 0:
            data_dict["bs_all_log_probs"].append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
        else:
            data_dict["bs_all_log_probs"].append(word_logprob.unsqueeze(2))
        ###

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        data_dict["bs_log_probs"] = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in data_dict["bs_log_probs"])
        data_dict["bs_log_probs"].append(this_word_logprob)
        data_dict["bs_selected_words"] = selected_words.view(-1, 1)
        data_dict["lang_ids_model"] = data_dict["bs_selected_words"]

        return data_dict

    def select(self, t, candidate_logprob, **kwargs):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        return selected_idx, selected_logprob

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn