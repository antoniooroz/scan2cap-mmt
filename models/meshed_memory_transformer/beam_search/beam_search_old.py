import torch
import utils_meshed_memory_transformer as utils


class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

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

    def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
            visual_exp = visual.view(visual_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual

    def apply(self, data_dict, out_size=1, return_probs=False, **kwargs):
        # Prepare
        # data_dict = self._batchify(data_dict) # num_proposals are also seen as batches
        # lang_ids_model = torch.ones([data_dict["target_object_proposal"].shape[0], 1]).to(data_dict["target_object_proposal"].device).int() * 2
        # data_dict["lang_ids_model"] = lang_ids_model
        # data_dict["beam_search_outputs"] = []
        
        self.b_s = utils.get_batch_size(data_dict["bbox_feature"])
        self.device = utils.get_device(data_dict["bbox_feature"])
        
        for proposal_idx in range(data_dict["bbox_feature"].shape[1]):
            # preparation
            data_dict["target_object_proposal"] = data_dict["bbox_feature"][:, proposal_idx]
            lang_ids_model = torch.ones([data_dict["target_object_proposal"].shape[0], 1], dtype=int).to(data_dict["target_object_proposal"].device) * 2
            data_dict["lang_ids_model"] = lang_ids_model
            data_dict["beam_search_outputs"] = []
            enc_output, enc_mask = self.model.encode_for_beam_search(data_dict)
            
            self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
            self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
            self.log_probs = []
            self.selected_words = None
            if return_probs:
                self.all_log_probs = []

            with self.model.statefulness(self.b_s):
                self.model.enc_output = enc_output
                self.model.enc_mask = enc_mask
                for t in range(self.max_len):
                    data_dict = self.iter(t, data_dict, return_probs, **kwargs)

            # Sort result
            seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
            outputs = data_dict["beam_search_outputs"]
            outputs = torch.cat(outputs, -1)
            outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
            log_probs = torch.cat(self.log_probs, -1)
            log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
            if return_probs:
                all_log_probs = torch.cat(self.all_log_probs, 2)
                all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                                            self.max_len,
                                                                                            all_log_probs.shape[-1]))
            outputs = outputs.contiguous()[:, :out_size]
            log_probs = log_probs.contiguous()[:, :out_size]
            if out_size == 1:
                outputs = outputs.squeeze(1)
                log_probs = log_probs.squeeze(1)

            if return_probs:
                return outputs, log_probs, all_log_probs
            else:
                return outputs, log_probs

    def select(self, t, candidate_logprob, **kwargs):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        return selected_idx, selected_logprob

    def iter(self, t: int, data_dict, return_probs, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size            

        data_dict = self.model.step(t, data_dict, mode='feedback', **kwargs)# previously: self.selected_words, visual instead of data_dict
        word_logprob = data_dict["lang_cap"]
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)
        candidate_logprob = self.seq_logprob + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)

        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs)
        selected_beam = selected_idx / candidate_logprob.shape[-1]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))
        # visual = self._expand_visual(visual, cur_beam_size, selected_beam)

        self.seq_logprob = selected_logprob.unsqueeze(-1)
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
        data_dict["beam_search_outputs"] = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in data_dict["beam_search_outputs"])
        data_dict["beam_search_outputs"].append(selected_words.unsqueeze(-1))

        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        self.log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs)
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)

        return data_dict
        
    def _batchify(self, data_dict):
        B, N = data_dict["bbox_feature"].shape[0], data_dict["bbox_feature"].shape[1]
        
        data_dict["lang_ids_model"] = torch.zeros([B * N, data_dict["lang_ids"].shape[1] - 1]).cuda().int()
        data_dict["target_object_proposal"] = data_dict["bbox_feature"].reshape(B*N, -1)
        
        return data_dict
    
    def _unbatchify(self, data_dict, B, N):
        data_dict["lang_cap"] = data_dict["lang_cap"].reshape(B,N,data_dict["lang_cap"].shape[-2], data_dict["lang_cap"].shape[-1])
        return data_dict
