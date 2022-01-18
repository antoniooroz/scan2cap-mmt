import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
# Meshed Memory Transformer imports
from models.meshed_memory_transformer.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory


class CapNetTransformer(nn.Module):
    def __init__(self, num_class, vocabulary, embeddings, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=256, num_locals=-1, vote_factor=1, sampling="vote_fps",
    no_caption=False, use_topdown=False, query_mode="corner", 
    graph_mode="graph_conv", num_graph_steps=0, use_relation=False, graph_aggr="add",
    use_orientation=False, num_bins=6, use_distance=False, use_new=False, 
    emb_size=300, hidden_size=512, attention_module_memory_slots=40, d_model=128, max_len=32, decoder_layers=3, transformer_d_k=64, transformer_d_v=64, transformer_h=8, transformer_d_ff=2048, transformer_dropout=0):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.no_caption = no_caption
        self.num_graph_steps = num_graph_steps

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        # Meshed Memory Transformer
        encoder = MemoryAugmentedEncoder(3, 0, 
            d_model=d_model, 
            d_in=128,
            num_proposals=num_proposal, 
            d_k=transformer_d_k,
            d_v=transformer_d_v,
            h=transformer_h,
            d_ff=transformer_d_ff,
            dropout=transformer_dropout,
            attention_module=ScaledDotProductAttentionMemory,
            attention_module_kwargs={'m': attention_module_memory_slots}
        )

        decoder = MeshedDecoder(
            len(vocabulary["word2idx"]), 
            max_len, 
            decoder_layers, 
            d_model=d_model,
            d_k = transformer_d_k,
            d_v = transformer_d_v,
            h = transformer_h,
            d_ff = transformer_d_ff,
            dropout=transformer_dropout
        )
        self.transformer = Transformer(encoder, decoder).cuda()

    def forward(self, data_dict, use_tf=True, is_eval=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        data_dict = self._detection_branch(data_dict, use_tf, is_eval)
        
        if self.num_graph_steps > 0: data_dict = self.graph(data_dict)
        #######################################
        #                                     #
        #      Meshed-Memory-Transformer      #
        #                                     #
        #######################################
        # object_proposals, object_masks, seq, *args):
        data_dict = self.transformer(data_dict, is_eval)

        return data_dict
    
    def _detection_branch(self, data_dict, use_tf=True, is_eval=False):
        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)
        return data_dict
    
    def beam_search(self, data_dict, use_tf=True, is_eval=False, max_len=32, beam_size=5):
        data_dict = self._detection_branch(data_dict, use_tf, is_eval)
        data_dict = self.transformer.beam_search(data_dict, max_len=max_len, eos_idx=3, is_eval=is_eval, beam_size=beam_size)
        return data_dict
    
    def iterative(self, data_dict, use_tf=True, is_eval=False, max_len=32):
        data_dict = self._detection_branch(data_dict, use_tf, is_eval)
        data_dict = self.transformer.iterative(data_dict, max_len=max_len, eos_idx=3, is_eval=is_eval)
        return data_dict
    
    def get_best_object_proposal(self, data_dict):
        return self.transformer.get_best_object_proposal(data_dict)
