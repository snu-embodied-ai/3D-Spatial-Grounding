import torch
import torch.nn as nn
import torch.nn.functional as F

from .detr_layers import DetrTransformerEncoderLayer
from .attention import CrossAttentionLayer
from .DINO import BiAttentionBlock

class BiDirectionalTransformerEncoder(nn.Module):
    def __init__(
            self,
            v_dim,
            l_dim,
            embed_dim,
            num_heads,
            num_biformers,
            pre_norm,
            dropout=0.1,
            use_whole_scene=False
    ):
        super(BiDirectionalTransformerEncoder, self).__init__()

        self.use_whole_scene = use_whole_scene

        self.self_attentions = nn.ModuleList()
        self.bi_attentions = nn.ModuleList()
        self.num_biformers = num_biformers
        self.word_self_attentions = nn.ModuleList()

        for i in range(num_biformers):
            bi_attention = BiAttentionBlock(v_dim, l_dim, embed_dim, num_heads, dropout=dropout)
            self.bi_attentions.append(bi_attention)

            self_attention = DetrTransformerEncoderLayer(
                self_attn_cfg=dict(num_heads=num_heads, embed_dims=v_dim, dropout=dropout, batch_first=True),
                ffn_cfg=dict(embed_dims=v_dim, feedforward_channels=embed_dim, ffn_drop=dropout),
            )
            self.self_attentions.append(self_attention)

            word_self_attention = DetrTransformerEncoderLayer(
                self_attn_cfg=dict(num_heads=num_heads, embed_dims=l_dim, dropout=dropout, batch_first=True),
                ffn_cfg=dict(embed_dims=l_dim, feedforward_channels=embed_dim, ffn_drop=dropout),
            )
            self.word_self_attentions.append(word_self_attention)

    def forward(self, v_dict, l_dict):
        if self.use_whole_scene:
            pass
        else:
            vis_embed = v_dict["region_embedding"]
            vis_pos = v_dict["region_position_encoding"]

        lang_embed = l_dict["input_ids"]
        lang_padding_mask = l_dict["padding_mask"]
        lang_mask = l_dict["attention_mask"]
        lang_pos = l_dict["text_position_encoding"]
        
        for i in range(self.num_biformers):
            # 1. Vision self attention
            # No mask applied to vision embedding since point cloud tokens are all valid tokens
            vis_embed = self.self_attentions[i](query=vis_embed,
                                                query_pos=vis_pos,
                                                attn_masks=None,
                                                key_padding_mask=None)
            
            # 2. Text self attention
            lang_embed = self.word_self_attentions[i](
                query=lang_embed,
                query_pos=lang_pos,
                attn_masks=lang_mask.bool(),
                key_padding_mask=lang_padding_mask.bool()
            )
            
            # 3. Vision-to-Text attetion & Text-to-Vision attention (concurrent operation)
            # Only masking the padding tokens of the language embeding, not masking the vision tokens
            vis_embed, lang_embed = self.bi_attentions[i](vis_embed, lang_embed,
                                                        attention_mask_v=None, 
                                                        attention_mask_l=~lang_padding_mask.bool()) # 0 allow 
            
        l_dict["input_ids"] = lang_embed

        if self.use_whole_scene:
            pass
        else:
            v_dict["region_embedding"] = vis_embed

        return v_dict, l_dict
    

class BiDirectionalTransformerDecoder(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        num_biformers,
        pre_norm,
        dropout=0.1,
        use_whole_scene=False
    ):
        super(BiDirectionalTransformerDecoder, self).__init__()

        self.use_whole_scene = use_whole_scene

        self.self_attentions = nn.ModuleList()
        self.cross_attentions = nn.ModuleList()
        self.bi_attentions = nn.ModuleList()
        self.num_biformers = num_biformers
        # self.word_self_attentions = nn.ModuleList()

        for i in range(num_biformers):
            bi_attention = BiAttentionBlock(v_dim, l_dim, embed_dim, num_heads, dropout=dropout)
            self.bi_attentions.append(bi_attention)

            cross_attention = CrossAttentionLayer(
                d_model=v_dim,
                nhead=num_heads,
                dropout=dropout,
                normalize_before=pre_norm,
            )
            self.cross_attentions.append(cross_attention)

            # Using the encoder layer since we don't use the cross attention layers in the decoder layer
            self_attention = DetrTransformerEncoderLayer(
                self_attn_cfg=dict(num_heads=num_heads, embed_dims=v_dim, dropout=dropout, batch_first=True),
                ffn_cfg=dict(embed_dims=v_dim, feedforward_channels=embed_dim, ffn_drop=dropout),
            )
            self.self_attentions.append(self_attention)

            # word_self_attention = DetrTransformerEncoderLayer(
            #     self_attn_cfg=dict(num_heads=num_heads, embed_dims=l_dim, dropout=dropout, batch_first=True),
            #     ffn_cfg=dict(embed_dims=l_dim, feedforward_channels=embed_dim, ffn_drop=dropout),
            # )
            # self.word_self_attentions.append(word_self_attention)
            
    def forward(self, v_dict, l_dict):
        queries = v_dict["query_embedding"]
        query_pos = v_dict["query_position_encoding"]

        if self.use_whole_scene:
            pass
        else:
            vis_embed = v_dict["region_embedding"]
            vis_pos = v_dict["region_position_encoding"]

        lang_embed = l_dict["input_ids"]
        lang_padding_mask = l_dict["padding_mask"]
        lang_mask = l_dict["attention_mask"]
        lang_pos = l_dict["text_position_encoding"]
        
        for i in range(self.num_biformers):
            # 1. Query self attention (Query = Candidates)
            # No masking applied to Query Token self attetions
            queries = self.self_attentions[i](query=queries,
                                              query_pos=query_pos,
                                              attn_masks=None,
                                              key_padding_mask=None)
            
            # 2. Query-to-Region(vision) cross attention
            queries = self.cross_attentions[i](tgt=queries,
                                               memory=vis_embed,
                                               memory_mask=None,
                                               memory_key_padding=None,
                                               pos=vis_pos,
                                               query_pos=query_pos)
            
            # 3. Text self attention
            # lang_embed = self.word_self_attentions[i](
            #     query=lang_embed,
            #     query_pos=lang_pos,
            #     attn_masks=lang_padding_mask,
            #     key_padding_mask=lang_mask
            # )
            
            # 4. Query-to-Text attetion & Text-to-Query attention (concurrent operation)
            # Only masking the padding tokens of the language embeding, not masking the query tokens
            queries, _ = self.bi_attentions[i](queries, lang_embed,
                                               attention_mask_v=None, attention_mask_l=~lang_padding_mask.bool()) # 0 allow 
            
        # l_dict["text_embedding"] = lang_embed
        v_dict["query_embedding"] = queries

        return v_dict, l_dict