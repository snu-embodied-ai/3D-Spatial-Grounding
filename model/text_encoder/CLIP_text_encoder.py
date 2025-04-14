import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir, os.pardir))
sys.path.extend([ROOT_DIR, CUR_DIR])

import open_clip

from .tokenizer import SimpleTokenizer

def load_text_encoder(model_cfg):
    pretrained_path = os.path.join(ROOT_DIR, model_cfg["CLIP"]["pretrained_path"])
    text_encoder = OpenCLIPTextEncoder(model_name=model_cfg["CLIP"]["model_name"],pretrained_path=pretrained_path,)

    return text_encoder

def generate_masks_with_special_tokens_and_transfer_map(
        tokenized_text, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Only token pairs in between two special tokens are attended to
    and thus the attention mask for these pairs is positive.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.

    Returns:
        Tuple(Tensor, Tensor):
        - attention_mask is the attention mask between each tokens.
          Only token pairs in between two special tokens are positive.
          Shape: [bs, num_token, num_token].
        - position_ids is the position id of tokens within each valid sentence.
          The id starts from 0 whenenver a special token is encountered.
          Shape: [bs, num_token]
    """
    input_ids = tokenized_text
    bs, num_token = input_ids.shape
    # special_tokens_mask:
    # bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token),
                                      device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token,
                  device=input_ids.device).bool().unsqueeze(0).repeat(
                      bs, 1, 1))
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)

class OpenCLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str,
                 pretrained_path: str,
                 use_sub_sentence_represent=False):
        """
        - `use_sub_sentence_represent`:  (bool, optional): whether to use sub
            sentence represent introduced in `Grounding DINO
            <https://arxiv.org/abs/2303.05499>`. Defaults to False.
        """
        super().__init__()
        
        self.model_name = model_name
        self.pretrained_path = pretrained_path
 
        self.use_sub_sentence_represent = use_sub_sentence_represent

        self.text_encoder = open_clip.create_model(model_name=model_name, pretrained=pretrained_path)

        self._tokenizer = SimpleTokenizer()
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise Exception("Tokenizer not build yet. Build a tokenizer first!")
        else:
            return self._tokenizer
        
    def encode_text(self, text, normalize: bool = False):
        text_tower = self.text_encoder.text
        cast_dtype = text_tower.transformer.get_cast_dtype()

        x = text_tower.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + text_tower.positional_embedding.to(cast_dtype)
        x = text_tower.transformer(x, attn_mask=text_tower.attn_mask)
        x = text_tower.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        if text_tower.text_projection is not None:
            if isinstance(text_tower.text_projection, nn.Linear):
                x = text_tower.text_projection(x)
            else:
                x = x @ text_tower.text_projection

        if normalize:
            return F.normalize(x, dim=-1), text_tower.attn_mask
        else:
            return x, text_tower.attn_mask

    def forward(self, language_dict):
        """
        Forward function - Extract text token embedding using EVA-CLIP
        """
        device = next(self.text_encoder.parameters()).device

        tokenized_text = language_dict["input_ids"]

        encoded_input_ids, attention_mask = self.encode_text(language_dict["input_ids"],
                                                             normalize=True)

        if self.use_sub_sentence_represent:
            attention_mask, position_ids = \
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized_text, self.special_tokens)
            # token_type_ids = tokenized['token_type_ids']
            
            language_dict["attention_mask"] = attention_mask
            language_dict["position_ids"] = position_ids

        language_dict["input_ids"] = encoded_input_ids          # (batch_size, token_size, embed_dim)
        language_dict["attention_mask"] = attention_mask        # (token_size, token_size)

        return language_dict
