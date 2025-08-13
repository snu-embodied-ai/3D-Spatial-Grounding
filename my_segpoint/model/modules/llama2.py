from transformers import LlamaForCausalLM
import torch
import torch.nn.functional as F
from transformers.generation.utils import GenerationMixin, BeamSearchOutput

"""
Generate by GPT-4o
Use this when you lack memory
"""
class LlamaWithFinalHiddenState(LlamaForCausalLM):
    def beam_search(
        self,
        input_ids,
        beam_scorer,
        logits_processor=None,
        stopping_criteria=None,
        max_length=None,
        pad_token_id=None,
        eos_token_id=None,
        output_attentions=False,
        output_hidden_states=True,  # must be True
        return_dict_in_generate=True,
        synced_gpus=False,
        **model_kwargs,
    ):
        # === Setup ===
        batch_size = input_ids.shape[0]
        num_beams = beam_scorer.num_beams
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        final_hidden_states = [[] for _ in range(batch_size)]

        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # Forward pass with hidden states
            outputs = self(
                **model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=True,  # Required
                return_dict=True,
            )

            # Get logits and hidden states
            next_token_logits = outputs.logits[:, -1, :]
            hidden_states = outputs.hidden_states[-1][:, -1, :]  # last hidden state at current time

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores[:, None]

            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            next_token_ids = next_tokens.indices
            next_token_scores = next_tokens.values

            next_input_ids, beam_indices = beam_scorer.process(
                input_ids, next_token_ids, next_token_scores, eos_token_id=eos_token_id
            )

            # Save final hidden states for selected beams
            for i in range(batch_size):
                beam_id = beam_indices[i].item()
                final_hidden_states[i].append(hidden_states[beam_id].detach())

            input_ids = next_input_ids
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )

            if beam_scorer.is_done or stopping_criteria(input_ids, scores=next_token_scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            next_token_scores,
            next_token_ids,
            beam_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        # Stack final hidden states
        final_hidden_tensor = [
            torch.stack(states, dim=0)  # (seq_len, hidden_dim)
            for states in final_hidden_states
        ]
        final_hidden_tensor = torch.stack(final_hidden_tensor, dim=0)  # (batch, seq_len, hidden_dim)

        if not return_dict_in_generate:
            return sequence_outputs[0]

        return BeamSearchOutput(
            sequences=sequence_outputs[0],
            sequences_scores=sequence_outputs[1],
            scores=None,
            beam_indices=None,
            hidden_states=None,
            attentions=None,
            final_hidden_states=final_hidden_tensor,
        )