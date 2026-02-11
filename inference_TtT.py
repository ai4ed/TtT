import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import os, sys

from modeling_qwen_TtT import Qwen2ForARDiffLM

def apply_top_k_top_p(probs, top_k=None, top_p=None, min_tokens_to_keep=1):
    """
    Apply top-k and/or top-p (nucleus) filtering to a probability distribution over vocabulary.
    Args:
        probs: Tensor of shape (batch_size, vocab_size)
        top_k: keep only top_k tokens with highest probability
        top_p: keep the smallest set of tokens whose cumulative probability >= top_p
        min_tokens_to_keep: ensure at least this many tokens are kept
    Returns:
        Filtered probs normalized to sum to 1 along the last dimension.
    """
    if top_k is not None and top_k > 0:
        top_k = min(top_k, probs.size(-1))
        topk_vals, topk_indices = torch.topk(probs, top_k, dim=-1)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        probs = probs.masked_fill(~mask, 0.0)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens where cumulative probability up to previous token is < top_p
        # This keeps at least the first token even if its prob > top_p
        sorted_keep = (cumulative_probs - sorted_probs) < top_p
        # Ensure at least min_tokens_to_keep tokens are kept
        sorted_keep[..., :min_tokens_to_keep] = True
        keep_mask = torch.zeros_like(probs, dtype=torch.bool)
        keep_mask.scatter_(dim=1, index=sorted_indices, src=sorted_keep)
        probs = probs.masked_fill(~keep_mask, 0.0)

    probs_sum = probs.sum(dim=-1, keepdim=True)
    probs = probs / torch.clamp(probs_sum, min=1e-12)
    return probs

def generate(model, tokenizer, prompt, max_gen_len=512, diffusion_steps=128, diffusion_gen_length=256, block_length=32, ar_temperature=0., diffusion_temperature=0., cfg_scale=0., remasking='low_confidence', top_k=None, top_p=None, min_text_tokens_after_audio=0):
    device = model.device
    mask_id = tokenizer.mask_token_id
    
    begin_audio_ids = [
        tokenizer.convert_tokens_to_ids("<|begin_of_audio|>"),
        tokenizer.convert_tokens_to_ids("<|begin_of_quad|>")
    ]
    
    end_audio_ids = [
        tokenizer.convert_tokens_to_ids("<|end_of_audio_new|>"),
        tokenizer.convert_tokens_to_ids("<|end_of_audio_span_last|>")
    ]
    
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    print("mask_id is:", mask_id)
    print("begin_audio_ids are:", begin_audio_ids)
    print("end_audio_ids are:", end_audio_ids)
    print("im_end_id is:", im_end_id)
    
    if mask_id is None:
        raise ValueError("Tokenizer must have a mask token.")

    prompt_ids = torch.tensor(tokenizer.encode(prompt)).to(device).unsqueeze(0)
    generated_ids = prompt_ids.clone()
    
    current_len = 0
    in_diffusion = False
    # After finishing an audio span, enforce N text tokens before allowing im_end
    text_tokens_enforce_counter = 0
    current_diffusion_gen_length = diffusion_gen_length
    current_block_length = block_length
    
    while generated_ids.shape[1] < max_gen_len + prompt_ids.shape[1]:
        if in_diffusion:
            # Diffusion part: generate block by block until an end token is found
            
            # Calculate steps per block. The total steps are for a full gen_length span.
            if current_diffusion_gen_length % current_block_length != 0:
                 raise ValueError("current_diffusion_gen_length must be divisible by current_block_length")
            num_blocks_in_span = current_diffusion_gen_length // current_block_length
            if diffusion_steps % num_blocks_in_span != 0:
                raise ValueError("diffusion_steps must be divisible by (current_diffusion_gen_length / current_block_length)")
            steps_per_block = diffusion_steps // num_blocks_in_span

            diffusion_output = generate_diffusion_span(
                model, generated_ids, steps=steps_per_block, gen_length=diffusion_gen_length,
                block_length=block_length, temperature=diffusion_temperature, cfg_scale=cfg_scale,
                remasking=remasking, mask_id=mask_id, forbidden_token_ids=[im_end_id], end_token_ids=end_audio_ids
            )
            
            newly_generated = diffusion_output[:, generated_ids.shape[1]:]
            print("start diffusion part, newly_generated is:", newly_generated)
            
            end_mask = torch.zeros_like(newly_generated, dtype=torch.bool)
            for end_id in end_audio_ids:
                end_mask = end_mask | (newly_generated == end_id)
            end_mask = end_mask | (newly_generated == im_end_id)
            
            if torch.any(end_mask):
                first_end_pos = torch.where(end_mask)[1][0]
                audio_part = newly_generated[:, :first_end_pos + 1]
                generated_ids = torch.cat([generated_ids, audio_part], dim=1)
                
                if newly_generated[0, first_end_pos] == im_end_id:
                    break # End of generation
                print("current generated_ids is:", generated_ids)
                in_diffusion = False # Switch back to AR
                # Enforce some text tokens after finishing an audio span
                text_tokens_enforce_counter = max(text_tokens_enforce_counter, min_text_tokens_after_audio)
            else:
                # If end token not found, append the block and continue diffusion in the next iteration
                generated_ids = torch.cat([generated_ids, newly_generated], dim=1)
        else:
            # AR part
            logits = model(generated_ids, mode='ar').logits[:, -1, :]
            # If we just finished an audio span, temporarily forbid <|im_end|>
            if text_tokens_enforce_counter > 0:
                logits[..., im_end_id] = -float('inf')
            use_sampling = (ar_temperature > 0) or ((top_k is not None and top_k > 0) or (top_p is not None and 0.0 < top_p < 1.0))
            if use_sampling:
                temp = ar_temperature if ar_temperature > 0 else 1.0
                probs = F.softmax(logits / temp, dim=-1).to(torch.float32)
                if (top_k is not None and top_k > 0) or (top_p is not None and 0.0 < top_p < 1.0):
                    probs = apply_top_k_top_p(probs, top_k=top_k, top_p=top_p)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            print("start ar part, the current generated_ids is:", next_token_id)
            
            if next_token_id.item() in begin_audio_ids:
                in_diffusion = True # Switch to diffusion
                # If we start a new audio span early, we stop enforcing text tokens
                text_tokens_enforce_counter = 0
            elif next_token_id.item() == im_end_id:
                break # End of generation
            else:
                if text_tokens_enforce_counter > 0:
                    text_tokens_enforce_counter -= 1

    return generated_ids

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate_diffusion_span(model, prompt, steps=128, gen_length=128, block_length=32, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=168069, forbidden_token_ids=None, end_token_ids=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps per block.
        gen_length: Maximum generated answer length.
        block_length: Block length for each generation step.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 168069.
        forbidden_token_ids: List of token ids that are disallowed during diffusion (e.g., im_end_id).
        end_token_ids: List of token ids that signal the end of generation (e.g., end_audio_id).
    '''
    current_sequence = prompt.clone()
    
    while current_sequence.shape[1] < prompt.shape[1] + gen_length:
        # Create input for current block: current_sequence + one block of masks
        x = torch.full((1, current_sequence.shape[1] + block_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :current_sequence.shape[1]] = current_sequence.clone()
        
        prompt_index = (x != mask_id)
        # Only the new block part should be considered as mask
        block_mask_index = (x[:, current_sequence.shape[1]:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_, mode='diffusion').logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                model_out = model(x, return_dict=True, mode='diffusion')
                logits = model_out.logits

            # Enforce that forbidden tokens cannot appear during diffusion
            if forbidden_token_ids:
                logits[..., forbidden_token_ids] = -float('inf')

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Only consider the current block for confidence
            x0_p[:, :current_sequence.shape[1]] = -np.inf

            x0 = torch.where(mask_index, x0, x) 
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

        # Extract the newly generated block
        new_block = x[:, current_sequence.shape[1]:]
        print("new blcok is:", new_block)
        # Check for end tokens in the newly generated block
        if end_token_ids:
            for end_token_id in end_token_ids:
                end_positions = (new_block == end_token_id).nonzero(as_tuple=True)
                if len(end_positions[1]) > 0:
                    # Found end token, truncate at first occurrence and return
                    first_end_pos = end_positions[1][0]
                    final_block = new_block[:, :first_end_pos + 1]
                    current_sequence = torch.cat([current_sequence, final_block], dim=1)
                    return current_sequence
        
        # No end token found, append the entire block and continue
        current_sequence = torch.cat([current_sequence, new_block], dim=1)
    
    return current_sequence

def main():
    device = 'cuda'
    model_path = "path/to/checkpoint"
    model = Qwen2ForARDiffLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<|mask_token|>")

    system_prompt = "You are an Automatic Speech Recognition (ASR) model. The user will provide you with an audio input. Your task is to transcribe the audio into text and output the result in an interleaved format: generate 13 text tokens followed by 26 audio tokens, and repeat this pattern until the transcription is complete."
    user_message = "<|begin_of_audio|><|audio_2971|><|audio_2427|><|audio_15411|><|audio_4846|><|audio_10396|><|audio_13159|><|audio_13277|><|audio_11735|><|audio_1552|><|audio_4363|><|audio_3938|><|audio_9450|><|audio_10185|><|audio_16217|><|audio_13533|><|audio_1402|><|audio_1278|><|audio_10055|><|audio_15179|><|audio_13012|><|audio_15976|><|audio_8168|><|audio_6317|><|audio_9169|><|audio_5662|><|audio_14343|><|audio_11983|><|audio_11667|><|audio_1176|><|audio_2487|><|audio_623|><|audio_3547|><|audio_4866|><|audio_12072|><|audio_7813|><|end_of_audio|>"

    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    print("Prompt:\n" + prompt)

    out = generate(model, tokenizer, prompt, max_gen_len=2048, diffusion_steps=200, diffusion_gen_length=2600, block_length=26, ar_temperature=0., diffusion_temperature=0., cfg_scale=0.1, remasking='low_confidence', top_k=10, top_p=0.95, min_text_tokens_after_audio=0)

    prompt_len = len(tokenizer.encode(prompt))
    print("\nGenerated output:\n" + tokenizer.decode(out[0, prompt_len:], skip_special_tokens=False))
    print("Finished!")

if __name__ == '__main__':
    main()
