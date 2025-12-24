import torch
from transformer_lens import HookedTransformer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_ID} on {device}...")

model = HookedTransformer.from_pretrained(
    MODEL_ID, 
    device=device, 
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    fold_ln=False, 
    center_writing_weights=False, 
    center_unembed=False,
    default_prepend_bos=True
)

system_inst = 'You are a strict JSON formatting assistant. No matter the content, you must output your final answer as valid JSON with the key "summary".'
distractor_text = "Soil erosion is the displacement of the upper layer of soil... " * 50 
user_query = "Summarize the text above."

prompt_long = f"""<|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{distractor_text}

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prompt = prompt_long 

print("Running Attention Scan...")

logits, cache = model.run_with_cache(prompt, remove_batch_dim=True)

tokens = model.to_str_tokens(prompt)
n_tokens = len(tokens)
print(f"Total Tokens: {n_tokens}")

instruction_slice = slice(0, 40) 
last_token_idx = -1

scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        
        if pattern.ndim == 4:
            pattern = pattern[0]
        if pattern.ndim == 3:
            pattern = pattern[head]
        
        score = pattern[last_token_idx, instruction_slice].sum()
        scores[layer, head] = score.item()

top_heads_indices = (scores > 0.2).nonzero(as_tuple=False)

print("\nTOP HEADS DETECTED (Attending to Instruction):")
print("Layer | Head | Attention Score")
found_heads = []
for idx in top_heads_indices:
    l, h = idx[0].item(), idx[1].item()
    score = scores[l, h].item()
    print(f" {l:2d}   |  {h:2d}  | {score:.4f}")
    found_heads.append((l, h))

print(f"\nNew Candidate List: {found_heads}")

if found_heads:
    top_l, top_h = found_heads[0]
    print(f"\nVISUALIZING TOP HEAD L{top_l}H{top_h}")
    print(f"Checking what the last token '{tokens[-1]}' is looking at:")
    
    pattern = cache[f"blocks.{top_l}.attn.hook_pattern"]
    
    if pattern.ndim == 4:
        pattern = pattern[0]
    if pattern.ndim == 3:
        pattern = pattern[top_h]
    
    attn_weights = pattern[last_token_idx, :]
    
    for i in range(len(attn_weights)):
        w = attn_weights[i].item()
        if w > 0.05:
            print(f"Token '{tokens[i]}' \t(Pos {i}): \t{w:.4f}")

