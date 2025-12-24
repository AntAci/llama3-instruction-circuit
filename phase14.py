import torch
from transformer_lens import HookedTransformer
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}...")

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
print(f"Loading {MODEL_ID}...")

model = HookedTransformer.from_pretrained(
    MODEL_ID,
    device=device,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    low_cpu_mem_usage=True,
    dtype=torch.float16,
    default_prepend_bos=True
)

needle = "515151"
distractor_phrase = "The soil erosion process is natural and ongoing. "
distractor_block = distractor_phrase * 60

prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

The magic number is {needle}.

{distractor_block}

What is the magic number?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

layer_idx = 13
hit_squad_indices = [2, 3, 6, 8, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31]

print("Running 'Needle in a Haystack' probe...")
logits, cache = model.run_with_cache(prompt, remove_batch_dim=True)

attn_pattern = cache[f"blocks.{layer_idx}.attn.hook_pattern"]

str_tokens = model.to_str_tokens(prompt)
trigger_pos = -1
needle_zone_end = 15

print(f"\nATTENTION SCORES (Layer {layer_idx})")
print(f"Checking attention from 'Trigger' to 'Needle Zone' (Tokens 0-{needle_zone_end})")
print("-" * 50)
print(f"{'Head':<10} | {'Score':<10} | {'Status':<15} | {'Visualization'}")
print("-" * 50)

hit_squad_scores = []

for head_idx in hit_squad_indices:
    score = attn_pattern[head_idx, trigger_pos, :needle_zone_end].sum().item()
    hit_squad_scores.append(score)
    
    bar_len = int(score * 20)
    bar = "#" * bar_len
    
    if score > 0.5:
        status = "ACTIVE"
    else:
        status = "QUIET"

    print(f"L{layer_idx}H{head_idx:<5} | {score:.4f}     | {status:<15} | {bar}")

print("-" * 50)

avg_score = np.mean(hit_squad_scores)
print(f"\nAverage Hit Squad Score: {avg_score:.4f}")

if avg_score < 0.4:
    print("\nRESULT: PASS")
    print("The heads are QUIET.")
else:
    print("\nRESULT: WARNING")
    print("The heads are ACTIVE.")

