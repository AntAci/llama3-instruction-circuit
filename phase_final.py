import torch
from transformer_lens import HookedTransformer
import pandas as pd

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained(
    MODEL_ID, device=device, dtype=torch.float16, low_cpu_mem_usage=True, default_prepend_bos=True
)

L13_HIT_SQUAD = [
    (13, 2), (13, 3), (13, 6), (13, 8), 
    (13, 20), (13, 21), (13, 22), (13, 23), 
    (13, 24), (13, 26), (13, 27), (13, 29), 
    (13, 30), (13, 31)
]

system_inst = 'You are a strict JSON formatting assistant.'
distractor_text = "Soil erosion... " * 10
user_query = "Summarize the text above."

prompt = f"""<|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|>

{distractor_text}

<|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

print("Running Scan on Layer 13 Hit Squad...")
_, cache = model.run_with_cache(prompt, remove_batch_dim=True)

print(f"\n{'HEAD':<6} | {'ATTENTION TO SYSTEM INSTRUCTION':<35}")
print("-" * 45)

last_token_idx = -1
sys_slice = slice(0, 20)

for layer, head in L13_HIT_SQUAD:
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][head]
    score = pattern[last_token_idx, sys_slice].sum().item()
    
    bar = "#" * int(score * 20)
    print(f"L{layer}H{head:<2} | {bar:<20} ({score:.1%})")

print("-" * 45)

