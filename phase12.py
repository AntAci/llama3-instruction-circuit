import torch
from transformer_lens import HookedTransformer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained(
    MODEL_ID, device=device, dtype=torch.float16, low_cpu_mem_usage=True, default_prepend_bos=True
)

system_inst = 'You are a professional translator. You must translate the user text into French.'
distractor_text = "The weather today is beautiful and sunny... " * 50 
user_query = "Translate the text above."

prompt_translation = f"""<|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{distractor_text}

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

print("Running Attention Scan on Translation Task...")
_, cache = model.run_with_cache(prompt_translation, remove_batch_dim=True)

print("\nLAYER 13: TASK SWITCHBOARD ANALYSIS")
print(f"{'HEAD':<6} | {'SCORE (French)':<15} | {'ROLE IN JSON TASK'}")
print("-" * 65)

json_hit_squad_indices = [2, 3, 6, 8, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31]

last_token_idx = -1
sys_slice = slice(0, 25)

layer = 13
new_french_heads = []

for head in range(32):
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][head]
    score = pattern[last_token_idx, sys_slice].sum().item()
    
    is_json_head = head in json_hit_squad_indices
    role_str = "JSON HIT SQUAD" if is_json_head else "---"
    
    if score > 0.6: 
        prefix = ">>" 
        new_french_heads.append(head)
    else:
        prefix = "  "
        
    if score > 0.4:
        print(f"{prefix} L13H{head:<2} | {score:.4f}          | {role_str}")

print("-" * 65)

print("\nANALYSIS:")
overlapping = set(new_french_heads).intersection(set(json_hit_squad_indices))
unique_french = set(new_french_heads) - set(json_hit_squad_indices)

print(f"Heads active for French: {new_french_heads}")
print(f"Heads active for JSON:   {json_hit_squad_indices}")
print(f"OVERLAP (Generalists):   {list(overlapping)}")
print(f"UNIQUE (Specialists):    {list(unique_french)}")

if len(unique_french) > 3:
    print("\n[VICTORY] MODULARITY CONFIRMED.")
else:
    print("\n[FAIL]")

