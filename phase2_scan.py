import torch
from transformer_lens import HookedTransformer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda"

print(f"Loading {MODEL_ID} (Refined Scan)...")
model = HookedTransformer.from_pretrained(
    MODEL_ID, 
    device=device, 
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    fold_ln=False, center_writing_weights=False, center_unembed=False
)
model.eval()

system_inst = 'You are a strict JSON formatting assistant. No matter the content, you must output your final answer as valid JSON with the key "summary".'
distractor_text = "Soil erosion is the displacement of the upper layer of soil... " * 50
user_query = "Summarize the text above."
prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{distractor_text}

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

print("Tokenizing...")
tokens = model.to_tokens(prompt)
str_tokens = model.to_str_tokens(prompt)
seq_len = tokens.shape[1]

sys_content_start = 0
sys_end = 0

for i, t in enumerate(str_tokens):
    if "You" in t and "are" in str_tokens[i+1]:
        sys_content_start = i
        break

for i, t in enumerate(str_tokens):
    if t == "<|eot_id|>":
        sys_end = i
        break

print(f"Refined System Span (Words Only): Tokens {sys_content_start} to {sys_end}")
print(f"Sample: {str_tokens[sys_content_start:sys_content_start+5]} ...")

last_token_idx = seq_len - 1

head_scores = []

def anchor_hook(pattern, hook):
    relevant_attn = pattern[0, :, -1, sys_content_start:sys_end]
    score = relevant_attn.sum(dim=-1).cpu().float()
    layer = hook.layer()
    for head_idx, s in enumerate(score):
        head_scores.append((layer, head_idx, s.item()))

print("\nRunning Refined Scan (Ignoring BOS Sinks)...")
pattern_filter = lambda name: name.endswith("pattern")

with torch.no_grad():
    model.run_with_hooks(
        tokens, 
        fwd_hooks=[(pattern_filter, anchor_hook)]
    )

head_scores.sort(key=lambda x: x[2], reverse=True)

print("\nTRUE ANCHOR HEADS (Content Aware):")
print(f"{'Layer':<10} {'Head':<10} {'Attn Score':<10}")
print("-" * 30)
for i in range(20):
    l, h, s = head_scores[i]
    print(f"L{l:<9} H{h:<9} {s:.4f}")

top_5 = [(x[0], x[1]) for x in head_scores[:5]]
print(f"\nNew Top 5 to Ablate: {top_5}")