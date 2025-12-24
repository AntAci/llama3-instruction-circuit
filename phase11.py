import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_ID} on {device}...")

model = HookedTransformer.from_pretrained(
    MODEL_ID, 
    device=device, 
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    default_prepend_bos=True
)

system_inst = 'You are a strict JSON formatting assistant. No matter the content, you must output your final answer as valid JSON with the key "summary".'
distractor_text = "Soil erosion is the displacement of the upper layer of soil... " * 50 
user_query = "Summarize the text above."

prompt_long = f"""<|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{distractor_text}

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prompt_short = f"""<|start_header_id|>user<|end_header_id|>

{distractor_text}

[INSTRUCTION]: {system_inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

L13_HIT_SQUAD = [
    (13, 2), (13, 3), (13, 6), (13, 8), 
    (13, 20), (13, 21), (13, 22), (13, 23), 
    (13, 24), (13, 26), (13, 27), (13, 29), 
    (13, 30), (13, 31)
]

print(f"Targeting {len(L13_HIT_SQUAD)} Heads in Layer 13...")

def noise_ablate_hook(value, hook):
    layer = hook.layer()
    heads_to_jam = [h for (l, h) in L13_HIT_SQUAD if l == layer]
    mean = value.mean()
    std = value.std()
    noise_block = torch.randn_like(value) * std + mean
    for h in heads_to_jam:
        value[:, :, h, :] = noise_block[:, :, h, :]
    return value

hook_point = get_act_name("z", 13)
hooks = [(hook_point, noise_ablate_hook)]

print("\nPHASE 11: LAYER 13 HIT SQUAD")

print("Testing Long Range (Retrieval)...")
with model.hooks(fwd_hooks=hooks):
    out = model.generate(model.to_tokens(prompt_long), max_new_tokens=60, temperature=0, stop_at_eos=True, verbose=False)
resp_long = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
is_json_long = resp_long.startswith("{") or '"summary":' in resp_long
print(f"Output: {resp_long[:100]}...")

print("Testing Short Range (Intelligence Check)...")
with model.hooks(fwd_hooks=hooks):
    out = model.generate(model.to_tokens(prompt_short), max_new_tokens=60, temperature=0, stop_at_eos=True, verbose=False)
resp_short = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
is_coherent_short = len(resp_short) > 10 and "soil erosion" not in resp_short[:20].lower()
print(f"Output: {resp_short[:100]}...")

if not is_json_long and is_coherent_short:
    print("\n[SUCCESS] CIRCUIT MAPPED.")
elif is_json_long:
    print("\n[WARNING] Signal Survived.")
else:
    print("\n[FAIL] Lobotomy.")

