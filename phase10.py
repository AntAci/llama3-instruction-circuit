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

def noise_ablate_hook(value, hook):
    mean = value.mean()
    std = value.std()
    noise = torch.randn_like(value) * std + mean
    return noise

def run_layer_test(layer_idx):
    print(f"\nTESTING LAYER {layer_idx} (Single Layer Noise)")
    
    hook_point = get_act_name("z", layer_idx)
    hooks = [(hook_point, noise_ablate_hook)]
    
    with model.hooks(fwd_hooks=hooks):
        out = model.generate(model.to_tokens(prompt_long), max_new_tokens=50, temperature=0, stop_at_eos=True, verbose=False)
    resp_long = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
    is_json_long = resp_long.startswith("{") or '"summary":' in resp_long
    
    with model.hooks(fwd_hooks=hooks):
        out = model.generate(model.to_tokens(prompt_short), max_new_tokens=50, temperature=0, stop_at_eos=True, verbose=False)
    resp_short = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
    is_coherent_short = len(resp_short) > 10 and "soil erosion" not in resp_short[:20].lower()
    
    print(f"Long Range Output: {resp_long[:80]}...")
    print(f"Short Range Output: {resp_short[:80]}...")
    
    if not is_json_long and is_coherent_short:
        print(f"[SUCCESS] LAYER {layer_idx}")
    elif not is_json_long and not is_coherent_short:
        print(f"[FAIL] LAYER {layer_idx} are vita")
    elif is_json_long:
        print(f"[WARNING] Signal Survived (Layer {layer_idx})")

for layer in [12, 13, 14, 15]:
    run_layer_test(layer)

