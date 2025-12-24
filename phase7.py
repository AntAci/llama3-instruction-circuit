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

def run_noise_test(name, layers):
    print(f"\nPHASE 7: NOISE ABLATION - {name} (Layers {min(layers)}-{max(layers)})")
    
    hook_points = [get_act_name("z", l) for l in layers]
    hooks = [(point, noise_ablate_hook) for point in hook_points]
    
    print("Testing Long Range (Retrieval)...")
    with model.hooks(fwd_hooks=hooks):
        out = model.generate(model.to_tokens(prompt_long), max_new_tokens=60, temperature=0, stop_at_eos=True, verbose=False)
    
    resp_long = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
    is_json_long = resp_long.startswith("{") or '"summary":' in resp_long
    
    print(f"Output: {resp_long[:100]}...")
    if is_json_long:
        print("Result: [PASS]")
    else:
        print("Result: [FAIL]")

    print("Testing Short Range...")
    with model.hooks(fwd_hooks=hooks):
        out = model.generate(model.to_tokens(prompt_short), max_new_tokens=60, temperature=0, stop_at_eos=True, verbose=False)
    
    resp_short = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
    is_json_short = resp_short.startswith("{") or '"summary":' in resp_short
    
    print(f"Output: {resp_short[:100]}...")
    if is_json_short:
        print("Result: [PASS] Signal Survived")
    else:
        print("Result: [FAIL] Lobotomy")

    if not is_json_long and is_json_short:
        print("\n[SUCCESS] Dissociation Confirmed!")
        print(f"The circuit is distributed across Layers {min(layers)}-{max(layers)}.")
    else:
        print("\n[WARNING] Experiment Inconclusive.")

run_noise_test("CRITICAL ZONE", [12, 13, 14, 15])

