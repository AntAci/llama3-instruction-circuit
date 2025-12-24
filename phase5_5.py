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

def layer_block_ablate_hook(value, hook):
    value[:] = 0.0 
    return value

def run_precision_test(name, layers):
    print(f"\nSURGICAL STRIKE: {name} (Layers {min(layers)}-{max(layers)})")
    
    hook_points = [get_act_name("z", l) for l in layers]
    hooks = [(point, layer_block_ablate_hook) for point in hook_points]
    
    print("Testing Long Range (Retrieval)...")
    with model.hooks(fwd_hooks=hooks):
        out = model.generate(model.to_tokens(prompt_long), max_new_tokens=40, temperature=0, stop_at_eos=True, verbose=False)
    resp = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
    pass_long = resp.startswith("{") or '"summary":' in resp
    print(f"Output: {resp[:80]}...")
    
    print("Testing Short Range (Intelligence Check)...")
    with model.hooks(fwd_hooks=hooks):
        out = model.generate(model.to_tokens(prompt_short), max_new_tokens=40, temperature=0, stop_at_eos=True, verbose=False)
    resp = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
    pass_short = resp.startswith("{") or '"summary":' in resp
    print(f"Output: {resp[:80]}...")

    if not pass_long and pass_short:
        print("[SUCCESS] RESULT: CIRCUIT ISOLATED!")
    elif not pass_long and not pass_short:
        print("[FAIL] RESULT: LOBOTOMY")
    else:
        print("[WARNING] RESULT: CIRCUIT SURVIVED")

run_precision_test("LAYER 12-14", [12, 13, 14])

