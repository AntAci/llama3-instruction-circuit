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

def layer_block_ablate_hook(value, hook):
    value[:] = 0.0 
    return value

block_deep = list(range(21, 32))
block_mid = list(range(10, 21))

def run_block_test(block_name, layers_to_kill):
    print(f"\nTESTING BLOCK: {block_name} (Layers {min(layers_to_kill)}-{max(layers_to_kill)})")
    
    hook_points = [get_act_name("z", l) for l in layers_to_kill]
    hooks = [(point, layer_block_ablate_hook) for point in hook_points]
    
    print(f"NUKE: Ablating ALL attention in {len(layers_to_kill)} layers...")
    
    input_ids = model.to_tokens(prompt_long)
    
    with model.hooks(fwd_hooks=hooks):
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=50, 
            temperature=0, 
            stop_at_eos=True,
            verbose=False
        )
        
    response = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"OUTPUT: {response[:100]}...")
    
    if '"summary":' in response or response.startswith("{"):
        print("RESULT: [PASS] Signal Survived")
    else:
        print("RESULT: [FAIL] Target Destroyed")

print("\nTHE SLEDGEHAMMER EXPERIMENT")

run_block_test("DEEP BLOCK", block_deep)
run_block_test("MID BLOCK", block_mid)

