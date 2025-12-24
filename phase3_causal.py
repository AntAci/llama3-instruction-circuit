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

#ANCHOR_HEADS = [(16, 8), (9, 2), (9, 31), (14, 13), (14, 12)]
ANCHOR_HEADS = [
    (9, 2), (9, 31), (14, 12), (14, 13), (16, 8),
    (20, 5), (20, 6), (24, 10), (24, 11), 
    (26, 6), (26, 26), (27, 26), (29, 5)
]

print(f"Targeting Combined Circuit ({len(ANCHOR_HEADS)} Heads): {ANCHOR_HEADS}")

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

def zero_ablate_hook(value, hook):
    layer = hook.layer()
    heads_to_kill = [h for (l, h) in ANCHOR_HEADS if l == layer]
    if heads_to_kill:
        for h in heads_to_kill:
            value[:, :, h, :] = 0.0
    return value

layers_to_hook = list(set(l for (l, h) in ANCHOR_HEADS))
hook_points = [get_act_name("z", l) for l in layers_to_hook]

def run_test(name, prompt, use_ablation=False):
    print(f"\nRunning: {name}")
    
    hooks = []
    if use_ablation:
        print(f"KILLING HEADS: {ANCHOR_HEADS}")
        hooks = [(point, zero_ablate_hook) for point in hook_points]
    else:
        print("Clean Run")
    
    input_ids = model.to_tokens(prompt)
    
    if hooks:
        with model.hooks(fwd_hooks=hooks):
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=50, 
                temperature=0, 
                top_p=0.0,
                stop_at_eos=True,
                verbose=False
            )
    else:
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=50, 
            temperature=0, 
            top_p=0.0,
            stop_at_eos=True,
            verbose=False
        )
    
    response = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"OUTPUT: {response[:100]}...") 

    is_json = response.strip().startswith("{") or '"summary":' in response
    
    if is_json:
        print("RESULT: [PASS] JSON")
    else:
        print("RESULT: [FAIL] Plain Text")
        
    return is_json

print("\nEXPERIMENT START")

res1 = run_test("Long Range (Clean)", prompt_long, use_ablation=False)
res2 = run_test("Long Range (Ablated)", prompt_long, use_ablation=True)
res3 = run_test("Short Range (Ablated)", prompt_short, use_ablation=True)

print("\nFINAL SCORECARD:")
print(f"Long Range (Clean):   {'[PASS]' if res1 else '[FAIL]'}")
print(f"Long Range (Ablated): {'[SUCCESS] Target Failed' if not res2 else '[FAIL] No Effect'}")
print(f"Short Range (Ablated):{'[PASS] Target' if res3 else '[FAIL] Lobotomized'}")