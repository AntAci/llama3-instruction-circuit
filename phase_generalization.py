import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

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

system_inst = 'You are a professional translator. You must translate the user text into French.'
distractor_text = "The weather today is beautiful and sunny... " * 50 
user_query = "Translate the text above."

prompt_long = f"""<|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{distractor_text}

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def noise_ablate_hook(value, hook):
    layer = hook.layer()
    heads_to_jam = [h for (l, h) in L13_HIT_SQUAD if l == layer]
    mean = value.mean()
    std = value.std()
    noise_block = torch.randn_like(value) * std + mean
    for h in heads_to_jam:
        value[:, :, h, :] = noise_block[:, :, h, :]
    return value

print("\nGENERALIZATION TEST")

hooks = [(get_act_name("z", 13), noise_ablate_hook)]

print("Running Translation Task (With L13 Jammed)...")
with model.hooks(fwd_hooks=hooks):
    out = model.generate(model.to_tokens(prompt_long), max_new_tokens=60, temperature=0, stop_at_eos=True, verbose=False)

response = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()

print(f"\nOUTPUT: {response[:100]}...\n")

french_markers = [" le ", " la ", " les ", " est ", " c'est ", " il ", " nous "]
is_french = any(m in response.lower() for m in french_markers)

if not is_french:
    print("[SUCCESS]")
else:
    print("[FAILED]")

