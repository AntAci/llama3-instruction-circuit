import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained(
    MODEL_ID, device=device, dtype=torch.float16, low_cpu_mem_usage=True, default_prepend_bos=True
)

GENERALISTS = [2, 3, 6, 20, 21, 23, 24, 26, 27, 29, 30, 31]
SPECIALISTS = [4, 5, 14, 17]

FULL_TRANSLATION_CIRCUIT = [(13, h) for h in list(set(GENERALISTS + SPECIALISTS))]

print(f"Targeting Full Translation Circuit ({len(FULL_TRANSLATION_CIRCUIT)} Heads)")

system_inst = 'You are a professional translator. You must translate the user text into French.'
distractor_text = "The weather today is beautiful and sunny... " * 50 
user_query = "Translate the text above."

prompt_translation = f"""<|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{distractor_text}

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def noise_ablate_hook(value, hook):
    layer = hook.layer()
    heads_to_jam = [h for (l, h) in FULL_TRANSLATION_CIRCUIT if l == layer]
    mean = value.mean()
    std = value.std()
    noise_block = torch.randn_like(value) * std + mean
    for h in heads_to_jam:
        value[:, :, h, :] = noise_block[:, :, h, :]
    return value

print("\nFINAL BOSS: KILLING THE SPECIALISTS")

hooks = [(get_act_name("z", 13), noise_ablate_hook)]

print("Running Translation Task (With FULL Circuit Jammed)...")
with model.hooks(fwd_hooks=hooks):
    out = model.generate(model.to_tokens(prompt_translation), max_new_tokens=60, temperature=0, stop_at_eos=True, verbose=False)

response = model.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
print(f"\nOUTPUT: {response[:100]}...\n")

french_markers = [" le ", " la ", " les ", " est ", " c'est "]
is_french = any(m in response.lower() for m in french_markers)

if not is_french:
    print("[VICTORY] The Translation Circuit is broken.")
else:
    print("[WARNING] Surivived")

