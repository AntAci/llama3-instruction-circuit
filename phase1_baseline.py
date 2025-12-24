import torch
from transformer_lens import HookedTransformer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda"

print(f"Loading {MODEL_ID} on {device}...")

try:
    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        device=device,
        dtype=torch.float16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        low_cpu_mem_usage=True 
    )
    print(f"Model loaded successfully!")

system_inst = 'You are a strict JSON formatting assistant. No matter the content, you must output your final answer as valid JSON with the key "summary".'

distractor_text = """
Soil erosion is the displacement of the upper layer of soil; it is a form of soil degradation. 
This natural process is caused by the dynamic activity of erosive agents, that is, water, ice, 
snow, air, plants, animals, and humans. In accordance with these agents, erosion is sometimes 
divided into water erosion, glacial erosion, snow erosion, wind erosion, zoogenic erosion 
and anthropogenic erosion. Soil erosion may be a slow process that continues relatively unnoticed, 
or may occur at an alarming rate causing serious loss of topsoil. The loss of soil from farmland 
may be reflected in reduced crop production potential, lower surface water quality and damaged drainage networks.
""" * 20

user_query = "Summarize the text above."

prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{distractor_text}

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

print(f"\nRunning generation (Prompt Length: {len(model.to_tokens(prompt)[0])} tokens)...")

output = model.generate(
    prompt, 
    max_new_tokens=100, 
    temperature=0
)

print("\nMODEL OUTPUT:")
print(output)

