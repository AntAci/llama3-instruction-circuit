import json
import random

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 88

# CLI:
#   python phase15_controls.py --fast
#   python phase15_controls.py --full
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true")
parser.add_argument("--full", action="store_true")
args = parser.parse_args()

if args.fast and args.full:
    raise SystemExit("Pick one: --fast or --full")

if args.fast:
    N_PROMPTS_JSON = 5
    N_PROMPTS_TRANS = 5
    N_RANDOM_SETS = 10
    MAX_NEW_TOKENS = 40
    DISTRACTOR_REPEATS = 30
    KV_PERMUTATIONS = 1000
else:
    N_PROMPTS_JSON = 10
    N_PROMPTS_TRANS = 10
    N_RANDOM_SETS = 30
    MAX_NEW_TOKENS = 60
    DISTRACTOR_REPEATS = 50
    KV_PERMUTATIONS = 5000

L13_HIT_SQUAD = [2, 3, 6, 8, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31]
GENERALISTS = [2, 3, 6, 20, 21, 23, 24, 26, 27, 29, 30, 31]
SPECIALISTS = [4, 5, 14, 17]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Loading {MODEL_ID} on {device}...")
model = HookedTransformer.from_pretrained(
    MODEL_ID,
    device=device,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    default_prepend_bos=True,
)
model.eval()


def noise_jam_heads(layer_target: int, heads_to_jam: list[int]):
    def hook(value, hook):
        if hook.layer() != layer_target:
            return value
        mean = value.mean()
        std = value.std()
        noise_block = torch.randn_like(value) * std + mean
        for h in heads_to_jam:
            value[:, :, h, :] = noise_block[:, :, h, :]
        return value

    return hook


def decode_new_tokens(output_ids, input_len: int) -> str:
    return model.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def extract_first_json_obj(text: str):
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def check_json_success(text: str) -> bool:
    obj = extract_first_json_obj(text)
    return isinstance(obj, dict) and ("summary" in obj)


def check_french_success(text: str) -> bool:
    t = " " + text.lower() + " "
    markers = [" le ", " la ", " les ", " des ", " est ", " c'est ", " il ", " elle ", " nous ", " vous ", " dans ", " pour "]
    hits = sum(1 for m in markers if m in t)
    has_accent = any(ch in t for ch in "éèêàùçôîï")
    return hits >= 2 or (hits >= 1 and has_accent)


def make_long_distractor(base_sentence: str, repeats: int = 50) -> str:
    return (base_sentence.strip() + " ") * repeats


def make_json_prompt(distractor_base: str) -> str:
    system_inst = 'You are a strict JSON formatting assistant. No matter the content, you must output your final answer as valid JSON with the key "summary".'
    user_query = "Summarize the text above."
    distractor = make_long_distractor(distractor_base, repeats=DISTRACTOR_REPEATS)
    return f"""<|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{distractor}

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def make_translation_prompt(distractor_base: str, system_inst: str | None = None) -> str:
    if system_inst is None:
        system_inst = "You are a professional translator. You must translate the user text into French."
    user_query = "Translate the text above."
    distractor = make_long_distractor(distractor_base, repeats=DISTRACTOR_REPEATS)
    return f"""<|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{distractor}

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


JSON_BASES = [
    "Soil erosion is the displacement of the upper layer of soil.",
    "Climate change affects global weather patterns significantly.",
    "Renewable energy sources are becoming more cost-effective.",
    "Artificial intelligence transforms modern technology rapidly.",
    "Urban planning requires careful consideration of population growth.",
    "Ocean currents regulate Earth's temperature distribution.",
    "Quantum computing promises revolutionary computational power.",
    "Biodiversity conservation protects ecosystem stability.",
    "Space exploration expands human knowledge frontiers.",
    "Sustainable agriculture balances productivity and environmental health.",
    "Neural networks learn complex pattern recognition tasks.",
    "Genetic engineering enables precise biological modifications.",
]

TRANS_BASES = [
    "The weather today is beautiful and sunny.",
    "Technology advances rapidly in modern society.",
    "Education plays a crucial role in personal development.",
    "Healthcare systems require continuous improvement.",
    "Economic growth depends on innovation and investment.",
    "Cultural diversity enriches human experience.",
    "Scientific research drives technological progress.",
    "Environmental protection is essential for future generations.",
    "Social media connects people across the globe.",
    "Literature reflects the complexity of human nature.",
    "Mathematics provides tools for understanding the world.",
    "History teaches valuable lessons about human behavior.",
]


def build_prompt_sets():
    json_prompts = [make_json_prompt(JSON_BASES[i % len(JSON_BASES)]) for i in range(N_PROMPTS_JSON)]
    trans_prompts = [make_translation_prompt(TRANS_BASES[i % len(TRANS_BASES)]) for i in range(N_PROMPTS_TRANS)]
    return json_prompts, trans_prompts


@torch.no_grad()
def eval_mask_on_prompts(layer: int, heads: list[int], json_prompts: list[str], trans_prompts: list[str]):
    hook_fn = noise_jam_heads(layer, heads)
    hooks = [(get_act_name("z", layer), hook_fn)]

    json_passes = 0
    for i, p in enumerate(json_prompts, start=1):
        input_ids = model.to_tokens(p)
        with model.hooks(fwd_hooks=hooks):
            out = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0, stop_at_eos=True, verbose=False)
        resp = decode_new_tokens(out, input_ids.shape[1])
        json_passes += 1 if check_json_success(resp) else 0
        if (i % 2) == 0 or i == len(json_prompts):
            print(f"JSON prompts: {i}/{len(json_prompts)}")

    trans_passes = 0
    for i, p in enumerate(trans_prompts, start=1):
        input_ids = model.to_tokens(p)
        with model.hooks(fwd_hooks=hooks):
            out = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0, stop_at_eos=True, verbose=False)
        resp = decode_new_tokens(out, input_ids.shape[1])
        trans_passes += 1 if check_french_success(resp) else 0
        if (i % 2) == 0 or i == len(trans_prompts):
            print(f"Translation prompts: {i}/{len(trans_prompts)}")

    return json_passes / max(1, len(json_prompts)), trans_passes / max(1, len(trans_prompts))


def percentile_rank(x: float, samples: list[float]) -> float:
    if not samples:
        return float("nan")
    return 100.0 * (sum(1 for s in samples if s <= x) / len(samples))


print("\nPHASE 15: CONTROL EXPERIMENTS")

json_prompts, trans_prompts = build_prompt_sets()

print("\nF1: RANDOM-HEAD CONTROL (Layer 13, 14-head masks)")
print(f"N_RANDOM_SETS={N_RANDOM_SETS}, N_PROMPTS_JSON={N_PROMPTS_JSON}, N_PROMPTS_TRANS={N_PROMPTS_TRANS}")
print(f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}, DISTRACTOR_REPEATS={DISTRACTOR_REPEATS}")

random_json_rates = []
random_trans_rates = []

for i in range(N_RANDOM_SETS):
    heads = random.sample(range(model.cfg.n_heads), k=len(L13_HIT_SQUAD))
    jr, tr = eval_mask_on_prompts(13, heads, json_prompts, trans_prompts)
    random_json_rates.append(jr)
    random_trans_rates.append(tr)
    if (i + 1) % 5 == 0:
        print(f"Random set {i+1}/{N_RANDOM_SETS} done...")

hit_json_rate, hit_trans_rate = eval_mask_on_prompts(13, L13_HIT_SQUAD, json_prompts, trans_prompts)

print("\nRandom sets (Layer 13):")
print(f"JSON success rate: mean={np.mean(random_json_rates):.2%} std={np.std(random_json_rates):.3f}")
print(f"Trans success rate: mean={np.mean(random_trans_rates):.2%} std={np.std(random_trans_rates):.3f}")

print("\nHit Squad (Layer 13):")
print(f"JSON success rate: {hit_json_rate:.2%} (percentile={percentile_rank(hit_json_rate, random_json_rates):.1f})")
print(f"Trans success rate: {hit_trans_rate:.2%} (percentile={percentile_rank(hit_trans_rate, random_trans_rates):.1f})")

print("\nF2: NEARBY-LAYER CONTROL (same head indices, same mask size)")

layer_json_rates = {}
layer_trans_rates = {}
for layer in [12, 13, 14]:
    jr, tr = eval_mask_on_prompts(layer, L13_HIT_SQUAD, json_prompts, trans_prompts)
    layer_json_rates[layer] = jr
    layer_trans_rates[layer] = tr
    print(f"Layer {layer}: JSON={jr:.2%} Translation={tr:.2%}")

print("\nF3: MULTI-PROMPT ROBUSTNESS (Hit Squad, Layer 13)")
print(f"JSON mean over prompts: {hit_json_rate:.2%}")
print(f"Translation mean over prompts: {hit_trans_rate:.2%}")

print("\nF4: PROMPT INTERPOLATION (attention to system tokens)")

interp_systems = [
    ("Pure JSON", 'You are a strict JSON formatting assistant. No matter the content, you must output your final answer as valid JSON with the key "summary".'),
    ("JSON+Bilingual", 'You are a JSON formatting assistant. Output your answer as valid JSON with key "summary". Also provide a French translation.'),
    ("Bilingual", "You are a bilingual assistant. Provide responses in both English and French."),
    ("Translation+JSON", "You are a translator. Translate to French and format as JSON."),
    ("Pure Translation", "You are a professional translator. You must translate the user text into French."),
]

interp_distractor = make_long_distractor(TRANS_BASES[0], repeats=DISTRACTOR_REPEATS)
interp_query = "Translate the text above."

for label, system_inst in interp_systems:
    prompt = f"""<|start_header_id|>system<|end_header_id|>

{system_inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{interp_distractor}

{interp_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    _, cache = model.run_with_cache(prompt, remove_batch_dim=True)
    pattern = cache["blocks.13.attn.hook_pattern"]
    last_token_idx = -1
    sys_slice = slice(0, 25)

    specialist_scores = []
    generalist_scores = []
    for h in range(model.cfg.n_heads):
        s = pattern[h, last_token_idx, sys_slice].sum().item()
        if h in SPECIALISTS:
            specialist_scores.append(s)
        if h in GENERALISTS:
            generalist_scores.append(s)

    avg_spec = float(np.mean(specialist_scores)) if specialist_scores else 0.0
    avg_gen = float(np.mean(generalist_scores)) if generalist_scores else 0.0
    print(f"{label:18s} | specialists={avg_spec:.3f} generalists={avg_gen:.3f}")

print("\nF5: KV-GROUP ENRICHMENT (from model config if available)")

n_heads = int(getattr(model.cfg, "n_heads", 32))
n_kv = getattr(model.cfg, "n_key_value_heads", None)
if n_kv is None:
    n_kv = getattr(model.cfg, "n_kv_heads", None)
if n_kv is None:
    print("KV-group info not available on model.cfg; skipping enrichment test.")
else:
    n_kv = int(n_kv)
    group_size = n_heads // n_kv if n_kv else None
    if not group_size or (n_heads % n_kv != 0):
        print(f"Unexpected KV config: n_heads={n_heads}, n_kv={n_kv}; skipping.")
    else:
        def kv_group(h: int) -> int:
            return h // group_size

        gen_groups = [kv_group(h) for h in GENERALISTS]
        spec_groups = [kv_group(h) for h in SPECIALISTS]
        overlap = set(gen_groups) & set(spec_groups)
        print(f"n_heads={n_heads}, n_kv={n_kv}, heads/group={group_size}")
        print(f"Generalist groups: {sorted(set(gen_groups))}")
        print(f"Specialist groups: {sorted(set(spec_groups))}")
        print(f"Overlap groups: {sorted(overlap)}")

        observed_overlap = len(overlap)
        overlaps = []
        for _ in range(KV_PERMUTATIONS):
            gen = random.sample(range(n_heads), k=len(GENERALISTS))
            spec = random.sample(range(n_heads), k=len(SPECIALISTS))
            overlaps.append(len(set(kv_group(h) for h in gen) & set(kv_group(h) for h in spec)))
        p = (sum(1 for x in overlaps if x >= observed_overlap) + 1) / (len(overlaps) + 1)
        print(f"Observed overlap groups: {observed_overlap}")
        print(f"Permutation p-value (>= overlap): {p:.4f}")

print("\nCONTROL EXPERIMENTS COMPLETE")

