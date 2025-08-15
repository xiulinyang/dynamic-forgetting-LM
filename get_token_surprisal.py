import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import forgetting_transformer.model  # registers classes
import forgetting_transformer.tokenizer
import argparse
from tqdm import tqdm
from pathlib import Path
parser = argparse.ArgumentParser(
        description="Compute token surprisal for a given text."
    )

parser.add_argument(
    "--model",
    type=str,
    default="xiulinyang/forgetting_transformer",
    help="Hugging Face model name or path",
)
parser.add_argument(
    "--text_path",
    type=str,
    required=True,
    help="Text to compute surprisals for",
)
parser.add_argument(
    "--save_path",
    type=str,
    help="The path to save the calculated surprisal.",
)

args = parser.parse_args()

@torch.no_grad()
def token_surprisals(text: str, model, tokenizer):
    LN2 = math.log(2.0)

    # 1) Encode *without* auto special tokens; we'll add BOS manually
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device)
    offsets = enc.offset_mapping[0].tolist()

    bos_id = 0
    bos = torch.tensor([[bos_id]], device=input_ids.device)
    input_ids = torch.cat([bos, input_ids], dim=1)        # [1, 1+T]
    attn = torch.cat([torch.ones_like(attn[:, :1]), attn], dim=1)
    offsets = [(0, 0)] + offsets

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn)

    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    target_mask = attn[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    tok_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    tok_bits = (-tok_logp / LN2).squeeze(0)


    shifted_offsets = offsets[1:]
    valid = target_mask.squeeze(0).bool().cpu()
    real = torch.tensor([s != e for (s, e) in shifted_offsets], dtype=torch.bool)
    keep = (valid & real)

    kept_bits = tok_bits[keep.to(tok_bits.device)]
    kept_ids = targets.squeeze(0)[keep.to(targets.device)].tolist()
    kept_tokens = tokenizer.convert_ids_to_tokens(kept_ids)
    kept_offsets = [o for o, k in zip(shifted_offsets, keep.tolist()) if k]

    cleaned_tokens = [tokenizer.convert_tokens_to_string([x]) for x in kept_tokens]
    return {
        "tokens": cleaned_tokens,
        "token_bits": kept_bits.tolist(),
        "sentence_bits": kept_bits.sum().item(),
        "bits_per_token": kept_bits.mean().item(),
        "offsets": kept_offsets,
        "text": text,
    }

def write_surprisal(input_path, output_path, model, tokenizer):
    texts = Path(input_path).read_text().strip()
    sents = []
    for chunk in texts.split("!ARTICLE"):
        sents.extend(line for line in chunk.splitlines() if line.strip())
    with open(output_path, "w") as f:
        f.write(f'word\ttotsurp\n')
        for sent in tqdm(sents):
            surprisal_info = token_surprisals(sent, model, tokenizer)
            for tok, surp in zip(surprisal_info['tokens'], surprisal_info['token_bits']):
                f.write(f'{tok}\t{surp}\n')
                print(tok, surp)


def main():
    model = args.model
    input_path = args.text_path
    output_path = args.save_path
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    write_surprisal(input_path, output_path, model)


if __name__ == "__main__":
    main()

