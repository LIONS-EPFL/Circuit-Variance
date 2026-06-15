import argparse
import json

import pandas as pd
from transformers import AutoTokenizer

from ioi_dataset import IOIDataset
from ceap.utils import model2family


def parse_args():
    parser = argparse.ArgumentParser(description="Generate deterministically shuffled IOI datasets.")
    parser.add_argument("--model-name", default="gpt2", help="Model identifier passed to AutoTokenizer.")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10000,
        help="Number of prompts to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset generation and shuffling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional override for the output CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = IOIDataset("mixed", N=args.num_examples, tokenizer=tokenizer, seed=args.seed)

    abc_dataset = ds.gen_flipped_prompts(("S2", "RAND"))
    abb_dataset = ds.gen_flipped_prompts(("S2", "IO"))

    d = {
        "clean": [],
        "corrupted": [],
        "corrupted_hard": [],
        "correct_idx": [],
        "incorrect_idx": [],
        "template_meta": [],
        "template_family": [],
        "template_idx": [],
        "template_text": [],
        "template_label": [],
    }
    for i in range(len(ds)):
        clean = " ".join(ds.sentences[i].split()[:-1])
        corrupted = " ".join(abc_dataset.sentences[i].split()[:-1])
        corrupted_hard = " ".join(abb_dataset.sentences[i].split()[:-1])
        correct = ds.toks[i, ds.word_idx["IO"][i]].item()
        incorrect = ds.toks[i, ds.word_idx["S"][i]].item()
        prompt_meta = ds.ioi_prompts[i]
        template_idx = int(prompt_meta["TEMPLATE_IDX"])
        template_family = prompt_meta["TEMPLATE_FAMILY"]
        template_text = prompt_meta["TEMPLATE_TEXT"]
        template_label = prompt_meta["TEMPLATE_LABEL"]
        template_meta = json.dumps(
            {
                "family": template_family,
                "template_idx": template_idx,
                "template": template_text,
            },
            ensure_ascii=False,
        )
        d["clean"].append(clean)
        d["corrupted"].append(corrupted)
        d["corrupted_hard"].append(corrupted_hard)
        d["correct_idx"].append(correct)
        d["incorrect_idx"].append(incorrect)
        d["template_meta"].append(template_meta)
        d["template_family"].append(template_family)
        d["template_idx"].append(template_idx)
        d["template_text"].append(template_text)
        d["template_label"].append(template_label)

    df = pd.DataFrame.from_dict(d)
    sample_kwargs = {"frac": 1}
    if args.seed is not None:
        sample_kwargs["random_state"] = args.seed
    df = df.sample(**sample_kwargs)

    output_path = args.output or f"{model2family(args.model_name)}.csv"
    df.to_csv(output_path)


if __name__ == "__main__":
    main()
