#%%
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from greater_than_dataset import YearDataset, get_valid_years
from ceap.utils import model2family
#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Generate deterministically shuffled greater-than datasets.")
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


def create_dataset(model_name, num_examples, seed=None, output_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = YearDataset(
        get_valid_years(tokenizer, 1100, 1800),
        num_examples,
        Path("potential_nouns.txt"),
        tokenizer,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    random_order = rng.permutation(num_examples)
    def apply_order(xs):
        return [xs[i] for i in random_order]
    clean_sentences = apply_order(ds.good_sentences)
    correct_idx = apply_order(ds.years_YY.tolist())
    template_words = []
    for sentence in clean_sentences:
        parts = sentence.split()
        if len(parts) < 2:
            raise ValueError(f"Expected at least two words in sentence: {sentence}")
        template_words.append(parts[1])
    select_tens_digit = lambda x: x//10
    template_label = [select_tens_digit(idx) for idx in correct_idx ]
    d = {
        "clean": clean_sentences,
        "corrupted": apply_order(ds.bad_sentences),
        "correct_idx": correct_idx,
        "template_label": template_label,
    }

    df = pd.DataFrame.from_dict(d)
    sample_kwargs = {"frac": 1}
    if seed is not None:
        sample_kwargs["random_state"] = seed
    df = df.sample(**sample_kwargs)
    output_path = output_path or f"{model2family(model_name)}.csv"
    df.to_csv(output_path, index=False)

#%%
if __name__ == '__main__':
    args = parse_args()
    create_dataset(args.model_name, args.num_examples, seed=args.seed, output_path=args.output)
# %%
