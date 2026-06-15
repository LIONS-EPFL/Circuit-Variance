#%%
import argparse

import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from ceap.utils import model2family
#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Generate deterministically shuffled SVA datasets.")
    parser.add_argument("--model-name", default="gpt2", help="Model identifier passed to AutoTokenizer.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional override for the output CSV path.",
    )
    return parser.parse_args()


def create_dataset(model_name, seed=None, output_path=None): # note that the blimp ... stuff are not included
    ML_files = ['sva_raw_data/ML-subj_rel_all.csv',
    'sva_raw_data/ML-prep_inanim_all.csv',
    'sva_raw_data/ML-obj_rel_within_anim_all.csv',
    'sva_raw_data/ML-obj_rel_no_comp_within_inanim_all.csv',
    'sva_raw_data/ML-sent_comp_all.csv',
    'sva_raw_data/ML-obj_rel_no_comp_across_inanim_all.csv',
    'sva_raw_data/ML-obj_rel_within_inanim_all.csv',
    'sva_raw_data/ML-obj_rel_no_comp_within_anim_all.csv',
    'sva_raw_data/ML-obj_rel_across_anim_all.csv',
    'sva_raw_data/ML-obj_rel_across_inanim_all.csv',
    'sva_raw_data/ML-simple_agrmt_all.csv',
    'sva_raw_data/ML-long_vp_coord_all.csv',
    'sva_raw_data/ML-prep_anim_all.csv',
    'sva_raw_data/ML-vp_coord_all.csv',
    'sva_raw_data/ML-obj_rel_no_comp_across_anim_all.csv']

    model_family = model2family(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dfs = []
    for template_id, ML_file in enumerate(ML_files):
        df = pd.read_csv(ML_file)
        d = {'sentence_singular': df['sentence'][df['label']==0].values,\
             'sentence_plural': df['sentence'][df['label']==1].values,\
             'group':[ML_file.split('/')[-1].split('.')[0]]*(len(df)//2),\
             'template_label': template_id }
        new_df = pd.DataFrame.from_dict(d)
        dfs.append(new_df)

    big_df = pd.concat(dfs)

    # need to make sure that the clean-corrupt pairs are of the same lengths after tokenization.
    same_len = []
    for sing, plur in zip(big_df['sentence_singular'],big_df[ 'sentence_plural']):
        sing_tok = tokenizer(sing, return_tensors='pt')['input_ids'].squeeze()
        plur_tok = tokenizer(plur, return_tensors='pt')['input_ids'].squeeze()
        same_len.append(len(sing_tok) == len(plur_tok))

    same_len = np.array(same_len)
    print(same_len.sum(), '/', len(big_df))

    same_len_df = big_df[same_len]
    template_ids = same_len_df["template_label"].tolist()
    template_label = [f"single_{tid}" for tid in template_ids] + [f"plural_{tid}" for tid in template_ids]
    d2 = {
        "clean": same_len_df["sentence_singular"].tolist() + same_len_df["sentence_plural"].tolist(),
        "corrupted": same_len_df["sentence_plural"].tolist() + same_len_df["sentence_singular"].tolist(),
        "group": same_len_df["group"].tolist() * 2,
        "template_label": template_label,
        "plural": [0] * len(same_len_df) + [1] * len(same_len_df),
    }
    final_df = df.from_dict(d2)
    sample_kwargs = {"frac": 1}
    if seed is not None:
        sample_kwargs["random_state"] = seed
    final_df = final_df.sample(**sample_kwargs)
    
    output_path = output_path or f"{model_family}.csv"
    final_df.to_csv(output_path, index=False)
# %%
if __name__ == '__main__':
    args = parse_args()
    create_dataset(args.model_name, seed=args.seed, output_path=args.output)
# %%
