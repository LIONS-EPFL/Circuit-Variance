#%%
from functools import partial
import ast

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pandas import DataFrame


def collate_EAP(xs, task):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    first_label = labels[0] if labels else None
    is_token_set_task = (
        isinstance(first_label, (list, tuple))
        and len(first_label) == 2
        and all(isinstance(x, torch.Tensor) for x in first_label)
    )
    if not is_token_set_task:
        labels = torch.tensor(labels)
    return clean, corrupted, labels

def make_train_test_dataframe(df: pd.DataFrame, train_ratio: float | None= None, 
                              train_sample_number: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    assert bool(train_ratio is None) + bool(train_sample_number is None) == 1
    total_len = len(df)

    if train_ratio is not None:
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        split_idx = int(total_len * train_ratio)
        if total_len > 1:# In very rare cases, the train and test dataframe might be empty. Need to guard agains that.
            split_idx = min(max(split_idx, 1), total_len - 1)


    if train_sample_number is not None:
        if not 0<train_sample_number<=total_len:
            raise ValueError(f"train_sample_number must be between 0 and {total_len}, got {train_sample_number}")
        split_idx = train_sample_number


    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df

class EAPDataset(Dataset):
    def __init__(self, task:str, dataframe:DataFrame):
        self.df = dataframe

        self.task = task

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = None
        if self.task == 'ioi':
            label = [row['correct_idx'], row['incorrect_idx']]
        elif 'greater-than' in self.task:
            label = row['correct_idx']
        elif 'hypernymy' in self.task:
            answer = torch.tensor(ast.literal_eval(row['answers_idx']))
            corrupted_answer = torch.tensor(ast.literal_eval(row['corrupted_answers_idx']))
            label = [answer, corrupted_answer]
        elif 'single_double_quote' in self.task:
            label = [row['correct_idx'], row['incorrect_idx']]
        elif self.task == 'else_elif':
            label = row['colon_newline']
        elif 'fact-retrieval' in self.task:
            label = [row['country_idx'], row['corrupted_country_idx']]
        elif 'gender' in self.task:
            label = [row['clean_answer_idx'], row['corrupted_answer_idx']]
        elif self.task == 'sva':
            label = row['plural']
        elif self.task == 'colored-objects':
            label = [row['correct_idx'], row['incorrect_idx']]
        elif self.task in {'dummy-easy', 'dummy-medium', 'dummy-hard'}:
            label = 0 
        else:
            raise ValueError(f'Got invalid task: {self.task}')
        return row['clean'], row['corrupted'], label
    
    def to_dataloader(self, batch_size: int): # no shuffling at all.
        return DataLoader(self, batch_size=batch_size, collate_fn=partial(collate_EAP, task=self.task))
