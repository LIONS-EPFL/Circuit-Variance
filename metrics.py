"""Task metrics used by the faithfulness experiments.

This module centralizes the scalar objectives used for attribution, circuit
selection, and circuit evaluation. The public entry point is ``get_metric``,
which returns a callable with the common signature expected by the experiment
drivers:

    metric(clean_logits, corrupted_logits, input_length, labels, ...)

The implemented metric families are:

- KL/JS divergence between clean and corrupted next-token distributions;
- logit-difference and probability-difference metrics for two-token labels;
- task-specific variants for SVA, greater-than, else-elif, and token-set tasks.

All metrics select the logits at each prompt's final input position, support
``mean``/``sum``/``none`` reductions, and can optionally return the relevant
logits needed by the CEAP/EAP attribution code.
"""

from typing import Optional, List, Union, Literal, Tuple
from functools import partial 

import pandas as pd
import torch 
from torch.nn.functional import kl_div
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

def get_metric(metric_name: str, task: str, tokenizer:Optional[PreTrainedTokenizer]=None, model: Optional[HookedTransformer]=None):
    if metric_name == 'kl_divergence' or metric_name == 'kl':
        return partial(divergence, divergence_type='kl')
    elif metric_name == 'js_divergence' or metric_name == 'js':
        return partial(divergence, divergence_type='js')
    elif metric_name == 'logit_diff' or metric_name == 'prob_diff':
        prob = (metric_name == 'prob_diff')
        if 'greater-than' in task:
            if tokenizer is None:
                if model is None:
                    raise ValueError("Either tokenizer or model must be set for greater-than and prob / logit diff")
                else:
                    tokenizer = model.tokenizer
            logit_diff_fn = get_logit_diff_greater_than(tokenizer)
        elif 'hypernymy' in task: 
            logit_diff_fn = logit_diff_token_sets
        elif task == 'else_elif':
            if tokenizer is None:
                if model is None:
                    raise ValueError("Either tokenizer or model must be set for else_elif and prob / logit diff")
                tokenizer = model.tokenizer
            logit_diff_fn = get_logit_diff_else_elif(tokenizer)
        elif task == 'sva':
            if model is None:
                raise ValueError("model must be set for sva and prob / logit diff")
            logit_diff_fn = get_logit_diff_sva(model)
        else:
            logit_diff_fn = logit_diff
        return partial(logit_diff_fn, prob=prob)
    else: 
        raise ValueError(f"got bad metric_name: {metric_name}")

def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    """Get the logits corresponding to the last token in each sequence."""
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    input_length = input_length.to(logits.device)

    if torch.any(input_length < 1) or torch.any(input_length > logits.size(1)):
        min_len = int(input_length.min().detach().cpu())
        max_len = int(input_length.max().detach().cpu())
        raise ValueError(
            f"input_length must be in [1, {logits.size(1)}], got range [{min_len}, {max_len}]."
        )

    logits = logits[idx, input_length - 1]
    return logits

def js_div(p: torch.tensor, q: torch.tensor):
    # maybe the mean at the end should be changed to sum, but this is used in the original repo. It does not really matter as well.
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return 0.5 * (kl_div(m, p.log(), log_target=True, reduction='none').mean(-1) + kl_div(m, q.log(), log_target=True, reduction='none').mean(-1))

def divergence(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, divergence_type: Union[Literal['kl'], Literal['js']]='kl', reduction='mean', loss=True, return_relevant_logits=False):
    clean_logits = get_logit_positions(clean_logits, input_length)
    corrupted_logits = get_logit_positions(corrupted_logits, input_length)

    clean_probs = torch.softmax(clean_logits, dim=-1)
    corrupted_probs = torch.softmax(corrupted_logits, dim=-1) # (B,V)

    if divergence_type == 'kl':
        results = kl_div(clean_probs.log(), corrupted_probs.log(), log_target=True, reduction='none').sum(-1) 
    elif divergence_type == 'js':
        results = js_div(clean_probs, corrupted_probs)
    else: 
        raise ValueError(f"Expected divergence_type of 'kl' or 'js', but got '{divergence_type}'")
    results = result_reduction(results, reduction)
    if return_relevant_logits:
        clean_relevant_logits = clean_logits
        corrupted_relevant_logits = corrupted_logits
        assert clean_relevant_logits.ndim == 2 and corrupted_relevant_logits.ndim == 2
        relevant_lengths = torch.full(
            (clean_relevant_logits.size(0),),
            clean_relevant_logits.size(1),
            device=clean_relevant_logits.device,
            dtype=torch.long,
        )
        return results, clean_relevant_logits, corrupted_relevant_logits, relevant_lengths
    return results

def logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, reduction='mean', prob=False, loss=False,\
               return_relevant_logits = False):
    clean_logits = get_logit_positions(clean_logits, input_length) # get the logits corresponding to the last position
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
    labels_device = labels.to(cleans.device)
    if torch.any(labels_device < 0) or torch.any(labels_device >= cleans.size(-1)):
        min_label = int(labels_device.min().detach().cpu())
        max_label = int(labels_device.max().detach().cpu())
        raise ValueError(
            f"labels must be in [0, {cleans.size(-1) - 1}], got range [{min_label}, {max_label}]."
        )
    good_bad = torch.gather(cleans, -1, labels_device) # cleans: (B,vocab_size); labels: (B, 2)
    results = good_bad[:, 0] - good_bad[:, 1] # good minus bad (B,)

    if loss:
        results = -results
    if return_relevant_logits:
        corrupted_logits = get_logit_positions(corrupted_logits, input_length)
        clean_relevant_logits = torch.gather(clean_logits, -1, labels_device)
        corrupted_relevant_logits = torch.gather(corrupted_logits, -1, labels_device)
        assert clean_relevant_logits.ndim == 2 and corrupted_relevant_logits.ndim == 2
        relevant_lengths = torch.full(
            (clean_relevant_logits.size(0),),
            clean_relevant_logits.size(1),
            device=clean_relevant_logits.device,
            dtype=torch.long,
        )
        return results, clean_relevant_logits, corrupted_relevant_logits, relevant_lengths
    results = result_reduction(results, reduction)
    return results

def logit_diff_token_sets(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: List[torch.Tensor], reduction='mean', prob=False,\
                         loss=False, return_relevant_logits = False):
    clean_logits = get_logit_positions(clean_logits, input_length)
    if return_relevant_logits:
        corrupted_logits = get_logit_positions(corrupted_logits, input_length)
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits

    results = []
    relevant_clean_logits_per_sample = [] if return_relevant_logits else None
    relevant_corrupted_logits_per_sample = [] if return_relevant_logits else None
    relevant_counts = [] if return_relevant_logits else None
    for i, (ls,corrupted_ls) in enumerate(labels):
        ls_device = ls.to(cleans.device)
        corrupted_ls_device = corrupted_ls.to(cleans.device)
        r = cleans[i][ls_device].sum() - cleans[i][corrupted_ls_device].sum()
        results.append(r)
        if return_relevant_logits:
            relevant_indices = torch.cat((ls_device, corrupted_ls_device), dim=0)
            sample_clean_relevant_logits = clean_logits[i][relevant_indices]
            sample_corrupted_relevant_logits = corrupted_logits[i][relevant_indices]
            relevant_clean_logits_per_sample.append(sample_clean_relevant_logits)
            relevant_corrupted_logits_per_sample.append(sample_corrupted_relevant_logits)
            relevant_counts.append(sample_clean_relevant_logits.numel())
    results = torch.stack(results)
    
    if loss:
        results = -results
    results = result_reduction(results, reduction)
    if return_relevant_logits:
        batch_size = clean_logits.size(0)
        max_relevant = max(relevant_counts) if relevant_counts else 0
        # below there is some zero padding since the relevant counts might be different for different samples
        clean_relevant_logits = torch.zeros(
            (batch_size, max_relevant),
            device=clean_logits.device,
            dtype=clean_logits.dtype,
        )
        corrupted_relevant_logits = torch.zeros(
            (batch_size, max_relevant),
            device=clean_logits.device,
            dtype=clean_logits.dtype,
        )
        for i, (sample_clean_logits, sample_corrupted_logits) in enumerate(zip(relevant_clean_logits_per_sample, relevant_corrupted_logits_per_sample)):
            n_relevant = sample_clean_logits.numel()
            if n_relevant > 0:
                clean_relevant_logits[i, :n_relevant] = sample_clean_logits
                corrupted_relevant_logits[i, :n_relevant] = sample_corrupted_logits
        relevant_lengths = torch.tensor(relevant_counts, device=clean_logits.device, dtype=torch.long)
        return results, clean_relevant_logits, corrupted_relevant_logits, relevant_lengths
    return results


logit_diff_hypernymy = logit_diff_token_sets

def get_year_indices(tokenizer: PreTrainedTokenizer):
    return torch.tensor([tokenizer(f'{year:02d}').input_ids[0] for year in range(100)]) 

def get_logit_diff_greater_than(tokenizer: PreTrainedTokenizer):
    year_indices = get_year_indices(tokenizer) 
    def logit_diff_greater_than(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, reduction='mean', prob=False, loss=False, return_relevant_logits = False):
        clean_logits = get_logit_positions(clean_logits, input_length)
        if return_relevant_logits:
            corrupted_logits = get_logit_positions(corrupted_logits, input_length)
        cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
        cleans = cleans[:, year_indices] # Choose the positions corresponding to 00-99

        results = []
        if prob:
            for prob, year in zip(cleans, labels):
                results.append(prob[year + 1 :].sum() - prob[: year + 1].sum())
        else:
            for logit, year in zip(cleans, labels):
                results.append(logit[year + 1 :].mean() - logit[: year + 1].mean())

        results = torch.stack(results)
        if loss:
            results = -results
        results = result_reduction(results, reduction)

        if return_relevant_logits:
            clean_relevant_logits = clean_logits[:,year_indices]
            corrupted_relevant_logits = corrupted_logits[:,year_indices]
            assert clean_relevant_logits.ndim == 2 and corrupted_relevant_logits.ndim == 2
            relevant_lengths = torch.full(
                (clean_relevant_logits.size(0),),
                clean_relevant_logits.size(1),
                device=clean_relevant_logits.device,
                dtype=torch.long,
            )
            return results, clean_relevant_logits, corrupted_relevant_logits, relevant_lengths
        return results
    return logit_diff_greater_than


def _encode_single_token(tokenizer, text: str) -> int:
    if isinstance(tokenizer, PreTrainedTokenizer):
        toks = tokenizer(text, add_special_tokens=False)["input_ids"]
    elif hasattr(tokenizer, "encode"):
        toks = tokenizer.encode(text)
    else:
        raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)!r}")
    if len(toks) != 1:
        raise ValueError(f"Expected {text!r} to be a single token, got {toks}.")
    return toks[0]


def get_logit_diff_else_elif(tokenizer) -> torch.Tensor:
    colon_newline_id = _encode_single_token(tokenizer, ":\n")

    def else_elif_logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, reduction='mean', prob=False, loss=False, return_relevant_logits = False):
        clean_logits = get_logit_positions(clean_logits, input_length)
        if return_relevant_logits:
            corrupted_logits = get_logit_positions(corrupted_logits, input_length)
        cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits

        colon_score = cleans[:, colon_newline_id]
        if prob:
            other_score = 1 - colon_score
        else:
            other_score = (cleans.sum(-1) - colon_score) / (cleans.size(-1) - 1)

        results = torch.where(labels.to(cleans.device) == 1, colon_score - other_score, other_score - colon_score)
        if loss:
            results = -results
        results = result_reduction(results, reduction)

        if return_relevant_logits:
            clean_relevant_logits = clean_logits
            corrupted_relevant_logits = corrupted_logits
            assert clean_relevant_logits.ndim == 2 and corrupted_relevant_logits.ndim == 2
            relevant_lengths = torch.full(
                (clean_relevant_logits.size(0),),
                clean_relevant_logits.size(1),
                device=clean_relevant_logits.device,
                dtype=torch.long,
            )
            return results, clean_relevant_logits, corrupted_relevant_logits, relevant_lengths
        return results

    return else_elif_logit_diff


def get_singular_and_plural(model, strict=False, print_discard_count = False, combined_verb_list_dir = "data/sva/combined_verb_list.csv") -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = model.tokenizer
    tokenizer_length = model.cfg.d_vocab_out

    df: pd.DataFrame = pd.read_csv(combined_verb_list_dir)
    singular = df['sing'].to_list()
    plural = df['plur'].to_list()
    singular_set = set(singular)
    plural_set = set(plural)
    verb_set = singular_set | plural_set
    assert len(singular_set & plural_set) == 0, f"{singular_set & plural_set}"
    singular_indices, plural_indices = [], []
    discarded_count = 0

    for i in range(tokenizer_length):
        token = tokenizer._convert_id_to_token(i)
        if token is not None:
            if token[0] == 'Ġ':
                token = token[1:]
                if token in verb_set:    # only include in the output token in the tokenizer vocab and verb_set
                    # only include token whose plural/singular counterparts are also single-tokened.
                    if token in singular_set:
                        idx = singular.index(token)
                        plural_form = plural[idx]
                        plural_tokenized = tokenizer(f' {plural_form}', add_special_tokens=False)['input_ids']
                        if len(plural_tokenized) == 1 and plural_tokenized[0] != tokenizer.unk_token_id:
                            singular_indices.append(i)
                        elif not strict:
                            singular_indices.append(i)
                        else:
                            discarded_count+=1
                    else:  # token in plural_set:
                        idx = plural.index(token)
                        third_person_present = singular[idx]
                        third_person_present_tokenized = tokenizer(f' {third_person_present}', add_special_tokens=False)['input_ids']
                        if len(third_person_present_tokenized) == 1 and third_person_present_tokenized[0] != tokenizer.unk_token_id:
                            plural_indices.append(i)
                        elif not strict:
                            plural_indices.append(i)
                        else:
                            discarded_count+=1
    if print_discard_count: print(discarded_count)
    return torch.tensor(singular_indices, device=model.cfg.device), torch.tensor(plural_indices, device=model.cfg.device)

def get_logit_diff_sva(model, strict=True,combined_verb_list_dir='data/sva/combined_verb_list.csv') -> torch.Tensor:
    singular_indices, plural_indices = get_singular_and_plural(model, strict=strict, combined_verb_list_dir=combined_verb_list_dir)
    relevant_indices = torch.cat((singular_indices, plural_indices))
    def sva_logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, reduction='mean', prob=False, loss=False, return_relevant_logits = False):
        clean_logits = get_logit_positions(clean_logits, input_length)
        if return_relevant_logits:
            corrupted_logits = get_logit_positions(corrupted_logits, input_length)
        cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
        
        if prob:
            singular = cleans[:, singular_indices].sum(-1)
            plural = cleans[:, plural_indices].sum(-1)
        else: # Original repo does like this: if logit, then do mean, don't understand why. In hypernymy is the same situation, but they don't take mean for logit_diff there.
            singular = cleans[:, singular_indices].mean(-1)
            plural = cleans[:, plural_indices].mean(-1)

        results = torch.where(labels.to(cleans.device) == 0, singular - plural, plural - singular)
        if loss: 
            results = -results
        results = result_reduction(results, reduction)

        if return_relevant_logits:
            clean_relevant_logits = clean_logits[:, relevant_indices]
            corrupted_relevant_logits = corrupted_logits[:, relevant_indices]
            assert clean_relevant_logits.ndim == 2 and corrupted_relevant_logits.ndim == 2
            relevant_lengths = torch.full(
                (clean_relevant_logits.size(0),),
                clean_relevant_logits.size(1),
                device=clean_relevant_logits.device,
                dtype=torch.long,
            )
            return results, clean_relevant_logits, corrupted_relevant_logits, relevant_lengths
        return results
    return sva_logit_diff

def result_reduction(results, reduction):
    if reduction.lower() == 'mean': return results.mean()
    elif reduction.lower() == 'sum': return results.sum()
    elif reduction.lower() == 'none': return results
    else: raise ValueError(f"Expected reduction of 'mean', 'sum', or 'none', but got '{reduction}'")
