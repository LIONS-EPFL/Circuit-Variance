from typing import Callable, List 

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens.hook_points import HookedRootModule
from tqdm import tqdm

from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node


def _get_input_lengths(model: HookedRootModule, inputs) -> torch.Tensor:
    """Return last-token positions for model string forward passes."""
    tokenized = model.tokenizer(inputs, padding='longest', return_tensors='pt', add_special_tokens=False)
    return 1 + tokenized.attention_mask.sum(1)

def make_hooks_for_ablated_fwd_pass(graph: Graph, model: HookedRootModule,prune:bool=True, make_clean_cache:bool=False):
    """Build hooks for evaluating a subgraph with edges outside of it patched.

    The returned hooks cache corrupted activations, cache activations produced by
    the mixed forward pass, and replace inputs from edges outside the graph with
    their corrupted-cache values. When `make_clean_cache` is true, clean caching
    hooks are returned as well so the caller can compute clean baseline logits.
    """
    if prune:
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
    fwd_names = {edge.parent.out_hook for edge in graph.edges.values()} # Unique parent output hooks.
    fwd_filter = lambda x: x in fwd_names # Select output hooks before the LogitNode.

    # TransformerLens caches returned by get_caching_hooks are detached.
    corrupted_fwd_cache, corrupted_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)
    mixed_fwd_cache, mixed_fwd_hooks, _ = model.get_caching_hooks(fwd_filter, ) 
    if make_clean_cache:
        clean_fwd_cache, clean_fwd_hooks, _ = model.get_caching_hooks(fwd_filter,) 

    nodes_in_graph = [node for node in graph.nodes.values() if node.in_graph if not isinstance(node, InputNode)]

    # For each node in the graph, construct its input by corrupting incoming
    # edges that are not in the circuit. Attention nodes have separate q/k/v inputs.
    # The corrupted cache stores corrupted activations; the mixed cache stores
    # activations computed by preceding nodes in the current forward pass.
    def make_input_construction_hook(node: Node, qkv=None):
        def input_construction_hook(activations, hook):
            activations = activations.clone() # Avoid in-place modification of hook inputs.
            for edge in node.parent_edges: # Filter manually because edges are stored at node level, not kqv level.
                if edge.qkv != qkv: # For MLP edges, both values are None.
                    continue

                parent:Node = edge.parent 
                if not edge.in_graph: 
                    activations[edge.index] -= mixed_fwd_cache[parent.out_hook][parent.index] # The index selects the attention head.
                    activations[edge.index] += corrupted_fwd_cache[parent.out_hook][parent.index]
            return activations
        return input_construction_hook

    # Input construction hooks are attached to input hooks, while mixed-cache
    # hooks are attached to output hooks. This ordering ensures the mixed cache
    # is populated before downstream inputs are reconstructed.
    input_construction_hooks = []
    for node in nodes_in_graph:
        if isinstance(node, InputNode):
            pass
        elif isinstance(node, LogitNode) or isinstance(node, MLPNode):
            input_construction_hooks.append((node.in_hook, make_input_construction_hook(node)))
        elif isinstance(node, AttentionNode):
            for i, letter in enumerate('qkv'):
                input_construction_hooks.append((node.qkv_inputs[i], make_input_construction_hook(node, qkv=letter)))
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")
    if not make_clean_cache:
        return (corrupted_fwd_hooks, corrupted_fwd_cache, mixed_fwd_hooks, mixed_fwd_cache,input_construction_hooks,)
    else:
        return (corrupted_fwd_hooks, corrupted_fwd_cache, mixed_fwd_hooks, mixed_fwd_cache, clean_fwd_hooks, clean_fwd_cache,input_construction_hooks,)


def evaluate_graph(model: HookedRootModule, graph: Graph, dataloader: DataLoader, metrics: List[Callable[[Tensor], Tensor]], prune:bool=True, quiet=False):
    """Evaluate a graph as an ablated circuit over a dataloader.

    The graph is interpreted as the circuit to keep: edges outside the graph are
    patched with corrupted activations during the mixed forward pass. Each metric
    is evaluated on the resulting logits against the corrupted baseline logits.
    Returns one concatenated tensor per metric, or a single tensor when `metrics`
    is passed as a single callable.
    """

    empty_circuit = not graph.nodes['logits'].in_graph

    corrupted_fwd_hooks, _, mixed_fwd_hooks, _, clean_fwd_hooks, _,input_construction_hooks =\
          make_hooks_for_ablated_fwd_pass(graph=graph, model=model,prune=prune,make_clean_cache=True)

    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results_ls = [[] for _ in metrics]
    
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        input_lengths = _get_input_lengths(model, clean)
        with torch.inference_mode():
            
            with model.hooks(clean_fwd_hooks):
                clean_logits = model(clean)

            with model.hooks(corrupted_fwd_hooks): # registering the corrupted module output
                corrupted_logits = model(corrupted)

            with model.hooks(mixed_fwd_hooks + input_construction_hooks):
                if empty_circuit:
                    logits = model(corrupted)
                else:
                    logits = model(clean)

        for i, metric in enumerate(metrics):
            r = metric(logits, corrupted_logits, input_lengths, label,)
            r = r.cpu()
            assert len(r.size()) > 0, "You probably reduced the metrics for each batch to a scalar."
            results_ls[i].append(r)

    results_ls= [torch.cat(rs) for rs in results_ls]

    if not metrics_list:
        results_ls = results_ls[0]
    
    return results_ls 

def evaluate_baseline(model: HookedRootModule, dataloader:DataLoader, metrics: List[Callable[[Tensor], Tensor]], run_corrupted=False):
    """Evaluate baseline metric values over a dataloader.

    By default, metrics are evaluated on clean logits against corrupted baseline
    logits. When `run_corrupted` is true, metrics are evaluated on corrupted logits
    against themselves, which gives the corrupted baseline for each metric. Returns
    one concatenated tensor per metric, or a single tensor when `metrics` is passed
    as a single callable.
    """
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    
    results = [[] for _ in metrics]
    for clean, corrupted, label in tqdm(dataloader):
        input_lengths = _get_input_lengths(model, clean)
        with torch.inference_mode():
            corrupted_logits = model(corrupted)
            logits = model(clean)
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, corrupted_logits, input_lengths, label).cpu() # KL is zero for identical logits.
            else:
                r = metric(logits, corrupted_logits, input_lengths, label).cpu() # Non-KL metrics ignore the second logits argument.
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results] # Concatenate batched metric values.
    if not metrics_list:
        results = results[0]
    return results
