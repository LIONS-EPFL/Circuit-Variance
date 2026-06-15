from typing import Callable, Literal, Optional, Union
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens.hook_points import HookedRootModule
from tqdm import tqdm
from einops import einsum
import warnings

from .graph import Graph, InputNode, LogitNode, AttentionNode


def get_npos_input_lengths(model, inputs):
    tokenized = model.tokenizer(inputs, padding='longest', return_tensors='pt', add_special_tokens=False)
    n_pos = 1 + tokenized.attention_mask.size(1) # attention mask is a binary mask  
    input_lengths = 1 + tokenized.attention_mask.sum(1)
    return n_pos, input_lengths

def make_hooks_and_matrices(
    model: HookedRootModule,
    graph: Graph,
    batch_size: int,
    n_pos: int,
    scores,
    total_steps: int,
    ig_style: Literal["ceap", "eap", "eap-ig"] = "ceap",):
    """Build activation/gradient hooks and the activation buffer used for edge scoring.

    The returned hooks update `scores` in place during backward passes. For CEAP, the
    activation buffer stores previous/current integrated-gradient activations so each
    step contributes only its activation delta. For EAP/EAP-IG, the buffer stores the
    clean-minus-corrupted activation difference and divides scores by `total_steps`.
    """
    if ig_style == "ceap": # CEAP and original EAP-IG have different conductance accumulation rules.
        activation_prev=torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
        activation_curr=torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

        processed_attn_layers = set() 
        fwd_hooks = []
        bwd_hooks = []
        
        def activation_hook(index, activations, hook, ):
            """Update activation_prev and activation_curr."""
            acts = activations.detach()
            try:
                activation_prev[:, :, index] = activation_curr[:, :, index]
                activation_curr[:, :, index] = acts
            except RuntimeError as e:
                print(hook.name, activation_prev[:, :, index].size(), activation_curr[:, :, index].size(), acts.size())
                raise e

        def gradient_hook(fwd_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook, ):

            grads = gradients.detach()
            try:
                if isinstance(fwd_index, slice):
                    fwd_index = fwd_index.start # several heads might get grouped together in a slice
                if grads.ndim == 3:
                    grads = grads.unsqueeze(2)
                activation_delta=activation_curr[:, :, :fwd_index] - activation_prev[:, :, :fwd_index]
                s = einsum(activation_delta, grads, 'batch pos forward hidden, batch pos backward hidden -> forward backward')
                s = s.squeeze(1)
                # below is the integration
                scores[:fwd_index, bwd_index] += s # the score is computed for all the forward nodes up to fwd_index, but probably not all of them are instantiated by an edge. But it's OK. We are not accessing the scores that does not linked to edges anyway.
            except RuntimeError as e:
                print(hook.name, activation_delta.size(), grads.size())
                raise e
    else: 
        activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

        processed_attn_layers = set()
        fwd_hooks_subtract= []
        fwd_hooks_add= []
        bwd_hooks = []
        
        def activation_hook(index, activations, hook, add:bool=True):
            acts = activations.detach()
            if not add:
                acts = -acts
            try:
                activation_difference[:, :, index] += acts
            except RuntimeError as e:
                print(hook.name, activation_difference[:, :, index].size(), acts.size())
                raise e
        
        def gradient_hook(fwd_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
            grads = gradients.detach()
            try:
                if isinstance(fwd_index, slice):
                    fwd_index = fwd_index.start # several heads might get grouped together in a slice
                if grads.ndim == 3:
                    grads = grads.unsqueeze(2)
                s = einsum(activation_difference[:, :, :fwd_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
                s = s.squeeze(1)
                scores[:fwd_index, bwd_index] += s/total_steps # the score is computed for all the forward nodes up to fwd_index, but probably not all of them are instantiated by an edge. But it's OK. We are not accessing the scores that does not linked to edges anyway.
            except RuntimeError as e:
                print(hook.name, activation_difference.size(), grads.size())
                raise e 
    
    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode): # For each layer, only do it once, thus only registering for head 0 for attention layer. 
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        fwd_index =  graph.forward_index(node)

        # out_hooks
        if not isinstance(node, LogitNode): # the following hook should be invoked every forward pass. Not just the first and last one.
            if ig_style == "ceap":
                fwd_hooks.append((node.out_hook, partial(activation_hook, fwd_index)))
            else:
                fwd_hooks_subtract.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
                fwd_hooks_add.append((node.out_hook, partial(activation_hook, fwd_index, )))
        # in_hooks
        if not isinstance(node, InputNode):
            if isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, fwd_index, bwd_index)))
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append((node.in_hook, partial(gradient_hook, fwd_index, bwd_index)))
    if ig_style == "ceap": return (fwd_hooks, bwd_hooks), activation_curr 
    else:  return (fwd_hooks_subtract, fwd_hooks_add, bwd_hooks), activation_difference


def get_scores(model: HookedRootModule, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],\
                            ig_style: Literal["ceap", "eap", "eap-ig"] = "ceap", steps=30, quiet=False,):
    """Compute attribution scores for every edge in `graph` over `dataloader`.

    For each clean/corrupted batch, this builds the hooks needed by the selected
    attribution style, interpolates the input activation from corrupted to clean,
    backpropagates `metric`, and accumulates the resulting edge scores. The final
    score matrix has shape `(graph.n_forward, graph.n_backward)` and is averaged
    over the number of datapoints.
    """
    if ig_style not in {"ceap", "eap", "eap-ig"}:
        raise ValueError(f"ig_style not supported! It should be eap, ceap, or eap-ig, got {ig_style!r}.")

    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        n_pos, input_lengths = get_npos_input_lengths(model, clean)
        if ig_style == "ceap":
            (fwd_hooks,  bwd_hooks), activation_curr= make_hooks_and_matrices(model=model, graph=graph, batch_size=batch_size, n_pos=n_pos, scores=scores,\
                                                                                            total_steps=steps, ig_style=ig_style, )
        else:
            (fwd_hooks_subtract, fwd_hooks_add, bwd_hooks), activation_difference = make_hooks_and_matrices(model=model, graph=graph, batch_size=batch_size, n_pos=n_pos, scores=scores,\
                                                                                            total_steps=steps, ig_style=ig_style, )
        if ig_style == "ceap": 
            with torch.inference_mode():
                with model.hooks(fwd_hooks=fwd_hooks):
                    _ = model(clean)
                input_activations_clean = activation_curr[:, :, graph.forward_index(graph.nodes['input'])].clone() 
                # Reset before recording corrupted activations.
                activation_curr.zero_()
                # Initialize activation_curr with the corrupted input while activation_prev remains zero.
                with model.hooks(fwd_hooks=fwd_hooks): 
                    corrupted_logits = model(corrupted)
                input_activations_corrupted = activation_curr[:, :, graph.forward_index(graph.nodes['input'])].clone()
        else:
            with torch.inference_mode(): 
                with model.hooks(fwd_hooks=fwd_hooks_add):
                    _ = model(clean)
                input_activations_clean = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()
                with model.hooks(fwd_hooks=fwd_hooks_subtract):
                    corrupted_logits = model(corrupted)
                input_activations_corrupted = input_activations_clean - activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()
        
        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) 
                new_input.requires_grad = True 
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            fwd_hooks_all =[(graph.nodes['input'].out_hook, input_interpolation_hook(step))] 
            if ig_style == "ceap": fwd_hooks_all += fwd_hooks # Update previous/current activation buffers. 
            bwd_hooks_all = bwd_hooks
            with model.hooks(fwd_hooks=fwd_hooks_all, 
                             bwd_hooks=bwd_hooks_all):
                logits = model(clean)
                metric_value = metric(logits, corrupted_logits, input_lengths, label)  # only divergence compute uses the second field.
                model.zero_grad(set_to_none=True) # not necessary, .grad are never used.  
                metric_value.backward()

    scores /= total_items # This assumes metric uses reduction='sum'.

    return scores

def attribute(model: HookedRootModule, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], \
              ig_style: Literal["ceap", "eap", "eap-ig"] = "ceap", integration_steps: Optional[int]=None, quiet=False,):
    """Compute edge scores and write them back to graph."""
 
    ig_style = ig_style.lower()
    if integration_steps is None:
        integration_steps = 30
    if integration_steps == 1:
        ig_style = "eap"
    elif ig_style == "eap":
        warnings.warn("Conducting EAP. Setting integrated gradient steps to 1.")
        ig_style = "eap"
        integration_steps=1
    elif ig_style not in {"ceap", "eap-ig"}:
        raise ValueError(f"ig_style not supported! It should be eap, ceap, or eap-ig, got {ig_style!r}.")
    
    scores = get_scores(model=model,graph=graph,dataloader=dataloader,metric=metric,ig_style=ig_style,steps=integration_steps,quiet=quiet,)
        
    scores = scores.cpu().numpy()

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        # Update in place so layer-wise edge scores stay synchronized with edge scores.
        edge.score[...] = scores[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)]
