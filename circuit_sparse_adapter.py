"""TransformerLens-compatible adapter for OpenAI circuit-sparsity models.

The released circuit-sparsity checkpoints use a NanoGPT-style architecture with
sparsified attention/MLP internals and a TinyPython tokenizer. This module
provides the small amount of compatibility code needed to use those checkpoints
inside the faithfulness pipeline:

- download and cache released ``csp_*`` model artifacts;
- wrap the TinyPython tokenizer with the subset of the HuggingFace tokenizer API
  used by the project;
- define TransformerLens hook-compatible sparse transformer modules; and
- load released sparse checkpoint weights into that hookable model.

The adapter intentionally implements only the architecture features required by
the released circuit-sparsity models, rather than trying to be a general
TransformerLens replacement.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CIRCUIT_SPARSE_CACHE_DIR = Path.home() / ".cache" / "faithfulness" / "circuit_sparsity"
if "TIKTOKEN_CACHE_DIR" not in os.environ and "DATA_GYM_CACHE_DIR" not in os.environ:
    os.environ["TIKTOKEN_CACHE_DIR"] = str(DEFAULT_CIRCUIT_SPARSE_CACHE_DIR)

from tiktoken import Encoding
from tiktoken.load import read_file_cached
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint, HookedRootModule, NamesFilter

from circuit_sparsity_assets import tinypython as vendored_tinypython


MODEL_BASE_URL = "https://openaipublic.blob.core.windows.net/circuit-sparsity/models"
SPECIAL_EOT = 2047
ALLOWED_AFRAC_LOCTYPES = {
    "attn_in",
    "attn_out",
    "mlp_in",
    "mlp_out",
    "mlp_neuron",
    "attn_q",
    "attn_k",
    "attn_v",
}
MODEL_NAME_ALIASES = {
    "csp_bridge1": "csp_bridges1",
    "csp_bridge2": "csp_bridges2",
}

def get_circuit_sparse_cache_dir() -> Path:
    """Return the cache directory used by ``read_file_cached`` for sparse blobs."""
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        return Path(os.environ["TIKTOKEN_CACHE_DIR"]).expanduser()
    if "DATA_GYM_CACHE_DIR" in os.environ:
        return Path(os.environ["DATA_GYM_CACHE_DIR"]).expanduser()
    return DEFAULT_CIRCUIT_SPARSE_CACHE_DIR


def clear_circuit_sparse_cache() -> Path:
    """Delete the local cache used for circuit-sparsity model blob downloads."""
    cache_dir = get_circuit_sparse_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    return cache_dir


def _load_tinypython_encoding(name: str) -> Encoding:
    if name != "tinypython_2k":
        raise NotImplementedError(f"Unsupported circuit-sparsity tokenizer: {name}")
    return Encoding(**vendored_tinypython.tinypython_2k())


class TokenizerOutput(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class TinyPythonTokenizerWrapper:
    def __init__(self, name: str = "tinypython_2k", padding_side: str = "right"):
        self.name = name
        self.padding_side = padding_side
        self.encoding = _load_tinypython_encoding(name)
        self.eos_token_id = SPECIAL_EOT
        self.bos_token_id = SPECIAL_EOT
        self.pad_token_id = SPECIAL_EOT
        self.unk_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = self.encoding.encode(text)
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def decode(self, token_ids: Union[int, Sequence[int], torch.Tensor]) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.encoding.decode(list(token_ids))

    def _convert_id_to_token(self, idx: int) -> str:
        return self.decode([idx])

    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        padding: Union[bool, str] = False,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False,
    ) -> TokenizerOutput:
        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)
        input_ids = [self.encode(text, add_special_tokens=add_special_tokens) for text in batch]
        attention_mask = [[1] * len(ids) for ids in input_ids]

        if padding in (True, "longest"):
            max_len = max((len(ids) for ids in input_ids), default=0)
            padded_ids = []
            padded_mask = []
            for ids, mask in zip(input_ids, attention_mask):
                pad_len = max_len - len(ids)
                pad_ids = [self.pad_token_id] * pad_len
                pad_mask = [0] * pad_len
                if self.padding_side == "left":
                    padded_ids.append(pad_ids + ids)
                    padded_mask.append(pad_mask + mask)
                else:
                    padded_ids.append(ids + pad_ids)
                    padded_mask.append(mask + pad_mask)
            input_ids = padded_ids
            attention_mask = padded_mask
        elif padding not in (False, None):
            raise NotImplementedError(f"Unsupported padding mode: {padding}")

        if return_tensors is None:
            result = TokenizerOutput(
                input_ids=input_ids[0] if single_input else input_ids,
                attention_mask=attention_mask[0] if single_input else attention_mask,
            )
        elif return_tensors == "pt":
            if not input_ids:
                input_ids_tensor = torch.empty((0, 0), dtype=torch.long)
                attention_mask_tensor = torch.empty((0, 0), dtype=torch.long)
            elif any(len(ids) != len(input_ids[0]) for ids in input_ids):
                raise ValueError("Non-uniform sequence lengths require padding when return_tensors='pt'.")
            else:
                input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
                attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
            result = TokenizerOutput(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
            )
        else:
            raise NotImplementedError(f"Unsupported return_tensors value: {return_tensors}")

        return result


class SparseNorm(nn.Module):
    def __init__(self, dim: int, rms: bool, bias: bool, eps: Optional[float] = None):
        super().__init__()
        self.dim = dim
        self.rms = rms
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if (bias and not rms) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rms:
            eps = self.eps
            if eps is None:
                eps = torch.finfo(x.dtype).eps
            scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
            return x * scale * self.weight
        eps = 1e-5 if self.eps is None else self.eps
        return F.layer_norm(x, (self.dim,), self.weight, self.bias, eps)


class SparseUnembed(nn.Module):
    def __init__(self, d_model: int, d_vocab: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.empty(d_model, d_vocab))
        self.b_U = nn.Parameter(torch.zeros(d_vocab))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bpd,dv->bpv", residual, self.W_U) + self.b_U


class SparseMLP(nn.Module):
    def __init__(self, cfg: HookedTransformerConfig, sparse_cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.sparse_cfg = sparse_cfg
        self.d_model = sparse_cfg["d_model"]
        self.d_pos_emb = sparse_cfg.get("d_pos_emb")
        self.d_mlp = sparse_cfg["d_mlp"]
        self.bias = sparse_cfg["bias"]
        self.c_fc_resid = nn.Linear(self.d_model, self.d_mlp, bias=False)
        self.c_fc_pos = (
            nn.Linear(self.d_pos_emb, self.d_mlp, bias=False) if self.d_pos_emb is not None else None
        )
        self.b_in = nn.Parameter(torch.zeros(self.d_mlp)) if self.bias else None
        self.c_proj = nn.Linear(self.d_mlp, self.d_model, bias=self.bias)
        self.hook_post_act = HookPoint()

    def _maybe_activation_sparsity(
        self,
        x: torch.Tensor,
        loctype: str,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        afrac = self.sparse_cfg.get("afrac")
        if afrac is None:
            if return_mask:
                return x, torch.ones_like(x)
            return x
        loctypes = set((self.sparse_cfg.get("afrac_loctypes") or "").split(","))
        if loctype not in loctypes:
            if return_mask:
                return x, torch.ones_like(x)
            return x
        k = int(afrac * x.shape[-1])
        if k <= 0:
            out = torch.zeros_like(x)
            if return_mask:
                return out, out
            return out
        vals, inds = torch.topk(x.abs(), k, dim=-1, sorted=False)
        del vals
        out = torch.zeros_like(x)
        out.scatter_(-1, inds, x.gather(-1, inds))
        if return_mask:
            mask = torch.zeros_like(x)
            mask.scatter_(-1, inds, torch.ones_like(inds, dtype=x.dtype))
            return out, mask
        return out

    def _project_from_concat_input(self, mlp_input: torch.Tensor) -> torch.Tensor:
        resid_input = mlp_input[..., : self.d_model]
        hidden = self.c_fc_resid(resid_input)
        if self.c_fc_pos is not None:
            pos_input = mlp_input[..., self.d_model :]
            hidden = hidden + self.c_fc_pos(pos_input)
        return hidden

    def forward(self, resid: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.c_fc_pos is not None:
            if pos is None:
                raise ValueError("Positional channels are required for this sparse MLP.")
            mlp_input = torch.cat([resid, pos], dim=-1)
        else:
            mlp_input = resid
        mlp_input = self._maybe_activation_sparsity(mlp_input, "mlp_in")
        hidden = self._project_from_concat_input(mlp_input)
        if self.b_in is not None:
            hidden = hidden + self.b_in
        act = F.gelu(hidden) if self.sparse_cfg["activation_type"] == "gelu" else F.relu(hidden)
        act = self._maybe_activation_sparsity(act, "mlp_neuron")
        act = self.hook_post_act(act)
        out = self.c_proj(act)
        # The native model applies dropout here, but all released checkpoints have
        # dropout=0.0 and this adapter is used in eval-mode inference only.
        out = self._maybe_activation_sparsity(out, "mlp_out")
        return out


class SparseAttention(nn.Module):
    def __init__(self, cfg: HookedTransformerConfig, sparse_cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.sparse_cfg = sparse_cfg
        self.n_heads = sparse_cfg["n_head"]
        self.d_head = sparse_cfg["d_head"]
        self.d_model = sparse_cfg["d_model"]
        self.d_pos_emb = sparse_cfg.get("d_pos_emb")
        self.bias = sparse_cfg["bias"]
        self.W_Q = nn.Parameter(torch.empty(self.n_heads, self.d_model, self.d_head))
        self.W_K = nn.Parameter(torch.empty(self.n_heads, self.d_model, self.d_head))
        self.W_V = nn.Parameter(torch.empty(self.n_heads, self.d_model, self.d_head))
        self.W_Q_pos = (
            nn.Parameter(torch.empty(self.n_heads, self.d_pos_emb, self.d_head))
            if self.d_pos_emb is not None
            else None
        )
        self.W_K_pos = (
            nn.Parameter(torch.empty(self.n_heads, self.d_pos_emb, self.d_head))
            if self.d_pos_emb is not None
            else None
        )
        self.W_V_pos = (
            nn.Parameter(torch.empty(self.n_heads, self.d_pos_emb, self.d_head))
            if self.d_pos_emb is not None
            else None
        )
        if self.bias:
            self.b_Q = nn.Parameter(torch.zeros(self.n_heads, self.d_head))
            self.b_K = nn.Parameter(torch.zeros(self.n_heads, self.d_head))
            self.b_V = nn.Parameter(torch.zeros(self.n_heads, self.d_head))
        else:
            self.b_Q = None
            self.b_K = None
            self.b_V = None
        self.W_O = nn.Parameter(torch.empty(self.n_heads, self.d_head, self.d_model))
        self.b_O = nn.Parameter(torch.zeros(self.d_model)) if self.bias else None
        self.sink_logit = nn.Parameter(torch.zeros(self.n_heads)) if sparse_cfg.get("sink") else None

        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_z = HookPoint()
        self.hook_result = HookPoint()

    def _maybe_activation_sparsity(
        self,
        x: torch.Tensor,
        loctype: str,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        afrac = self.sparse_cfg.get("afrac")
        if afrac is None:
            if return_mask:
                return x, torch.ones_like(x)
            return x
        loctypes = set((self.sparse_cfg.get("afrac_loctypes") or "").split(","))
        if loctype not in loctypes:
            if return_mask:
                return x, torch.ones_like(x)
            return x
        k = int(afrac * x.shape[-1])
        if k <= 0:
            out = torch.zeros_like(x)
            if return_mask:
                return out, out
            return out
        vals, inds = torch.topk(x.abs(), k, dim=-1, sorted=False)
        del vals
        out = torch.zeros_like(x)
        out.scatter_(-1, inds, x.gather(-1, inds))
        if return_mask:
            mask = torch.zeros_like(x)
            mask.scatter_(-1, inds, torch.ones_like(inds, dtype=x.dtype))
            return out, mask
        return out

    def _project(
        self,
        attn_input: torch.Tensor,
        W_main: torch.Tensor,
        b_main: Optional[torch.Tensor],
    ) -> torch.Tensor:
        projected = torch.einsum("bphm,hmd->bphd", attn_input, W_main)
        if b_main is not None:
            projected = projected + b_main[None, None, :, :]
        return projected

    def _flatten_head_outputs_for_sparsity(self, x: torch.Tensor, loctype: str) -> torch.Tensor:
        orig_shape = x.shape
        flattened = x.reshape(*orig_shape[:-2], orig_shape[-2] * orig_shape[-1])
        flattened = self._maybe_activation_sparsity(flattened, loctype)
        return flattened.reshape(orig_shape)

    def forward(
        self,
        query_input: torch.Tensor,
        key_input: torch.Tensor,
        value_input: torch.Tensor,
        pos_query: Optional[torch.Tensor],
        pos_key: Optional[torch.Tensor],
        pos_value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.d_pos_emb is not None:
            if pos_query is None or pos_key is None or pos_value is None:
                raise ValueError("Positional channels are required for this sparse attention block.")
            pos_query_heads = pos_query.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            pos_key_heads = pos_key.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            pos_value_heads = pos_value.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            query_cat = torch.cat([query_input, pos_query_heads], dim=-1)
            key_cat = torch.cat([key_input, pos_key_heads], dim=-1)
            value_cat = torch.cat([value_input, pos_value_heads], dim=-1)
            W_Q_full = torch.cat([self.W_Q, self.W_Q_pos], dim=1)
            W_K_full = torch.cat([self.W_K, self.W_K_pos], dim=1)
            W_V_full = torch.cat([self.W_V, self.W_V_pos], dim=1)
        else:
            query_cat = query_input
            key_cat = key_input
            value_cat = value_input
            W_Q_full = self.W_Q
            W_K_full = self.W_K
            W_V_full = self.W_V

        query_cat = self._maybe_activation_sparsity(query_cat, "attn_in")
        key_cat = self._maybe_activation_sparsity(key_cat, "attn_in")
        value_cat = self._maybe_activation_sparsity(value_cat, "attn_in")

        q = self._project(query_cat, W_Q_full, self.b_Q)
        k = self._project(key_cat, W_K_full, self.b_K)
        v = self._project(value_cat, W_V_full, self.b_V)

        q = self._flatten_head_outputs_for_sparsity(q, "attn_q")
        k = self._flatten_head_outputs_for_sparsity(k, "attn_k")
        v = self._flatten_head_outputs_for_sparsity(v, "attn_v")

        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / (self.d_head**0.5)
        T = scores.size(-1)
        causal_mask = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        if self.sink_logit is not None:
            sink_scores = self.sink_logit[None, :, None, None].expand(scores.size(0), -1, scores.size(2), 1)
            scores = torch.cat([sink_scores, scores], dim=-1)

        scores = self.hook_attn_scores(scores)
        pattern = F.softmax(scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)

        if self.sink_logit is not None:
            pattern_no_sink = pattern[..., 1:]
        else:
            pattern_no_sink = pattern
        z = torch.einsum("bhqk,bkhd->bqhd", pattern_no_sink, v)
        z = self.hook_z(z)

        raw_result = torch.einsum("bqhd,hdm->bqhm", z, self.W_O)
        raw_out = raw_result.sum(dim=2)
        if self.b_O is not None:
            raw_out = raw_out + self.b_O

        # The released sparse checkpoints apply top-k sparsity after summing
        # heads and adding b_O. EAP treats each hook_result head as an additive
        # residual contributor, so expose the post-sparsity decomposition here:
        # apply the same d_model mask to every head and fold an equal share of
        # the masked bias into each head. Then hook_result.sum(dim=2) is exactly
        # the sparse attention output used by downstream residual additions.
        out, attn_out_mask = self._maybe_activation_sparsity(raw_out, "attn_out", return_mask=True)
        result = raw_result * attn_out_mask.unsqueeze(2)
        if self.b_O is not None:
            result = result + (self.b_O * attn_out_mask).unsqueeze(2) / self.n_heads
        result = self.hook_result(result)
        out = result.sum(dim=2)
        # The native model applies dropout after c_proj, but all released
        # checkpoints have dropout=0.0 and this adapter runs inference only.
        return out


class SparseTransformerBlock(nn.Module):
    def __init__(self, cfg: HookedTransformerConfig, sparse_cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.sparse_cfg = sparse_cfg
        self.n_heads = sparse_cfg["n_head"]
        self.d_model = sparse_cfg["d_model"]
        self.d_pos_emb = sparse_cfg.get("d_pos_emb")
        rms_norm = sparse_cfg.get("rms_norm", False)
        ln_bias = sparse_cfg.get("ln_bias", True)

        self.ln1 = SparseNorm(self.d_model, rms=rms_norm, bias=ln_bias)
        self.ln2 = SparseNorm(self.d_model, rms=rms_norm, bias=ln_bias)
        self.ln_p1 = (
            SparseNorm(self.d_pos_emb, rms=rms_norm, bias=ln_bias) if self.d_pos_emb is not None else None
        )
        self.ln_p2 = (
            SparseNorm(self.d_pos_emb, rms=rms_norm, bias=ln_bias) if self.d_pos_emb is not None else None
        )
        self.attn = SparseAttention(cfg, sparse_cfg)
        self.mlp = SparseMLP(cfg, sparse_cfg)

        self.hook_resid_pre = HookPoint()
        self.hook_attn_in = HookPoint()
        self.hook_q_input = HookPoint()
        self.hook_k_input = HookPoint()
        self.hook_v_input = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def _repeat_heads(self, resid: torch.Tensor) -> torch.Tensor:
        return resid.unsqueeze(2).expand(-1, -1, self.n_heads, -1).clone()

    def forward(self, resid_pre: torch.Tensor, pos_emb_to_cat: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # This adapter intentionally commits to the same hook-flag configuration
        # set in pareto_dev_utils.load_model. Supporting every TL hook-flag
        # combination would add branching noise without helping the current EAP
        # workflow, so we fail fast outside that configuration.
        if not self.cfg.use_split_qkv_input:
            raise NotImplementedError(
                "HookedSparseTransformer currently requires cfg.use_split_qkv_input=True."
            )
        if not self.cfg.use_attn_in:
            raise NotImplementedError(
                "HookedSparseTransformer currently requires cfg.use_attn_in=True."
            )
        if not self.cfg.use_hook_mlp_in:
            raise NotImplementedError(
                "HookedSparseTransformer currently requires cfg.use_hook_mlp_in=True."
            )
        if not self.cfg.use_attn_result:
            raise NotImplementedError(
                "HookedSparseTransformer currently requires cfg.use_attn_result=True."
            )
        resid_pre = self.hook_resid_pre(resid_pre)
        attn_in = self.hook_attn_in(self._repeat_heads(resid_pre))
        query_input = self.hook_q_input(attn_in.clone())
        key_input = self.hook_k_input(attn_in.clone())
        value_input = self.hook_v_input(attn_in.clone())

        # whether positional embedding is trained/concatenated or not, they are not a function of the input tokens, thus there is no need to add hooks to them.
        pos_q = self.ln_p1(pos_emb_to_cat) if self.ln_p1 is not None and pos_emb_to_cat is not None else None
        pos_k = pos_q
        pos_v = pos_q
        attn_out = self.hook_attn_out(
            self.attn(
                query_input=self.ln1(query_input),
                key_input=self.ln1(key_input),
                value_input=self.ln1(value_input),
                pos_query=pos_q,
                pos_key=pos_k,
                pos_value=pos_v,
                attention_mask=attention_mask,
            )
        )
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)
        mlp_in = self.hook_mlp_in(resid_mid.clone())
        pos_mlp = self.ln_p2(pos_emb_to_cat) if self.ln_p2 is not None and pos_emb_to_cat is not None else None
        mlp_out = self.hook_mlp_out(self.mlp(self.ln2(mlp_in), pos=pos_mlp))
        # This forward pass did not account for cases where maybe_sparse_activation is applied to residual stream and where residual_activation is not identity. 
        # These cases never happen in their released model, despite their code making them possible. Also, these cases break the addition structure 
        # of the residual stream which attribution patching relies on. I am not sure if CEAP would work there...
        return self.hook_resid_post(resid_mid + mlp_out)


@dataclass
class CircuitSparseCheckpoint:
    model_path: str
    config_json: dict[str, Any]
    state_dict: dict[str, torch.Tensor]


class HookedSparseTransformer(HookedRootModule):
    def __init__(
        self,
        tl_cfg: HookedTransformerConfig,
        sparse_cfg: dict[str, Any],
        tokenizer: TinyPythonTokenizerWrapper,
    ):
        super().__init__()
        self.cfg = tl_cfg
        self.sparse_cfg = sparse_cfg
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.embed = nn.Embedding(tl_cfg.d_vocab, tl_cfg.d_model)
        self.hook_embed = HookPoint()
        self.pos_embed = nn.Embedding(tl_cfg.n_ctx, sparse_cfg.get("d_pos_emb") or tl_cfg.d_model)
        self.hook_pos_embed = HookPoint()
        self.blocks = nn.ModuleList([SparseTransformerBlock(tl_cfg, sparse_cfg) for _ in range(tl_cfg.n_layers)])
        self.ln_final = SparseNorm(
            tl_cfg.d_model,
            rms=sparse_cfg.get("rms_norm", False),
            bias=sparse_cfg.get("ln_bias", True),
        )
        self.unembed = SparseUnembed(tl_cfg.d_model, tl_cfg.d_vocab_out)
        self.final_logits_bias = nn.Parameter(torch.zeros(tl_cfg.d_vocab_out), requires_grad=False)
        if sparse_cfg.get("enable_bigram_table"):
            self.bigram_table = nn.Parameter(
                torch.zeros(tl_cfg.d_vocab_out, tl_cfg.d_vocab_out),
                requires_grad=bool(sparse_cfg.get("learnable_bigram_table", False)),
            )
        else:
            self.register_parameter("bigram_table", None)
        self.setup()

    def check_hooks_to_add(
        self,
        hook_point,
        hook_point_name,
        hook,
        dir="fwd",
        is_permanent=False,
        prepend=False,
    ) -> None:
        del hook_point, hook, dir, is_permanent, prepend
        if hook_point_name.endswith("attn.hook_result"):
            assert (
                self.cfg.use_attn_result
            ), f"Cannot add hook {hook_point_name} if use_attn_result is False"
        if hook_point_name.endswith(("hook_q_input", "hook_k_input", "hook_v_input")):
            assert (
                self.cfg.use_split_qkv_input
            ), f"Cannot add hook {hook_point_name} if use_split_qkv_input is False"
        if hook_point_name.endswith("hook_mlp_in"):
            assert (
                self.cfg.use_hook_mlp_in
            ), f"Cannot add hook {hook_point_name} if use_hook_mlp_in is False"
        if hook_point_name.endswith("hook_attn_in"):
            assert (
                self.cfg.use_attn_in
            ), f"Cannot add hook {hook_point_name} if use_attn_in is False"

    def get_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device=None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
    ):
        """Mostly the same as HookedRootModule.get_caching_hooks.

        Differences from the parent implementation:
        - only moves tensors when ``device is not None``; the parent always
          calls ``.to(device)``
        - iterates over hook names directly because the HookPoint objects are
          not needed here
        """
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif type(names_filter) == str:
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif type(names_filter) == list:
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        self.is_caching = True

        def save_hook(tensor, hook):
            stored = tensor.detach() 
            stored = stored.to(device) if device is not None else stored
            if remove_batch_dim:
                cache[hook.name] = stored[0]
            else:
                cache[hook.name] = stored

        def save_hook_back(tensor, hook):
            stored = tensor.detach()
            stored = stored.to(device) if device is not None else stored
            if remove_batch_dim:
                cache[hook.name + "_grad"] = stored[0]
            else:
                cache[hook.name + "_grad"] = stored

        fwd_hooks = []
        bwd_hooks = []
        for name in self.hook_dict:
            if names_filter(name):
                fwd_hooks.append((name, save_hook))
                if incl_bwd:
                    bwd_hooks.append((name, save_hook_back))
        return cache, fwd_hooks, bwd_hooks

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _tokenize_for_model(self, inputs: Union[str, Sequence[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = [inputs] if isinstance(inputs, str) else list(inputs)
        tokenized = self.tokenizer(batch, padding="longest", return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        bos = torch.full((input_ids.size(0), 1), self.tokenizer.bos_token_id, dtype=torch.long)
        input_ids = torch.cat([bos, input_ids], dim=1)
        attention_mask = torch.cat([torch.ones_like(bos), attention_mask], dim=1)
        return input_ids.to(self.device), attention_mask.to(self.device)

    def forward(
        self,
        inputs: Union[str, Sequence[str], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(inputs, (str, list, tuple)):
            tokens, attention_mask = self._tokenize_for_model(inputs)
        else:
            tokens = inputs.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(tokens, dtype=torch.long, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

        B, T = tokens.shape
        if T > self.cfg.n_ctx:
            raise ValueError(f"Input length {T} exceeds configured context {self.cfg.n_ctx}.")

        embed = self.hook_embed(self.embed(tokens))
        pos = self.hook_pos_embed(self.pos_embed.weight[:T].unsqueeze(0).expand(B, -1, -1))
        if self.sparse_cfg.get("d_pos_emb") is not None:
            resid = embed
            pos_emb_to_cat = pos
        else:
            resid = embed + pos
            pos_emb_to_cat = None

        for block in self.blocks:
            resid = block(resid, pos_emb_to_cat=pos_emb_to_cat, attention_mask=attention_mask)

        resid = self.ln_final(resid)
        logits = self.unembed(resid) + self.final_logits_bias
        if self.bigram_table is not None:
            logits = logits + F.embedding(tokens, self.bigram_table)
        return logits

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: CircuitSparseCheckpoint,
        device: str,
    ) -> "HookedSparseTransformer":
        sparse_cfg = normalize_sparse_config(checkpoint.config_json)
        validate_sparse_config(sparse_cfg)
        tl_cfg = sparse_config_to_hooked_transformer_config(
            sparse_cfg=sparse_cfg,
            model_name=checkpoint.model_path,
            device=device,
        )
        tokenizer = TinyPythonTokenizerWrapper(sparse_cfg.get("tokenizer_name", "tinypython_2k"))
        model = cls(tl_cfg=tl_cfg, sparse_cfg=sparse_cfg, tokenizer=tokenizer)
        load_sparse_weights(model, checkpoint.state_dict, sparse_cfg)
        model.to(device)
        model.eval()
        return model


def normalize_sparse_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(raw_cfg)
    if "n_mlp" in cfg and "d_mlp" not in cfg:
        cfg["d_mlp"] = cfg.pop("n_mlp")
    cfg.setdefault("sink", False)
    cfg.setdefault("d_pos_emb", None)
    cfg.setdefault("afrac", None)
    cfg.setdefault("afrac_loctypes", "")
    cfg.setdefault("enable_bigram_table", False)
    cfg.setdefault("learnable_bigram_table", False)
    cfg.setdefault("rms_norm", False)
    cfg.setdefault("ln_bias", True)
    cfg.setdefault("tokenizer_name", "tinypython_2k")
    return cfg


def validate_sparse_config(sparse_cfg: dict[str, Any]) -> None:
    residual_activation_type = sparse_cfg.get("residual_activation_type", "identity")
    if residual_activation_type != "identity":
        raise NotImplementedError(
            "Sparse adapter only supports residual_activation_type='identity'."
        )
    loctypes = {loc for loc in (sparse_cfg.get("afrac_loctypes") or "").split(",") if loc}
    unsupported = loctypes - ALLOWED_AFRAC_LOCTYPES
    if unsupported:
        unsupported_str = ", ".join(sorted(unsupported))
        raise NotImplementedError(
            "Sparse adapter only supports activation sparsity inside attention/MLP internals. "
            f"Unsupported afrac locations: {unsupported_str}"
        )


def sparse_config_to_hooked_transformer_config(
    sparse_cfg: dict[str, Any],
    model_name: str,
    device: str,
) -> HookedTransformerConfig:
    cfg = HookedTransformerConfig(
        n_layers=sparse_cfg["n_layer"],
        d_model=sparse_cfg["d_model"],
        n_ctx=sparse_cfg["block_size"],
        d_head=sparse_cfg["d_head"],
        n_heads=sparse_cfg["n_head"],
        d_mlp=sparse_cfg["d_mlp"],
        d_vocab=sparse_cfg["vocab_size"],
        d_vocab_out=sparse_cfg["vocab_size"],
        act_fn=sparse_cfg["activation_type"],
        eps=1e-5,
        use_attn_result=True,
        use_split_qkv_input=True,
        use_hook_mlp_in=True,
        use_attn_in=True,
        model_name=model_name,
        original_architecture="CircuitSparsityGPT",
        tokenizer_name=sparse_cfg.get("tokenizer_name"),
        normalization_type="RMS" if sparse_cfg.get("rms_norm") else "LN",
        attention_dir="causal",
        positional_embedding_type="standard",
        default_prepend_bos=True,
        device=device,
        dtype=torch.float32,
        init_weights=False,
    )
    cfg.parallel_attn_mlp = False
    cfg.use_attn_scale = True
    cfg.circuit_sparsity_config = sparse_cfg
    cfg.enable_bigram_table = sparse_cfg.get("enable_bigram_table", False)
    cfg.sink = sparse_cfg.get("sink", False)
    cfg.cat_pos_emb = sparse_cfg.get("d_pos_emb") is not None
    return cfg


def is_circuit_sparsity_model_identifier(model_name: str) -> bool:
    model_name = str(model_name)
    if os.path.isdir(model_name) and os.path.exists(os.path.join(model_name, "beeg_config.json")):
        return True
    if model_name.startswith(MODEL_BASE_URL):
        return True
    basename = model_name.rstrip("/").split("/")[-1]
    basename = MODEL_NAME_ALIASES.get(basename, basename)
    return basename.startswith("dense1_") or basename.startswith("csp_")


def resolve_circuit_sparsity_model_path(model_name: str) -> str:
    model_name = str(model_name)
    if os.path.isdir(model_name) and os.path.exists(os.path.join(model_name, "beeg_config.json")):
        return model_name
    if model_name.startswith(MODEL_BASE_URL):
        return model_name.rstrip("/")
    basename = model_name.rstrip("/").split("/")[-1]
    basename = MODEL_NAME_ALIASES.get(basename, basename)
    return f"{MODEL_BASE_URL}/{basename}"


def load_circuit_sparse_checkpoint(model_name: str, map_location: str = "cpu") -> CircuitSparseCheckpoint:
    model_path = resolve_circuit_sparsity_model_path(model_name)
    config_bytes = read_file_cached(f"{model_path}/beeg_config.json")
    config_json = json.loads(config_bytes.decode())
    ckpt_bytes = read_file_cached(f"{model_path}/final_model.pt")
    try:
        state_dict = torch.load(io.BytesIO(ckpt_bytes), map_location=map_location, weights_only=True)
    except TypeError:
        state_dict = torch.load(io.BytesIO(ckpt_bytes), map_location=map_location)
    if "final_logits_bias" not in state_dict:
        state_dict["final_logits_bias"] = torch.zeros(config_json["vocab_size"])
    return CircuitSparseCheckpoint(model_path=model_path, config_json=config_json, state_dict=state_dict)


def load_circuit_sparse_model(model_name: str, device: str = "cpu") -> HookedSparseTransformer:
    checkpoint = load_circuit_sparse_checkpoint(model_name, map_location="cpu")
    return HookedSparseTransformer.from_checkpoint(checkpoint, device=device)


def _split_attn_weight(
    packed_weight: torch.Tensor,
    n_heads: int,
    d_head: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_in = packed_weight.size(1)
    reshaped = packed_weight.view(3, n_heads, d_head, d_in)
    q, k, v = reshaped.unbind(dim=0)
    return q.permute(0, 2, 1), k.permute(0, 2, 1), v.permute(0, 2, 1)


def _split_attn_bias(
    packed_bias: Optional[torch.Tensor],
    n_heads: int,
    d_head: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if packed_bias is None:
        return None, None, None
    reshaped = packed_bias.view(3, n_heads, d_head)
    return tuple(reshaped.unbind(dim=0))


def _copy_norm_state(target: SparseNorm, state_dict: dict[str, torch.Tensor], prefix: str) -> None:
    target.weight.copy_(state_dict[f"{prefix}.weight"])
    if target.bias is not None and f"{prefix}.bias" in state_dict:
        target.bias.copy_(state_dict[f"{prefix}.bias"])


def load_sparse_weights(
    model: HookedSparseTransformer,
    state_dict: dict[str, torch.Tensor],
    sparse_cfg: dict[str, Any],
) -> None:
    with torch.no_grad():
        model.embed.weight.copy_(state_dict["transformer.wte.weight"])
        model.pos_embed.weight.copy_(state_dict["transformer.wpe.weight"])
        _copy_norm_state(model.ln_final, state_dict, "transformer.ln_f")
        if model.bigram_table is not None:
            model.bigram_table.copy_(state_dict["bigram_table"])
        model.final_logits_bias.copy_(state_dict.get("final_logits_bias", torch.zeros_like(model.final_logits_bias)))
        model.unembed.W_U.copy_(state_dict["lm_head.weight"].T)

        for layer, block in enumerate(model.blocks):
            prefix = f"transformer.h.{layer}"
            _copy_norm_state(block.ln1, state_dict, f"{prefix}.ln_1")
            _copy_norm_state(block.ln2, state_dict, f"{prefix}.ln_2")
            packed_attn_w = state_dict[f"{prefix}.attn.c_attn.weight"]
            q_w, k_w, v_w = _split_attn_weight(packed_attn_w, sparse_cfg["n_head"], sparse_cfg["d_head"])
            packed_attn_b = state_dict.get(f"{prefix}.attn.c_attn.bias")
            q_b, k_b, v_b = _split_attn_bias(packed_attn_b, sparse_cfg["n_head"], sparse_cfg["d_head"])
            if q_b is not None:
                block.attn.b_Q.copy_(q_b)
                block.attn.b_K.copy_(k_b)
                block.attn.b_V.copy_(v_b)

            c_proj_w = state_dict[f"{prefix}.attn.c_proj.weight"]
            block.attn.W_O.copy_(c_proj_w.view(sparse_cfg["d_model"], sparse_cfg["n_head"], sparse_cfg["d_head"]).permute(1, 2, 0))
            if block.attn.b_O is not None:
                block.attn.b_O.copy_(state_dict[f"{prefix}.attn.c_proj.bias"])

            d_pos_emb = sparse_cfg.get("d_pos_emb")
            if d_pos_emb is not None:
                _copy_norm_state(block.ln_p1, state_dict, f"{prefix}.ln_p1")
                _copy_norm_state(block.ln_p2, state_dict, f"{prefix}.ln_p2")
                d_model = sparse_cfg["d_model"]
                block.attn.W_Q.copy_(q_w[:, :d_model, :])
                block.attn.W_K.copy_(k_w[:, :d_model, :])
                block.attn.W_V.copy_(v_w[:, :d_model, :])
                block.attn.W_Q_pos.copy_(q_w[:, d_model:, :])
                block.attn.W_K_pos.copy_(k_w[:, d_model:, :])
                block.attn.W_V_pos.copy_(v_w[:, d_model:, :])
            else:
                block.attn.W_Q.copy_(q_w)
                block.attn.W_K.copy_(k_w)
                block.attn.W_V.copy_(v_w)

            mlp_fc_w = state_dict[f"{prefix}.mlp.c_fc.weight"]
            if sparse_cfg.get("d_pos_emb") is not None:
                d_model = sparse_cfg["d_model"]
                block.mlp.c_fc_resid.weight.copy_(mlp_fc_w[:, :d_model])
                block.mlp.c_fc_pos.weight.copy_(mlp_fc_w[:, d_model:])
            else:
                block.mlp.c_fc_resid.weight.copy_(mlp_fc_w)
            if block.mlp.b_in is not None:
                block.mlp.b_in.copy_(state_dict[f"{prefix}.mlp.c_fc.bias"])
            block.mlp.c_proj.weight.copy_(state_dict[f"{prefix}.mlp.c_proj.weight"])
            if block.mlp.c_proj.bias is not None:
                block.mlp.c_proj.bias.copy_(state_dict[f"{prefix}.mlp.c_proj.bias"])

        if sparse_cfg.get("sink"):
            for layer, block in enumerate(model.blocks):
                prefix = f"transformer.h.{layer}"
                sink_param = state_dict.get(f"{prefix}.attn.attn_imp.sink_logit")
                if sink_param is not None and block.attn.sink_logit is not None:
                    block.attn.sink_logit.copy_(sink_param)
