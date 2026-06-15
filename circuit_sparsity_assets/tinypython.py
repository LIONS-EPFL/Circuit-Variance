"""Vendored tinypython tokenizer spec used by circuit-sparsity models.

Provenance:
- Adapted from `circuit_sparsity/tiktoken_ext/tinypython.py`
- Source repository: `https://github.com/openai/circuit_sparsity/` (local checkout used during development)

This vendored copy exists so `faithfulness` does not require a neighboring
`circuit_sparsity` checkout or `sys.path` mutation at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

from tiktoken.load import data_gym_to_mergeable_bpe_ranks

ENDOFTEXT = "<|endoftext|>"


def _paths_from_env_or_default() -> tuple[str, str]:
    """Resolve local tokenizer asset paths.

    If `TINYPYTHON_TOK_DIR` is set, use that directory. Otherwise use the
    vendored assets shipped alongside this module.
    """

    local_dir = os.environ.get("TINYPYTHON_TOK_DIR")
    if local_dir is None:
        local_dir = str(Path(__file__).resolve().parent)
    vocab_bpe = os.path.join(local_dir, "vocab.bpe")
    encoder_json = os.path.join(local_dir, "encoder.json")
    return vocab_bpe, encoder_json


def tinypython_2k():
    """Return the `tiktoken.Encoding` spec for the tinypython_2k tokenizer."""

    vocab_bpe_file, encoder_json_file = _paths_from_env_or_default()
    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file=vocab_bpe_file,
        encoder_json_file=encoder_json_file,
    )

    return {
        "name": "tinypython_2k",
        "pat_str": r"""[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 2047},
    }


ENCODING_CONSTRUCTORS = {
    "tinypython_2k": tinypython_2k,
}
