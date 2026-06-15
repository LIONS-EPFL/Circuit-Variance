#!/usr/bin/env python3
"""Delete cached circuit-sparsity model blobs used by faithfulness."""

from circuit_sparse_adapter import clear_circuit_sparse_cache, get_circuit_sparse_cache_dir


def main() -> None:
    cache_dir = get_circuit_sparse_cache_dir()
    existed = cache_dir.exists()
    cleared_dir = clear_circuit_sparse_cache()
    if existed:
        print(f"Deleted {cleared_dir}")
    else:
        print(f"Skipped {cleared_dir} (not found)")


if __name__ == "__main__":
    main()
