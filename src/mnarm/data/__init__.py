"""Authorized dataset loaders, strict split-overlap guards, and tensor stores."""

from mnarm.data.crisismmd import (
    CRISISMMD_CLASSES,
    ORIGINAL_PROMPTS,
    UNSEEN_PROMPTS,
    assert_zero_split_overlap,
    extract_contracts_from_a2z_zip,
    load_crisismmd_contracts,
    uid_hash,
)
from mnarm.data.tensor_store import MaskedRowsDataset, PreprocessedTensorStore

__all__ = [
    "CRISISMMD_CLASSES",
    "MaskedRowsDataset",
    "ORIGINAL_PROMPTS",
    "PreprocessedTensorStore",
    "UNSEEN_PROMPTS",
    "assert_zero_split_overlap",
    "extract_contracts_from_a2z_zip",
    "load_crisismmd_contracts",
    "uid_hash",
]
