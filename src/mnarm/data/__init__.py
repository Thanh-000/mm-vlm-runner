"""Authorized dataset loaders, strict split-overlap guards, and tensor stores."""

from mnarm.data.archive_ops import (
    aria2_download_file,
    extract_archive_safe,
    hf_lfs_batch_download,
    is_valid_binary_archive,
)
from mnarm.data.crisismmd import (
    CRISISMMD_CLASSES,
    ORIGINAL_PROMPTS,
    UNSEEN_PROMPTS,
    assert_zero_split_overlap,
    extract_contracts_from_a2z_zip,
    load_crisismmd_contracts,
    uid_hash,
)
from mnarm.data.disaster_suite import (
    load_benchmark_contracts,
    prepare_disaster_m3,
    prepare_dmd,
    prepare_incidents_vqa,
)
from mnarm.data.protocol_builder import build_p1_p4_contracts
from mnarm.data.tensor_store import MaskedRowsDataset, PreprocessedTensorStore

__all__ = [
    "CRISISMMD_CLASSES",
    "MaskedRowsDataset",
    "ORIGINAL_PROMPTS",
    "PreprocessedTensorStore",
    "UNSEEN_PROMPTS",
    "aria2_download_file",
    "assert_zero_split_overlap",
    "build_p1_p4_contracts",
    "extract_archive_safe",
    "extract_contracts_from_a2z_zip",
    "hf_lfs_batch_download",
    "is_valid_binary_archive",
    "load_benchmark_contracts",
    "load_crisismmd_contracts",
    "prepare_disaster_m3",
    "prepare_dmd",
    "prepare_incidents_vqa",
    "uid_hash",
]
