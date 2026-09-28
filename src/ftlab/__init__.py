"""LoRA and QLoRA fine-tuning: exact budgets, real runs, checkable claims."""

from __future__ import annotations

from ftlab.budget import (
    adapter_parameters,
    base_parameters,
    fits_within,
    full_finetune_memory_gb,
    memory_budget,
    parameter_budget,
)
from ftlab.data import (
    IGNORE_INDEX,
    PreparedDataset,
    TokenizedExample,
    batches,
    collate,
    fingerprint,
    load_examples,
    prepare,
    tokenize_example,
)
from ftlab.evaluate import evaluate, generate
from ftlab.lora import (
    adapter_is_zero_initialised,
    apply_lora,
    base_parameters_are_frozen,
    max_logit_difference,
    merge_adapter,
    to_peft_config,
)
from ftlab.quantization import bnb_config_kwargs, compression_ratio, quantization_available
from ftlab.specs import (
    SPECS,
    build_model,
    count_parameters,
    get_spec,
    init_seed,
    published_specs,
)
from ftlab.templates import get_template
from ftlab.tokenizer import ByteTokenizer
from ftlab.tracking import RunRegistry, comparable
from ftlab.train import config_hash, environment_info, load_adapter, save_adapter, set_seed, train
from ftlab.types import (
    CausalLM,
    DatasetStats,
    EvalMetrics,
    Example,
    LoraSpec,
    MemoryBudget,
    ModelSpec,
    ParameterBudget,
    TrainingConfig,
    TrainingRun,
)

__version__ = "0.1.0"

__all__ = [
    "IGNORE_INDEX",
    "SPECS",
    "ByteTokenizer",
    "CausalLM",
    "DatasetStats",
    "EvalMetrics",
    "Example",
    "LoraSpec",
    "MemoryBudget",
    "ModelSpec",
    "ParameterBudget",
    "PreparedDataset",
    "RunRegistry",
    "TokenizedExample",
    "TrainingConfig",
    "TrainingRun",
    "__version__",
    "adapter_is_zero_initialised",
    "adapter_parameters",
    "apply_lora",
    "base_parameters",
    "base_parameters_are_frozen",
    "batches",
    "bnb_config_kwargs",
    "build_model",
    "collate",
    "comparable",
    "compression_ratio",
    "config_hash",
    "count_parameters",
    "environment_info",
    "evaluate",
    "fingerprint",
    "fits_within",
    "full_finetune_memory_gb",
    "generate",
    "get_spec",
    "get_template",
    "init_seed",
    "load_adapter",
    "load_examples",
    "max_logit_difference",
    "memory_budget",
    "merge_adapter",
    "parameter_budget",
    "prepare",
    "published_specs",
    "quantization_available",
    "save_adapter",
    "set_seed",
    "to_peft_config",
    "tokenize_example",
    "train",
]
