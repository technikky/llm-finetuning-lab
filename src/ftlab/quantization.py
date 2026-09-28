"""QLoRA: 4-bit base weights with full-precision adapters.

QLoRA's move is narrow and effective. The frozen base is stored in 4-bit NormalFloat, which
cuts the weight term by roughly 4x against bf16, while the adapters and their optimizer state
stay in higher precision because they are the only things being trained and they are tiny.
The compute dtype stays bf16: weights are dequantised per block on the way into each matmul.

**This module is honest about what it cannot do here.** 4-bit quantisation needs
``bitsandbytes`` and a CUDA device. On a CPU-only or non-CUDA machine the config can still be
constructed and inspected -- which is what the tests do -- but no model can be loaded in 4
bits. :func:`quantization_available` reports which case you are in, and the CLI prints it
rather than failing obscurely halfway through a load.

No 4-bit training results are published in this repository, because none were run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ftlab.types import BYTES_PER_PARAM, Quantization


@dataclass(frozen=True)
class QuantizationSupport:
    """What this machine can actually do."""

    bitsandbytes_installed: bool
    cuda_available: bool
    device_name: str = ""
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.bitsandbytes_installed and self.cuda_available


def quantization_available() -> QuantizationSupport:
    """Probe for 4-bit support without raising."""
    try:
        import bitsandbytes  # noqa: F401

        installed = True
    except ImportError:
        installed = False

    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else ""

    reasons: list[str] = []
    if not installed:
        reasons.append("bitsandbytes is not installed")
    if not cuda:
        reasons.append("no CUDA device is available")

    return QuantizationSupport(
        bitsandbytes_installed=installed,
        cuda_available=cuda,
        device_name=device,
        reason="; ".join(reasons),
    )


def bnb_config_kwargs(
    *,
    quantization: Quantization = "nf4",
    compute_dtype: str = "bfloat16",
    double_quant: bool = True,
) -> dict[str, Any]:
    """The ``BitsAndBytesConfig`` keyword arguments for a quantisation setting.

    Returned as a dict rather than a config object so this is inspectable and testable with
    no ``bitsandbytes`` installed. :func:`build_bnb_config` turns it into the real object when
    the library is present.

    ``double_quant`` quantises the per-block quantisation constants themselves, which the
    QLoRA paper measures at about 0.4 bits per parameter saved. ``nf4`` is the
    information-theoretically motivated 4-bit format from that paper; plain ``int8`` is the
    older, larger option.
    """
    if quantization == "none":
        return {}
    if quantization == "int8":
        return {"load_in_8bit": True}
    if quantization == "nf4":
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": double_quant,
            "bnb_4bit_compute_dtype": compute_dtype,
        }
    raise ValueError(f"unknown quantization {quantization!r}; expected none, int8 or nf4")


def build_bnb_config(**kwargs: Any) -> Any:  # noqa: ANN401 - BitsAndBytesConfig is untyped
    """Construct a real ``BitsAndBytesConfig``. Raises with the reason when unsupported."""
    support = quantization_available()
    if not support.bitsandbytes_installed:
        raise RuntimeError(
            "4-bit quantisation needs bitsandbytes. Install with: "
            "pip install 'llm-finetuning-lab[quant]'"
        )

    from transformers import BitsAndBytesConfig

    config_kwargs = dict(kwargs)
    dtype = config_kwargs.get("bnb_4bit_compute_dtype")
    if isinstance(dtype, str):
        config_kwargs["bnb_4bit_compute_dtype"] = getattr(torch, dtype)
    # transformers ships no annotation for this constructor.
    return BitsAndBytesConfig(**config_kwargs)  # type: ignore[no-untyped-call]


def weight_bytes_per_parameter(quantization: Quantization, precision: str = "bf16") -> float:
    """Bytes per base parameter under a quantisation setting."""
    return BYTES_PER_PARAM[quantization if quantization != "none" else precision]


def compression_ratio(quantization: Quantization, precision: str = "bf16") -> float:
    """How much smaller the base weights become, relative to unquantised.

    For nf4 against bf16 this is 2 / 0.5625 ~= 3.56x, not the 4x that "4-bit" suggests: the
    per-block scaling metadata is real and counted here.
    """
    baseline = BYTES_PER_PARAM[precision]
    return round(baseline / weight_bytes_per_parameter(quantization, precision), 4)
