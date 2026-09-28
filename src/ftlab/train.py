"""The training loop.

Real PyTorch, real gradients, real optimizer steps. It runs on CPU in seconds against the
tiny architectures in :mod:`ftlab.specs`, which is what lets CI exercise the whole path --
masking, batching, clipping, scheduling, checkpointing, evaluation and merging -- on every
commit rather than only on a machine with a GPU.

Three things are recorded that a bare loop usually drops, because without them a loss curve
cannot be compared to anything: the config hash, the environment, and the held-out metrics
from **before** training as well as after. A final loss on its own says nothing; the pair
says whether anything was learned.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from torch.nn.utils import clip_grad_norm_

from ftlab.budget import parameter_budget
from ftlab.data import PreparedDataset, batches, collate
from ftlab.evaluate import evaluate
from ftlab.lora import apply_lora, base_parameters_are_frozen
from ftlab.specs import build_model, get_spec
from ftlab.tokenizer import PAD_ID
from ftlab.types import StepMetric, TrainingConfig, TrainingRun


def set_seed(seed: int) -> None:
    """Seed every source of randomness the loop touches."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in CI
        torch.cuda.manual_seed_all(seed)


def config_hash(config: TrainingConfig) -> str:
    """Digest over the canonical config.

    Two runs with the same hash were configured identically, so a difference in their curves
    came from the environment rather than the settings.
    """
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def environment_info() -> dict[str, object]:
    """Hardware and library versions. A loss curve is only reproducible on a comparable stack."""
    import peft
    import transformers

    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
    }
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in CI
        info["device"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def _learning_rate_at(step: int, total_steps: int, config: TrainingConfig) -> float:
    """Linear warmup then linear decay.

    Warmup exists because the adapter starts at zero: the first steps produce large relative
    updates, and a full learning rate there can put the adapter somewhere it takes the rest of
    the run to leave.
    """
    if total_steps <= 0:
        return config.learning_rate
    warmup_steps = int(total_steps * config.warmup_ratio)
    if warmup_steps > 0 and step < warmup_steps:
        return config.learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return config.learning_rate * max(0.0, 1.0 - progress)


def train(
    dataset: PreparedDataset,
    config: TrainingConfig,
    *,
    run_id: str | None = None,
    notes: str = "",
) -> tuple[PeftModel, TrainingRun]:
    """Fine-tune a model with LoRA and return it along with the full run record."""
    if not dataset.train:
        raise ValueError("cannot train on an empty training split")
    if dataset.stats.n_leaked:
        raise ValueError(
            f"{dataset.stats.n_leaked} evaluation example(s) also appear in training "
            f"({', '.join(dataset.stats.leaked_ids[:5])}). Fix the split before training: any "
            f"result would be measuring memorisation."
        )

    spec = get_spec(config.model)
    set_seed(config.seed)

    base = build_model(spec)
    model = apply_lora(base, spec, config.lora)
    if not base_parameters_are_frozen(model):  # pragma: no cover - guards a peft regression
        raise RuntimeError(
            "a non-adapter parameter is trainable; this run would not be parameter-efficient"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    eval_before = evaluate(model, dataset.eval, batch_size=config.batch_size, pad_id=PAD_ID)

    all_batches = [
        batch
        for epoch in range(config.epochs)
        # A different shuffle each epoch, derived from the run seed so it stays reproducible.
        for batch in batches(dataset.train, config.batch_size, seed=config.seed + epoch)
    ]
    total_steps = len(all_batches)
    steps_per_epoch = max(1, total_steps // config.epochs)

    steps: list[StepMetric] = []
    started = time.perf_counter()
    model.train()

    for step, batch in enumerate(all_batches):
        learning_rate = _learning_rate_at(step, total_steps, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        input_ids, labels, attention_mask = collate(batch, pad_id=PAD_ID)
        outputs = model(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
        )
        outputs.loss.backward()
        grad_norm = float(clip_grad_norm_(trainable, config.max_grad_norm).item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        steps.append(
            StepMetric(
                step=step,
                epoch=round(step / steps_per_epoch, 4),
                loss=round(float(outputs.loss.item()), 6),
                learning_rate=learning_rate,
                grad_norm=round(grad_norm, 6),
            )
        )

    wall_seconds = time.perf_counter() - started
    eval_after = evaluate(model, dataset.eval, batch_size=config.batch_size, pad_id=PAD_ID)

    digest = config_hash(config)
    run = TrainingRun(
        run_id=run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{digest[:12]}",
        created_at=datetime.now(timezone.utc).isoformat(),
        ftlab_version=_version(),
        config=config,
        config_hash=digest,
        environment=environment_info(),
        dataset=dataset.stats,
        budget=parameter_budget(spec, config.lora),
        steps=steps,
        eval_before=eval_before,
        eval_after=eval_after,
        wall_seconds=round(wall_seconds, 3),
        notes=notes,
    )
    return model, run


def save_adapter(model: PeftModel, directory: str | Path) -> Path:
    """Save the adapter only.

    Megabytes rather than gigabytes: the base weights are unchanged, so storing them with
    every checkpoint would be storing the same frozen tensors repeatedly.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    return target


def load_adapter(spec_name: str, directory: str | Path) -> PeftModel:
    """Rebuild the base from its spec and load an adapter onto it.

    The base is reconstructed from the architecture rather than read from the checkpoint,
    which is the point of saving only the adapter.
    """
    source = Path(directory)
    if not source.is_dir():
        raise FileNotFoundError(f"no adapter at {source}")
    base = build_model(get_spec(spec_name))
    return PeftModel.from_pretrained(base, str(source))


def _version() -> str:
    from ftlab import __version__

    return __version__
