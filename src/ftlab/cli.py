"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ftlab.budget import (
    OPTIMIZER_STATE,
    base_parameters,
    fits_within,
    full_finetune_memory_gb,
    memory_budget,
    parameter_budget,
)
from ftlab.data import load_examples, prepare
from ftlab.evaluate import evaluate, generate
from ftlab.lora import merge_adapter
from ftlab.quantization import compression_ratio, quantization_available
from ftlab.report import (
    abbreviate,
    budget_payload,
    budget_report,
    comparison_table,
    dataset_table,
    memory_table,
    parameter_table,
    run_report,
    write_json,
    write_text,
)
from ftlab.specs import SPECS, SpecError, get_spec
from ftlab.tokenizer import ByteTokenizer
from ftlab.tracking import RegistryError, RunRegistry, comparable
from ftlab.train import load_adapter, save_adapter, train
from ftlab.types import CausalLM, LoraSpec, TrainingConfig

TARGETS = ("attention", "mlp", "all-linear")
PRECISIONS = ("fp32", "fp16", "bf16")
QUANTIZATIONS = ("none", "int8", "nf4")
TEMPLATES = ("alpaca", "chatml", "llama2-chat", "raw")


def _add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default: 2 * r)")
    parser.add_argument("--target", choices=TARGETS, default="attention")
    parser.add_argument("--dropout", type=float, default=0.0)


def _lora_from(args: argparse.Namespace) -> LoraSpec:
    return LoraSpec(
        r=args.r,
        alpha=args.alpha if args.alpha is not None else 2 * args.r,
        target=args.target,
        dropout=args.dropout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ftlab",
        description="LoRA and QLoRA fine-tuning: exact budgets, real runs, checkable claims.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    specs = sub.add_parser("specs", help="list the known model architectures")

    budget = sub.add_parser("budget", help="compute parameter and memory budgets")
    budget.add_argument("--model", default=None, help="one model, or all published specs")
    _add_lora_args(budget)
    budget.add_argument("--precision", choices=PRECISIONS, default="bf16")
    budget.add_argument("--quantization", choices=QUANTIZATIONS, default="none")
    budget.add_argument("--optimizer", choices=tuple(OPTIMIZER_STATE), default="adamw")
    budget.add_argument("--vram", type=float, default=None, help="check against this many GiB")
    budget.add_argument("--out", default=None, help="directory for budget.json and .md")

    prepare_cmd = sub.add_parser("prepare", help="prepare a dataset and report its statistics")
    prepare_cmd.add_argument("data", help="JSONL of instruction examples")
    prepare_cmd.add_argument("--template", choices=TEMPLATES, default="alpaca")
    prepare_cmd.add_argument("--max-length", type=int, default=512)
    prepare_cmd.add_argument("--eval-fraction", type=float, default=0.2)
    prepare_cmd.add_argument("--seed", type=int, default=0)
    prepare_cmd.add_argument("--out", default=None)

    train_cmd = sub.add_parser("train", help="fine-tune a model with LoRA")
    train_cmd.add_argument("data", help="JSONL of instruction examples")
    train_cmd.add_argument("--model", default="tiny-llama")
    _add_lora_args(train_cmd)
    train_cmd.add_argument("--template", choices=TEMPLATES, default="alpaca")
    train_cmd.add_argument("--max-length", type=int, default=512)
    train_cmd.add_argument("--epochs", type=int, default=3)
    train_cmd.add_argument("--batch-size", type=int, default=4)
    train_cmd.add_argument("--lr", type=float, default=1e-3)
    train_cmd.add_argument("--warmup-ratio", type=float, default=0.1)
    train_cmd.add_argument("--seed", type=int, default=0)
    train_cmd.add_argument("--runs-dir", default="runs", help="run registry directory")
    train_cmd.add_argument("--adapter-out", default=None, help="directory to save the adapter")
    train_cmd.add_argument("--out", default=None, help="directory for the run report")
    train_cmd.add_argument("--notes", default="")

    evaluate_cmd = sub.add_parser("evaluate", help="evaluate a saved adapter on held-out data")
    evaluate_cmd.add_argument("data")
    evaluate_cmd.add_argument("--model", default="tiny-llama")
    evaluate_cmd.add_argument("--adapter", required=True, help="saved adapter directory")
    evaluate_cmd.add_argument("--template", choices=TEMPLATES, default="alpaca")
    evaluate_cmd.add_argument("--max-length", type=int, default=512)
    evaluate_cmd.add_argument("--batch-size", type=int, default=4)
    evaluate_cmd.add_argument("--merge", action="store_true", help="merge before evaluating")

    generate_cmd = sub.add_parser("generate", help="generate from a base model or an adapter")
    generate_cmd.add_argument("prompt")
    generate_cmd.add_argument("--model", default="tiny-llama")
    generate_cmd.add_argument("--adapter", default=None)
    generate_cmd.add_argument("--max-new-tokens", type=int, default=48)
    generate_cmd.add_argument("--merge", action="store_true")

    runs_cmd = sub.add_parser("runs", help="list recorded runs")
    runs_cmd.add_argument("--runs-dir", default="runs")

    compare_cmd = sub.add_parser("compare", help="compare recorded runs")
    compare_cmd.add_argument("run_ids", nargs="*", help="run ids, or empty for all")
    compare_cmd.add_argument("--runs-dir", default="runs")

    quant_cmd = sub.add_parser("quantization", help="report 4-bit support on this machine")
    quant_cmd.add_argument("--precision", choices=PRECISIONS, default="bf16")

    del specs  # configured above; no extra arguments
    return parser


def _cmd_specs(args: argparse.Namespace) -> int:
    del args
    print(f"{'name':14} {'arch':8} {'params':>10} {'layers':>7} {'hidden':>7} {'kv':>4} published")
    for spec in SPECS.values():
        total = base_parameters(spec)
        size = abbreviate(total)
        print(
            f"{spec.name:14} {spec.architecture:8} {size:>10} {spec.num_hidden_layers:>7} "
            f"{spec.hidden_size:>7} {spec.num_key_value_heads:>4} {spec.published}"
        )
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    lora = _lora_from(args)

    if args.model is None:
        markdown = budget_report(
            lora,
            precision=args.precision,
            quantization=args.quantization,
            optimizer=args.optimizer,
        )
        print(markdown)
        if args.out:
            print(f"wrote {write_text(markdown, Path(args.out) / 'budget.md')}")
        return 0

    spec = get_spec(args.model)
    parameters = parameter_budget(spec, lora)
    memory = memory_budget(
        spec,
        lora,
        precision=args.precision,
        quantization=args.quantization,
        optimizer=args.optimizer,
    )
    full = full_finetune_memory_gb(spec, precision=args.precision, optimizer=args.optimizer)

    print("Computed, not measured: these follow from the architecture and the LoRA setting.\n")
    print(parameter_table([parameters]))
    print()
    print(memory_table([memory]))
    print(
        f"Full fine-tune for comparison: {full:.2f} GiB "
        f"({full / memory.total_gb:.1f}x the LoRA total)"
    )

    if args.quantization != "none":
        print(
            f"Quantization {args.quantization} compresses base weights "
            f"{compression_ratio(args.quantization, args.precision):.2f}x against {args.precision}"
        )

    if args.vram is not None:
        ok = fits_within(memory, args.vram)
        print(
            f"\nAgainst {args.vram:.0f} GiB with 15% reserved for excluded terms: "
            f"{'fits' if ok else 'does NOT fit'}"
        )

    if args.out:
        payload = budget_payload(parameters, memory, full)
        print(f"\nwrote {write_json(payload, Path(args.out) / 'budget.json')}")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    examples = load_examples(args.data)
    dataset = prepare(
        examples,
        ByteTokenizer(),
        template=args.template,
        max_length=args.max_length,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    print(dataset_table(dataset.stats))

    if args.out:
        print(f"\nwrote {write_json(dataset.stats, Path(args.out) / 'dataset.json')}")

    if dataset.stats.n_leaked:
        print(
            f"\nerror: {dataset.stats.n_leaked} evaluation example(s) also appear in training: "
            f"{', '.join(dataset.stats.leaked_ids[:5])}",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    examples = load_examples(args.data)
    dataset = prepare(
        examples,
        ByteTokenizer(),
        template=args.template,
        max_length=args.max_length,
        seed=args.seed,
    )

    config = TrainingConfig(
        model=args.model,
        lora=_lora_from(args),
        template=args.template,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )
    model, run = train(dataset, config, notes=args.notes)

    markdown = run_report(run)
    print(markdown)

    registry = RunRegistry(args.runs_dir)
    print(f"recorded {registry.save(run)}")

    if args.adapter_out:
        print(f"saved adapter to {save_adapter(model, args.adapter_out)}")
    if args.out:
        print(f"wrote {write_json(run, Path(args.out) / f'{run.run_id}.json')}")
        print(f"wrote {write_text(markdown, Path(args.out) / f'{run.run_id}.md')}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    examples = load_examples(args.data)
    dataset = prepare(examples, ByteTokenizer(), template=args.template, max_length=args.max_length)
    adapted = load_adapter(args.model, args.adapter)
    model: CausalLM = merge_adapter(adapted) if args.merge else adapted

    metrics = evaluate(model, dataset.eval, batch_size=args.batch_size)
    print(f"held-out examples : {metrics.n_examples}")
    print(f"loss              : {metrics.loss:.6f}")
    print(f"perplexity        : {metrics.perplexity:.4f}")
    print(f"token accuracy    : {metrics.token_accuracy:.6f}")
    print(f"merged            : {args.merge}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from ftlab.specs import build_model

    tokenizer = ByteTokenizer()
    model: CausalLM
    if args.adapter:
        adapted = load_adapter(args.model, args.adapter)
        model = merge_adapter(adapted) if args.merge else adapted
    else:
        model = build_model(get_spec(args.model))

    completion = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)
    print(completion)
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    registry = RunRegistry(args.runs_dir)
    summaries = registry.summaries()
    if not summaries:
        print(f"no runs in {args.runs_dir}")
        return 0
    for row in summaries:
        print(
            f"{row['run_id']}  {row['model']:14} r={row['r']:<3} {row['target']:<11} "
            f"loss {row['initial_loss']} -> {row['final_loss']}  "
            f"ppl {row['eval_perplexity_before']} -> {row['eval_perplexity_after']}"
        )
    print(f"\n{len(summaries)} run(s)")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    registry = RunRegistry(args.runs_dir)
    run_ids = args.run_ids or registry.run_ids()
    if len(run_ids) < 2:
        print("need at least two runs to compare", file=sys.stderr)
        return 1

    runs = [registry.load(run_id) for run_id in run_ids]
    print(comparison_table(runs))

    # Comparing across different models or datasets is not an ablation, and saying so is the
    # difference between a comparison and a misleading table.
    warnings: list[str] = []
    for other in runs[1:]:
        ok, differences = comparable(runs[0], other)
        if not ok:
            warnings.append(f"{other.run_id[-12:]}: {'; '.join(differences)}")
    if warnings:
        print("\nNot directly comparable to the first run:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


def _cmd_quantization(args: argparse.Namespace) -> int:
    support = quantization_available()
    print(f"bitsandbytes installed : {support.bitsandbytes_installed}")
    print(f"CUDA available         : {support.cuda_available}")
    if support.device_name:
        print(f"device                 : {support.device_name}")
    print(f"4-bit usable           : {support.usable}")
    if not support.usable:
        print(f"reason                 : {support.reason}")
    for quantization in ("int8", "nf4"):
        print(
            f"{quantization} compression vs {args.precision}: "
            f"{compression_ratio(quantization, args.precision):.2f}x"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "specs": _cmd_specs,
        "budget": _cmd_budget,
        "prepare": _cmd_prepare,
        "train": _cmd_train,
        "evaluate": _cmd_evaluate,
        "generate": _cmd_generate,
        "runs": _cmd_runs,
        "compare": _cmd_compare,
        "quantization": _cmd_quantization,
    }
    try:
        return handlers[args.command](args)
    except (
        FileNotFoundError,
        ValueError,
        SpecError,
        RegistryError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
