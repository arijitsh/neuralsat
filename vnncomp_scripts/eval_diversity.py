#!/usr/bin/env python3
"""Evaluate sample diversity and optionally export ReLU activation patterns.

Examples:
    python vnncomp_scripts/eval_diversity.py \
        --net model.onnx --csv-samples samples.csv

    python vnncomp_scripts/eval_diversity.py \
        --net model.onnx --csv-samples samples.csv --show-activation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / 'src'))

from helper.network.read_onnx import parse_onnx


def load_inputs(path: Path, expected_inputs: int) -> np.ndarray:
    """Read X_0 ... X_n columns from a CSV samples file."""
    with path.open(newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f'{path} has no header row')

        input_columns = [f'X_{index}' for index in range(expected_inputs)]
        missing = [column for column in input_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f'{path} is missing input columns: {", ".join(missing)}')

        rows = [[float(row[column]) for column in input_columns] for row in reader]

    if not rows:
        raise ValueError(f'{path} contains no samples')
    return np.asarray(rows, dtype=np.float32)


def diversity_metrics(inputs: torch.Tensor) -> tuple[float, float, float]:
    """Return mean pairwise distance, minimum pairwise distance, and coverage radius."""
    if len(inputs) < 2:
        return 0.0, 0.0, 0.0

    distances = torch.cdist(inputs, inputs)
    pairwise = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
    nearest = distances.masked_fill(torch.eye(len(inputs), dtype=torch.bool), float('inf')).min(dim=1).values
    return pairwise.mean().item(), pairwise.min().item(), nearest.max().item()


def activation_patterns(model: torch.nn.Module, inputs: torch.Tensor) -> list[str]:
    """Return one named bit-pattern string per sample for every ReLU layer."""
    layer_outputs: list[tuple[str, torch.Tensor]] = []
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(
                lambda _module, _args, output, layer_name=name:
                layer_outputs.append((layer_name, output.detach().cpu()))
            ))

    if not hooks:
        return [''] * len(inputs)

    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for hook in hooks:
            hook.remove()

    patterns = []
    for sample_index in range(len(inputs)):
        layers = []
        for name, output in layer_outputs:
            bits = ''.join('1' if value > 0 else '0' for value in output[sample_index].flatten().tolist())
            layers.append(f'{name}={bits}')
        patterns.append(';'.join(layers))
    return patterns


def write_augmented_csv(path: Path, inputs: torch.Tensor, outputs: torch.Tensor,
                        patterns: list[str]) -> None:
    header = (
        [f'X_{index}' for index in range(inputs.shape[1])]
        + [f'Y_{index}' for index in range(outputs.shape[1])]
        + ['activation_pattern']
    )
    with path.open('w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for sample, output, pattern in zip(inputs, outputs, patterns):
            writer.writerow([*sample.tolist(), *output.tolist(), pattern])


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate diversity of CSV samples for an ONNX network.')
    parser.add_argument('--net', required=True, help='ONNX network path.')
    parser.add_argument('--csv-samples', required=True, help='CSV containing X_0, X_1, ... input columns.')
    parser.add_argument('--show-activation', action='store_true',
                        help='Write diversity-samples.csv with outputs and named ReLU activation patterns.')
    args = parser.parse_args()

    model, input_shape, _ = parse_onnx(args.net, None, None)
    model.eval()
    expected_inputs = int(np.prod(input_shape))
    input_array = load_inputs(Path(args.csv_samples), expected_inputs)
    inputs = torch.from_numpy(input_array)

    sample_shape = tuple(input_shape[1:])
    with torch.no_grad():
        outputs = model(inputs.reshape(len(inputs), *sample_shape)).flatten(start_dim=1).cpu()

    mean_pairwise, min_pairwise, coverage_radius = diversity_metrics(inputs)
    print(f'samples: {len(inputs)}')
    print(f'mean_pairwise_l2: {mean_pairwise:.8f}')
    print(f'min_pairwise_l2: {min_pairwise:.8f}')
    print(f'coverage_radius: {coverage_radius:.8f}')

    if args.show_activation:
        output_path = Path('diversity-samples.csv')
        patterns = activation_patterns(model, inputs.reshape(len(inputs), *sample_shape))
        write_augmented_csv(output_path, inputs, outputs, patterns)
        print(f'wrote: {output_path}')


if __name__ == '__main__':
    main()
