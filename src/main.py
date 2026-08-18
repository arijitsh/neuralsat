import argparse
import csv
import logging
import warnings
import torch
import time
import os

from helper.network.read_onnx import parse_onnx
from helper.network.read_pth import parse_pth
from helper.spec.objective import parse_vnnlib


from helper.misc.logger import logger, LOGGER_LEVEL
from helper.misc.export import get_adv_string
from helper.misc.result import ReturnStatus

from verifier.verifier import Verifier
from verifier.utils import normalized_diversity

from setting import Settings


def activation_diversity(net: torch.nn.Module, samples: list[torch.Tensor], device: str) -> tuple[float | None, int, list[float]]:
    """Return aggregate and per-ReLU-layer normalized Hamming diversity."""
    if len(samples) < 2:
        return 0.0, 0, []

    activations = []
    hooks = []
    for module in net.modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(
                lambda _module, _args, output: activations.append((output > 0).flatten(start_dim=1).detach().cpu())
            ))

    if not hooks:
        return None, 0, []

    try:
        with torch.no_grad():
            net(torch.cat([sample.to(device) for sample in samples], dim=0))
    finally:
        for hook in hooks:
            hook.remove()

    pattern = torch.cat(activations, dim=1)
    layerwise = [
        (torch.pdist(layer.float(), p=1) / layer.shape[1]).mean().item()
        for layer in activations
    ]
    return (torch.pdist(pattern.float(), p=1) / pattern.shape[1]).mean().item(), pattern.shape[1], layerwise


if __name__ == '__main__':
    START_TIME = time.time()

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--input_shape', type=int, nargs='+', default=None,
                        help="Input shape of network, e.g., --input_shape 1 3 32 32")
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                        help="Output shape of network, e.g., --output_shape 1 10")
    parser.add_argument('--batch', type=int, default=1000,
                        help="maximum number of branches to verify in each iteration")
    parser.add_argument('--timeout', type=float, default=3600,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda',
                        help="choose device to use for verifying.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='show detailed sampler logs and model information.')
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=None,
                        help='the logger level (0: NOTSET, 1: INFO, 2: DEBUG).')
    parser.add_argument('--result_file', type=str, required=False,
                        help="file to save execution results.")
    parser.add_argument('--export_cex', action='store_true',
                        help="enable exporting counter-example to result file.")
    parser.add_argument('--disable_attack', action='store_false',
                        help="disable attack.")
    parser.add_argument('--disable_restart', action='store_false',
                        help="disable RESTART heuristic.")
    parser.add_argument('--disable_stabilize', action='store_false',
                        help="disable STABILIZE heuristic.")
    parser.add_argument('--force_split', type=str, choices=['input', 'hidden'],
                        help="select SPLITTING strategy.")
    parser.add_argument('--reasoning_output', type=str, required=False,
                        help="file to save reasoning steps.")
    parser.add_argument('--setting_file', type=str, required=False,
                        help="file to load specific settings.")
    parser.add_argument('--test', action='store_true',
                        help="test on small example with special settings.")
    parser.add_argument('--export_runtime', action='store_true', required=False,
                        help="output runtime.")
    parser.add_argument('-s', '--num_samples', type=int, default=None,
                        help="enable diversity sampling: collect this many distinct counter-examples instead of a single verification answer.")
    parser.add_argument('--sampling-engine', '--sampling_engine', dest='sampling_engine',
                        type=str, choices=['highdiv', 'random', 'auto'], default='auto',
                        help="sampling strategy: 'random' (multi-seed attack only), 'highdiv' (stochastic DPLL only), "
                             "or 'auto' (try random, fall back to highdiv if the samples are not diverse enough).")
    parser.add_argument('--diversity_threshold', type=float, default=None,
                        help="'auto' engine: normalized mean pairwise distance below which the random samples are rejected (default: 0.05).")
    parser.add_argument('--sample_output', type=str, required=False,
                        help="file to save sampled counter-examples to (.npy).")
    parser.add_argument('--csv-samples', type=str, required=False,
                        help="file to save sampled counter-examples and their network outputs to (.csv).")
    parser.add_argument('--diversity_prob', type=float, default=None,
                        help="probability of random neuron/phase selection in stochastic DPLL (default: 0.3).")
    parser.add_argument('--diversity_bam_n', type=int, default=None,
                        help="number of PAIS polytope samples per activation region (default: 5).")
    parser.add_argument('--diversity_blocking_topk', type=int, default=None,
                        help="block only the top-k most constrained neurons instead of the full activation pattern.")

    args = parser.parse_args()
    quiet_sampler = args.num_samples is not None and not (args.verbose or args.verbosity is not None)
    Settings.setup(args)
    Settings.quiet_sampler = quiet_sampler
    if not quiet_sampler:
        print(Settings)
    
    if os.environ.get('NEURALSAT_SYNTHETIC_BUG_DROP_PROBABILITY'):
        assert Settings.use_save_reasoning_step, 'Reasoning step is required for synthetic bug'
        
    # set device
    if not torch.cuda.is_available():
        args.device = 'cpu'
        
    # set logger level
    logger.setLevel(logging.WARNING if quiet_sampler else (
        logging.DEBUG if args.verbose or args.verbosity is None else LOGGER_LEVEL[args.verbosity]
    ))
    
    # network
    if args.net.endswith('.onnx'):
        model, input_shape, output_shape = parse_onnx(args.net, args.input_shape, args.output_shape)
    elif args.net.endswith('.pth'):
        model, input_shape, output_shape = parse_pth(args.net, args.input_shape, args.output_shape)
    else:
        raise NotImplementedError('Unsupported network type')
    
    logger.debug(f'net path: {args.net}')
    logger.debug(f'spec path: {args.spec}')
    logger.debug(f'timeout: {args.timeout}')
    logger.debug(f'device: {args.device}')
    
    model.to(args.device)

    if not quiet_sampler and (args.verbose or args.verbosity is None or args.verbosity):
        print(model)
    if quiet_sampler:
        print(f'c Input shape: {input_shape}')
        print(f'c Output shape: {output_shape}')
    else:
        logger.info(f'[!] Input shape: {input_shape}')
        logger.info(f'[!] Output shape: {output_shape}')
    
    # specification
    objectives = parse_vnnlib(args.spec, input_shape)
    if quiet_sampler:
        output_count = objectives.cs[0].shape[-1] if isinstance(objectives.cs, list) else objectives.cs.shape[-1]
        print(f'c VNNLIB: {objectives.lower_bounds.shape[-1]} inputs, {output_count} outputs')
    
    # verifier
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=args.batch,
        device=args.device,
    )
    
    
    # sample
    if args.num_samples is not None:
        Settings.sampling_engine = args.sampling_engine
        if args.diversity_threshold is not None:
            Settings.diversity_threshold = args.diversity_threshold
        if args.diversity_prob is not None:
            Settings.diversity_stochastic_prob = args.diversity_prob
        if args.diversity_bam_n is not None:
            Settings.diversity_bam_n = args.diversity_bam_n
        if args.diversity_blocking_topk is not None:
            Settings.diversity_blocking_topk = args.diversity_blocking_topk

        timeout = args.timeout - (time.time() - START_TIME)
        if quiet_sampler and args.sampling_engine in ('random', 'auto'):
            pre_attack_timeout = min(20.0, timeout * 0.5)
            print(f"c Starting multi-seed pre-attack for {pre_attack_timeout:.1f}s (engine='{args.sampling_engine}')")
        samples = verifier.sample_violations(
            dnf_objectives=objectives,
            k=args.num_samples,
            timeout=timeout,
            force_split=args.force_split,
        )
        runtime = time.time() - START_TIME

        score = normalized_diversity(samples, verifier.sampling_lower, verifier.sampling_upper)
        neuron_score, neuron_count, layerwise_neuron_scores = activation_diversity(verifier.net, samples, args.device)
        if quiet_sampler:
            print(f'c Collected {len(samples)}/{args.num_samples} samples in {runtime:.04f}s')
            if verifier.random_inputs_tested:
                print(
                    f'c Needed {verifier.random_inputs_tested} random generations to get '
                    f'{getattr(verifier, "random_samples_found", 0)} random samples '
                    f'({verifier.random_inputs_failed} rejected)'
                )
            print(f'c Diversity (normalized mean pairwise L2): {score:.04f}')
            print('c Activation diversity: unavailable (no ReLU modules found)'
                  if neuron_score is None else f'c Activation diversity: {neuron_score:.04f}')
            if neuron_score is not None:
                formatted_layerwise = ' '.join(f'{score:.04f}' for score in layerwise_neuron_scores)
                print(f'c Layerwise activation diversity: [{formatted_layerwise}]')
        else:
            logger.info(f'[!] Collected {len(samples)}/{args.num_samples} samples in {runtime:.04f}s')
            logger.info(f'[!] Diversity (normalized mean pairwise L2): {score:.04f}')
            if neuron_score is None:
                logger.info('[!] Activation diversity: unavailable (no ReLU modules found)')
            else:
                logger.info(
                    f'[!] Activation diversity (mean pairwise Hamming over {neuron_count} ReLU neurons): '
                    f'{neuron_score:.04f}'
                )

        if args.sample_output:
            import numpy as np
            np.save(args.sample_output, torch.stack([s.cpu() for s in samples]).numpy() if samples else np.empty(0))
            if quiet_sampler:
                print(f'c Saved {len(samples)} samples to {args.sample_output}')
            else:
                print(f'[!] Saved {len(samples)} samples to {args.sample_output}')

        if args.csv_samples:
            input_names = [f'X_{i}' for i in range(int(torch.tensor(input_shape).prod()))]
            output_names = [f'Y_{i}' for i in range(int(torch.tensor(output_shape).prod()))]
            with open(args.csv_samples, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([*input_names, *output_names])
                for sample in samples:
                    flat_input = sample.flatten().detach().cpu()
                    flat_output = verifier.net(sample.to(args.device)).flatten().detach().cpu()
                    writer.writerow([*flat_input.tolist(), *flat_output.tolist()])
            if quiet_sampler:
                print(f'c Saved {len(samples)} samples with outputs to {args.csv_samples}')
            else:
                print(f'[!] Saved {len(samples)} samples with outputs to {args.csv_samples}')

        if not quiet_sampler:
            print(f'{len(samples)},{runtime:.04f}')
        exit(0)

    # verify
    timeout = args.timeout - (time.time() - START_TIME)
    status = verifier.verify(objectives, timeout=timeout, force_split=args.force_split)
    runtime = time.time() - START_TIME

    # output
    logger.info(f'[!] Iterations: {verifier.iteration}')
    if verifier.adv is not None:
        logger.info(f'adv (first 5): {verifier.adv.flatten()[:5].detach().cpu()}')
        logger.debug(f'output: {verifier.net(verifier.adv).flatten().detach().cpu()}')
        
    # export
    if args.result_file:
        os.remove(args.result_file) if os.path.exists(args.result_file) else None
        with open(args.result_file, 'w') as fp:
            if args.export_runtime:
                print(f'{status},{runtime:.04f}', file=fp)
            else:
                print(status, file=fp)
            if (verifier.adv is not None) and args.export_cex:
                print(get_adv_string(inputs=verifier.adv, net_path=args.net), file=fp)

    if args.reasoning_output and Settings.use_save_reasoning_step and status == ReturnStatus.UNSAT:
        if hasattr(verifier, 'domains_list') and not isinstance(verifier.domains_list, list):
            verifier.domains_list.reasoning_domains.export_aptp(args.reasoning_output)
        else:
            print(f'[!] Does not have any reasoning step')

    logger.info(f'[!] Result: {status}')
    logger.info(f'[!] Runtime: {runtime:.04f}')
    
    print(f'{status},{runtime:.04f}')

    if os.environ.get('NEURALSAT_SYNTHETIC_BUG_DROP_PROBABILITY'):
        print('[!] Synthetic bug is enabled for demonstatration purpose. Do not enable for benchmarking.')
        
