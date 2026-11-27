"""Explicit random-stream snapshots for epoch-boundary experiments.

Not automatically enabled by the legacy trainer. Worker-local streams cannot
be captured here; callers must use workers=0 or separately prove worker replay.
"""
import copy
import random
import numpy as np
import torch


def capture_random_state(dataset_rng, loader_generator, *, workers=0):
    if workers != 0:
        raise ValueError('Exact stream snapshot currently requires workers=0')
    return dict(version=1,python=random.getstate(),numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                dataset=copy.deepcopy(dataset_rng.bit_generator.state),
                loader=loader_generator.get_state().clone())


def restore_random_state(state, dataset_rng, loader_generator, *, workers=0):
    if workers != 0 or state.get('version') != 1:
        raise ValueError('Unsupported random-state protocol')
    cuda_states=state['torch_cuda']
    device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states)!=device_count:
        raise ValueError('CUDA device count differs from snapshot')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
    dataset_rng.bit_generator.state=copy.deepcopy(state['dataset'])
    loader_generator.set_state(state['loader'].cpu())
