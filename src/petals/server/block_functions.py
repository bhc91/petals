"""
This module implements server-side computations on served blocks: forward, backward and inference; used by handler
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Union

import torch
from hivemind.compression.serialization import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger
from hivemind.utils.nested import nested_flatten

from petals.data_structures import Handle, InferenceMetadata
from petals.server.backend import TransformerBackend
from petals.server.task_pool import PrioritizedTaskPool
from petals.server.task_prioritizer import TaskPrioritizerBase
from petals.utils.convert_block import QuantType
from petals.utils.misc import DUMMY, is_dummy
from petals.utils.packaging import pack_args_kwargs, unpack_args_kwargs

# We prioritize short inference requests and make them use a *merged* inference pool,
# so they are processed without interruptions and extra overheads
# TODO: Increase the NF4 threshold once bitsandbytes ships efficient NF4 kernel for parallel forward
MAX_SHORT_INFERENCE_TOKENS = 128
MAX_NF4_SHORT_INFERENCE_TOKENS = 1

logger = get_logger(__name__)


async def run_rpc_forward(
    *flat_tensors: torch.Tensor,
    requested_backends: Sequence[TransformerBackend],
    active_adapter: str = "",
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
    args_structure: Any,
) -> torch.Tensor:
    """
    Run forward pass on deserialized inputs and prompts, used by rpc_forward and rpc_forward_stream

    :param flat_tensors: a list of tensors that includes first layer inputs, optional prompts and extra tensors
    :note: some input tensors can be missing, in which case they will be replaced with dummy tensors (see is_dummy)
    :param requested_backends: a sequence of transformer blocks in the same order as they appear in forward pass
    :returns: hidden states after the last layer [batch_size, seq_length, hid_size]
    """
    (hidden_states, prompts), backend_kwargs = _check_inputs(requested_backends, flat_tensors, args_structure)
    dtype = requested_backends[0].dtype
    # check parse input tensors and cast dtypes
    hidden_states = hidden_states.to(dtype)
    assert hidden_states.ndim == 3
    num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
    if prompts is None or is_dummy(prompts):
        prompts = [DUMMY] * len(requested_backends)
    else:
        prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

    # Run a chain of requested backends
    for backend, prompt, kwargs in zip(requested_backends, prompts, backend_kwargs):
        if not is_dummy(prompt):
            hidden_states[:, : prompt.shape[1]] += prompt

        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            hidden_states, points=points / len(requested_backends), backend=backend, type="forward"
        )
        (hidden_states,) = await backend.forward_pool.submit_task(
            active_adapter,
            hidden_states,
            **kwargs,
            priority=priority,
            size=num_tokens,
        )
        assert isinstance(hidden_states, torch.Tensor)
        assert (
            hidden_states.ndim == 3
        ), f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"

    return hidden_states


async def run_rpc_backward(
    *flat_tensors: torch.Tensor,
    requested_backends: Sequence[TransformerBackend],
    active_adapter: str = "",
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
    args_structure: Any,
) -> Tuple[Sequence[torch.Tensor], Any]:
    (hidden_states, grad_outputs, prompts), backend_kwargs = _check_inputs(
        requested_backends, flat_tensors, args_structure
    )
    # Cast inputs & grad outputs to backend dtype
    assert hidden_states.ndim == 3
    num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
    hidden_states = hidden_states.to(requested_backends[0].dtype)
    grad_outputs = grad_outputs.to(requested_backends[-1].dtype)

    if prompts is None or is_dummy(prompts):
        prompts = [DUMMY] * len(requested_backends)
    else:
        prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

    # Run a forward chain to collect intermediate inputs
    # Note that we do not forward for the last module since we do not need its output
    inter_inputs = []
    for backend, prompt, kwargs in zip(requested_backends[:-1], prompts[:-1], backend_kwargs):
        assert hidden_states.ndim == 3, f"inputs to {type(backend)} must be a single 3d tensor of hidden states"
        if not is_dummy(prompt):
            hidden_states[:, : prompt.shape[1]] += prompt
        inter_inputs.append(hidden_states)
        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            hidden_states, points=points / len(requested_backends), backend=backend, type="forward_in_backward"
        )
        (hidden_states,) = await backend.forward_pool.submit_task(
            active_adapter, hidden_states, **kwargs, priority=priority, size=num_tokens
        )

        assert isinstance(hidden_states, torch.Tensor)

    if not is_dummy(prompts[-1]):
        hidden_states[:, : prompts[-1].shape[1]] += prompts[-1]
    inter_inputs.append(hidden_states)

    assert len(inter_inputs) == len(prompts) == len(requested_backends), "internal shape error during backward"
    grad_prompts_reversed = []
    grad_backend_kwargs_reversed = []

    # Run a chain of requested backends
    for inp, prompt, backend, kwargs in reversed(list(zip(inter_inputs, prompts, requested_backends, backend_kwargs))):
        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            inp, grad_outputs, points=points / len(requested_backends), backend=backend, type="backward"
        )
        (*grad_outputs, grad_kwargs) = await backend.backward_pool.submit_task(
            active_adapter, grad_outputs, inp, **kwargs, priority=priority, size=num_tokens
        )
        assert isinstance(grad_outputs, torch.Tensor)
        if not is_dummy(prompt):
            grad_prompts_reversed.append(grad_outputs[:, : prompt.shape[1]].unsqueeze(0))
        grad_backend_kwargs_reversed.append(grad_kwargs)

    grad_prompts = torch.cat(grad_prompts_reversed[::-1], dim=0) if grad_prompts_reversed else DUMMY
    grad_args = [grad_outputs] if is_dummy(grad_prompts) else [grad_outputs, grad_prompts]
    return pack_args_kwargs((grad_args, reversed(grad_backend_kwargs_reversed)))


async def iterate_rpc_inference(
    requested_uids: Sequence[ExpertUID],
    requested_backends: Sequence[TransformerBackend],
    active_adapter: Optional[str],
    input_iterator: AsyncIterator[Tuple[runtime_pb2.ExpertRequest, dict]],
    cache_handles: Sequence[Sequence[Handle]],
    *,
    max_length: int,
    prioritizer: TaskPrioritizerBase,
    points: int,
    quant_type: QuantType,
    args_structure: Any = None,
) -> AsyncIterator[Tuple[Sequence[runtime_pb2.Tensor], bool]]:
    assert len(cache_handles) == len(requested_backends)

    prefix_length = 0
    point_per_piece = points / max_length if max_length > 0 else 0.0

    async for request, step_metadata in input_iterator:
        flat_tensors = tuple(deserialize_torch_tensor(tensor) for tensor in request.tensors)
        (hidden_states, prompts, hypo_ids), backend_kwargs = _check_inputs(
            requested_backends, flat_tensors, args_structure
        )
        batch_size, length_increment, _ = hidden_states.shape
        num_tokens = batch_size * length_increment

        # Cast inputs to backend dtype
        hidden_states = hidden_states.to(requested_backends[0].dtype)
        assert hypo_ids.dtype == torch.int64, f"hypo ids must be int64, got {hypo_ids.dtype}"

        # parse deep prompts (optional argument)
        has_prompts = prompts is not None and not is_dummy(prompts)
        if not has_prompts:
            prompts = [None] * len(requested_backends)
        else:
            prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]
            prompts = [prompt if not is_dummy(prompt) else None for prompt in prompts]

        if not (len(requested_backends) == len(prompts)):
            raise ValueError(f"Received {len(prompts)} prompts for {len(requested_backends)} backends")

        if prefix_length + length_increment > max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {prefix_length} + current {length_increment}"
                f" exceeds pre-allocated maximum {max_length}"
            )

        merge_max_tokens = MAX_NF4_SHORT_INFERENCE_TOKENS if quant_type == QuantType.NF4 else MAX_SHORT_INFERENCE_TOKENS
        can_merge_pools = batch_size * length_increment <= merge_max_tokens
        priority = prioritizer.prioritize(
            hidden_states,
            hypo_ids,
            points=point_per_piece,
            requested_uids=requested_uids,
            type="inference",
        )

        # A client may pass a tensor with 0 tokens. This is a special case that occurs, e.g.
        # when user wants to pre-allocate cache or check that server *can* allocate that cache.
        if hidden_states.numel() > 0:
            assert hidden_states.ndim == 3, f"hidden states must be a single 3d tensor"
            if can_merge_pools:
                inference_infos = tuple(
                    InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter)
                    for uid, handles in zip(requested_uids, cache_handles)
                )
                (hidden_states,) = await requested_backends[0].inference_pool.submit_task(
                    hidden_states,
                    hypo_ids,
                    inference_infos,
                    *prompts,
                    backend_kwargs=backend_kwargs,
                    priority=priority,
                    size=num_tokens,
                )
            else:
                for backend, uid, handles, prompt, kwargs in zip(
                    requested_backends, requested_uids, cache_handles, prompts, backend_kwargs
                ):
                    inference_infos = (InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter),)
                    (hidden_states,) = await backend.inference_pool.submit_task(
                        hidden_states,
                        hypo_ids,
                        inference_infos,
                        prompt,
                        backend_kwargs=(kwargs,),
                        priority=priority,
                        size=num_tokens,
                    )

        # serialize and send last layer outputs
        output_tensors = [
            serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
            for result, proto in zip((hidden_states,), nested_flatten(requested_backends[-1].outputs_schema))
        ]
        can_push = not has_prompts
        yield output_tensors, can_push

        # prepare for next step
        prefix_length += length_increment


def _check_inputs(
    requested_backends: Sequence[TransformerBackend], flat_tensors: Sequence[torch.Tensor], args_structure: Any
):
    if args_structure is not None:
        args, *backend_kwargs = unpack_args_kwargs(flat_tensors, args_structure)
    else:
        args, *backend_kwargs = flat_tensors, {}  # backward compatibility

    if len(backend_kwargs) not in (1, len(requested_backends)):
        raise RuntimeError(
            f"Server expected either one dict of keyword arguments or {len(requested_backends)} dicts "
            f"(one for each block). Found {len(backend_kwargs)} instead."
        )
    if len(backend_kwargs) == 1:
        backend_kwargs = backend_kwargs * len(requested_backends)
    assert len(backend_kwargs) == len(requested_backends)
    for i, kwargs in enumerate(backend_kwargs):
        if not isinstance(kwargs, dict):
            raise RuntimeError(f"Expected kwargs for block {i} to be a dictionary, got {type(kwargs)}")
    return args, backend_kwargs
