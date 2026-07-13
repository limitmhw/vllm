# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import regex as re
import torch


def deep_compare(dict1: Any, dict2: Any) -> bool:
    if type(dict1) is not type(dict2):
        return False
    if isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            return False
        return all(deep_compare(dict1[k], dict2[k]) for k in dict1)
    elif isinstance(dict1, list):
        # `dict1` may be a list of dict.
        return all(deep_compare(dict1[i], dict2[i]) for i in range(len(dict1)))
    else:
        return dict1 == dict2


def should_ignore_layer(
    layer_name: str | None,
    ignore: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    if layer_name is None:
        return False

    # layer_name = model.layers.0.self_attn.qkv_proj
    # proj_name = qkv_proj
    proj_name = layer_name.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_proj_names = fused_mapping[proj_name]

        # Convert fused_name --> [shard_names]
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)
            for shard_proj_name in shard_proj_names
        ]

        # Layer should be ignored if shards are ignored.
        should_ignore_layer = None
        for shard_name in shard_names:
            should_ignore_shard = check_equal_or_regex_match(
                layer_name=shard_name, targets=ignore
            )

            # If shard_idx=0, set layer ignore to match shard.
            if should_ignore_layer is None:
                should_ignore_layer = should_ignore_shard

            # If shard_idx=1+ confirm scheme matches prior shards.
            elif should_ignore_shard != should_ignore_layer:
                raise ValueError(
                    f"Found a different quantization schemes for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme."
                )

    # Unfused layers like down_proj and o_proj will match
    # the safetensors checkpoint already.
    else:
        should_ignore_layer = check_equal_or_regex_match(
            layer_name=layer_name, targets=ignore
        )

    assert should_ignore_layer is not None
    return should_ignore_layer


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.
    """
    return any(_is_equal_or_regex_match(layer_name, target) for target in targets)


def _is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


def dequantize_quark_shared_expert_gate(
    qweight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    values = qweight.to(torch.int32) & 0xF
    values = torch.where(values >= 8, values - 16, values)
    scales = scales.to(torch.float32).repeat_interleave(128, dim=0)
    return (values.to(torch.float32) * scales).T.contiguous()


def dequantize_quark_shared_expert_gate_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    pending_weights: dict[str, torch.Tensor] = {}
    pending_scales: dict[str, torch.Tensor] = {}
    dequantized_prefixes: set[str] = set()

    def maybe_emit(prefix: str):
        qweight = pending_weights.pop(prefix, None)
        scales = pending_scales.pop(prefix, None)
        if qweight is None or scales is None:
            if qweight is not None:
                pending_weights[prefix] = qweight
            if scales is not None:
                pending_scales[prefix] = scales
            return None
        dequantized_prefixes.add(prefix)
        return prefix + ".weight", dequantize_quark_shared_expert_gate(qweight, scales)

    for name, weight in weights:
        if name.endswith(".shared_expert_gate.weight_scale") or name.endswith(
            ".shared_expert_gate.scales"
        ):
            prefix = name.rsplit(".", 1)[0]
            pending_scales[prefix] = weight
            emitted = maybe_emit(prefix)
            if emitted is not None:
                yield emitted
            continue
        if (
            name.endswith(".shared_expert_gate.weight")
            or name.endswith(".shared_expert_gate.qweight")
        ) and weight.dtype in (torch.int32, torch.int64):
            prefix = name.rsplit(".", 1)[0]
            pending_weights[prefix] = weight
            emitted = maybe_emit(prefix)
            if emitted is not None:
                yield emitted
            continue
        if ".shared_expert_gate." in name and (
            name.endswith(".weight_zero_point")
            or name.endswith(".qzeros")
            or name.endswith(".qqzeros")
        ):
            continue
        yield name, weight

    for prefix, weight in pending_weights.items():
        if prefix not in dequantized_prefixes:
            yield prefix + ".weight", weight
    for prefix, scales in pending_scales.items():
        if prefix not in dequantized_prefixes:
            yield prefix + ".weight_scale", scales


# utility for tensor dims > 2 cases
def quark_quantize_weight_to_mxfp4(w: torch.Tensor):
    assert w.dtype == torch.bfloat16, (
        "Quark dynamic quantization is supported only for fp16 weights and only to MXF4"
    )

    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    *dims, d = w.shape
    w, w_scales = dynamic_mxfp4_quant(w.reshape(-1, d))
    return w.view(*dims, d // 2), w_scales.view(*dims, d // 32)
