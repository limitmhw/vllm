# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm import _custom_ops as ops, envs
from vllm.model_executor.layers.quantization.quark.schemes.quark_scheme import (
    QuarkScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)


class QuarkW4A16Int4(QuarkScheme):
    """Quark packed INT4 weight-only linear scheme."""

    def __init__(self, group_size: int, pack_method: str):
        self.group_size = group_size
        self.pack_factor = 8
        self.pack_reorder = pack_method == "reorder"

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        group_size = (
            self.group_size if self.group_size != -1 else input_size_per_partition
        )
        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )

        weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        num_groups = input_size_per_partition // group_size
        weight_zero_point = PackedvLLMParameter(
            data=torch.zeros(
                num_groups,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_zero_point", weight_zero_point)
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_zero_point = torch.nn.Parameter(
            layer.weight_zero_point.data, requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        qweight = layer.weight
        scales = layer.weight_scale
        qzeros = layer.weight_zero_point
        out_shape = x.shape[:-1] + (qweight.shape[-1] * self.pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        if x.shape[:-1].numel() >= 256 or envs.VLLM_BATCH_INVARIANT:
            out = ops.awq_dequantize(
                qweight,
                scales,
                qzeros,
                0,
                0,
                0,
                signed_int4=True,
                pack_reorder=self.pack_reorder,
            )
            out = torch.matmul(reshaped_x, out)
        else:
            out = ops.awq_gemm(
                reshaped_x,
                qweight,
                scales,
                qzeros,
                self.pack_factor,
                signed_int4=True,
                pack_reorder=self.pack_reorder,
            )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
