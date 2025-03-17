# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch import svd_lowrank
from transformers.activations import ACT2FN
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    StaticCache,
)
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import (
    FlashAttentionKwargs,
    _flash_attention_forward,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    LossKwargs,
    replace_return_docstrings,
)
from typing import (
    Dict,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    Union,
)

from .loss.SimilarityCalculator import SimilarityCalculator
from .modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeModelOutputWithPast,
    MoSLECausalLMOutputWithPast,
)
from .MoSLEConfig import MoSLEConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def release_linears(self) -> None:
        """ Release the linear layers. (yxhong-tw added)
        """

        with torch.no_grad():
            if hasattr(
                    self,
                    'gate_proj',
            ) and self.gate_proj is not None:
                del self.gate_proj.weight
                del self.gate_proj.bias

            if hasattr(
                    self,
                    'up_proj',
            ) and self.up_proj is not None:
                del self.up_proj.weight
                del self.up_proj.bias

            if hasattr(
                    self,
                    'down_proj',
            ) and self.down_proj is not None:
                del self.down_proj.weight
                del self.down_proj.bias

        del self.gate_proj
        del self.up_proj
        del self.down_proj

        gc.collect()
        torch.cuda.empty_cache()


class MoSLEParameter(nn.Module):
    """ The parameter class for MoSLE model. (yxhong-tw added)
    """

    def __init__(
        self,
        data: torch.Tensor,
        requires_grad: bool = True,
    ) -> None:
        """ The constructor method for MoSLEParameter class.

        Args:
            data (torch.Tensor): The value of the parameter.
            requires_grad (bool, optional): Whether to require the gradient. Defaults to True.
        """

        super().__init__()

        self.parameter = nn.Parameter(
            data=data,
            requires_grad=requires_grad,
        )

    def forward(self) -> nn.Parameter:
        """ The forward method for MoSLEParameter class.

        Returns:
            nn.Parameter: The parameter.
        """

        return self.parameter


class MoSLELlamaMLP(nn.Module):
    """ The MoSLELlamaMLP class. (yxhong-tw added)
    """

    def __init__(
        self,
        base_mlp: LlamaMLP,
        mosle_config: MoSLEConfig,
    ) -> None:
        """ The constructor method for MoSLELlamaMLP class.

        Args:
            base_mlp (LlamaMLP): The base MLP.
            mosle_config (MoSLEConfig): The mosle configuration.
        """

        super().__init__()

        self.config = base_mlp.config
        self.hidden_size = base_mlp.hidden_size
        self.intermediate_size = base_mlp.intermediate_size
        self.act_fn = base_mlp.act_fn

        # The MoSLE configuration.
        self.mosle_config = mosle_config
        self.ex_num = self.mosle_config.ex_num
        self.jitter_noise = self.mosle_config.jitter_noise
        self.selected_ex_num = self.mosle_config.selected_ex_num

        # Create the necessary components for MoSLE.
        self.router = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.mosle_config.ex_num,
        )

        self.gate_proj_exs = self._create_exs(
            ffn_weight=base_mlp.gate_proj.weight,
            rank=mosle_config.ex_rank,
            ex_num=mosle_config.ex_num,
        )
        self.gate_bias = base_mlp.gate_proj.bias

        self.up_proj_exs = self._create_exs(
            ffn_weight=base_mlp.up_proj.weight,
            rank=mosle_config.ex_rank,
            ex_num=mosle_config.ex_num,
        )
        self.up_bias = base_mlp.up_proj.bias

        self.down_proj_exs = self._create_exs(
            ffn_weight=base_mlp.down_proj.weight,
            rank=mosle_config.ex_rank,
            ex_num=mosle_config.ex_num,
        )
        self.down_bias = base_mlp.down_proj.bias

        # Release the linear layers of the base MLP.
        base_mlp.release_linears()

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """ The forward method for MoSLELlamaMLP class.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[ torch.Tensor, torch.Tensor, ]: The output tensor and the routing logits.
        """

        batch_size, sequence_length, hidden_size = x.shape

        # Add jitter noise.
        if self.training and self.jitter_noise > 0.0:
            x *= torch.empty_like(input=x).uniform_(
                from_=(1.0 - self.jitter_noise),
                to=(1.0 + self.jitter_noise),
            )

        # New shape of x: (batch_size * sequence_length, hidden_size)
        x = x.view(-1, hidden_size)

        # Shape of routing_logits: (batch_size * sequence_length, ex_num)
        routing_logits = self.router(input=x)

        # Do softmax on ex_num dimension.
        routing_weights = F.softmax(
            input=routing_logits,
            dim=1,
            dtype=torch.float,
        )

        # Select the top-k experts.
        # Shape of new routing_weights: (batch_size * sequence_length, selected_ex_num)
        # Shape of selected_experts: (batch_size * sequence_length, selected_ex_num)
        routing_weights, selected_experts = torch.topk(
            input=routing_weights,
            k=self.selected_ex_num,
            dim=-1,
        )

        # Do softmax on selected_ex_num dimension.
        routing_weights /= routing_weights.sum(
            dim=-1,
            keepdim=True,
        )

        routing_weights.to(dtype=x.dtype)

        # Do not know why the parameter name is `input` in `F.one_hot` not `tensor`.
        # Shape of experts_mask: (batch_size * sequence_length, selected_ex_num, ex_num)
        experts_mask = F.one_hot(
            input=selected_experts,
            num_classes=self.ex_num,
        )

        # Shape of new experts_mask: (ex_num, selected_ex_num, batch_size * sequence_length)
        # The meaning of new experts_mask (i, j, k) is that the i-th expert is the k-th token's j-th selected expert.
        experts_mask = experts_mask.permute(2, 1, 0)

        # Shape of y: (batch_size * sequence_length, hidden_size)
        y = torch.zeros(
            size=x.shape,
            dtype=x.dtype,
            device=x.device,
        )

        # There is an example for the following code.
        # Suppose:
        # - batch_size * sequence_length is 5.
        # - selected_ex_num is 2.
        # And:
        # - experts_mask[100] is: tensor([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
        # - top_ex_idx and token_idx will be tensor([0, 1]) and tensor([0, 4]).
        # That mean the 0-th token's 0-th selected expert and the 4-th token's 1-th selected expert are the 100-th expert.
        for ex_idx in range(self.ex_num):
            top_ex_idx, token_idx = torch.where(experts_mask[ex_idx])

            # current_x = x[None, token_idx].reshape(-1, hidden_size)
            current_x = x[token_idx]

            gate_output = current_x @ self.gate_proj_exs[ex_idx]['b']()
            gate_output = gate_output @ self.gate_proj_exs[ex_idx]['a']().T

            if self.gate_bias is not None:
                gate_output = gate_output + self.gate_bias

            up_output = current_x @ self.up_proj_exs[ex_idx]['b']()
            up_output = up_output @ self.up_proj_exs[ex_idx]['a']().T

            if self.up_bias is not None:
                up_output = up_output + self.up_bias

            output = self.act_fn(gate_output) * up_output

            down_output = output @ self.down_proj_exs[ex_idx]['b']()
            down_output = down_output @ self.down_proj_exs[ex_idx]['a']().T

            if self.down_bias is not None:
                down_output = down_output + self.down_bias

            # Multiply the routing weights.
            down_output *= routing_weights[token_idx, top_ex_idx, None]

            y.index_add_(
                dim=0,
                index=token_idx,
                source=down_output.to(dtype=x.dtype),
            )

            # Normalize the output with selected_ex_num.
            y /= self.selected_ex_num

        # Reshape y to the original shape.
        y = y.reshape(shape=(batch_size, sequence_length, hidden_size))

        return (
            y,
            routing_logits,
        )

    @classmethod
    def _decompose_ffn(
        cls: Type[Self],
        ffn_weight: torch.Tensor,
        rank: int,
    ) -> Tuple[
        MoSLEParameter,
        MoSLEParameter,
    ]:
        """ Decompose the FFN weight into two parts.

        Args:
            cls (Type[Self]): The class.
            ffn_weight (torch.Tensor): The FFN weight.
            rank (int): The rank.

        Returns:
            Tuple[ MoSLEParameter, MoSLEParameter, ]: The two parts of the FFN weight.
        """

        with torch.no_grad():
            a, c, b = svd_lowrank(
                A=ffn_weight,
                q=rank,
            )
            b *= c

        a_param = MoSLEParameter(
            data=a,
            requires_grad=True,
        )
        b_param = MoSLEParameter(
            data=b,
            requires_grad=True,
        )

        return (
            a_param,
            b_param,
        )

    @classmethod
    def _create_exs(
        cls: Type[Self],
        ffn_weight: torch.Tensor,
        rank: int,
        ex_num: int,
    ) -> nn.ModuleList:
        """ Create the experts.

        Args:
            cls (Type[Self]): The class.
            ffn_weight (torch.Tensor): The FFN weight.
            rank (int): The rank.
            ex_num (int): The number of experts.

        Returns:
            nn.ModuleList: The experts.
        """

        a_param, b_param = cls._decompose_ffn(
            ffn_weight=ffn_weight,
            rank=rank,
        )

        exs = nn.ModuleList(modules=[
            nn.ModuleDict({
                'a': deepcopy(x=a_param),
                'b': deepcopy(x=b_param),
            }) for _ in range(ex_num)
        ])

        return exs


class MoSLELlamaMLPV2(nn.Module):
    """ The MoSLELlamaMLPV2 class. (yxhong-tw added)
    """

    def __init__(
        self,
        base_mlp: LlamaMLP,
        mosle_config: MoSLEConfig,
        rank: int,
        create_empty_proj: bool = False,
    ) -> None:
        """ The constructor method for MoSLELlamaMLPV2 class.

        Args:
            base_mlp (LlamaMLP): The base MLP.
            mosle_config (MoSLEConfig): The mosle configuration.
            rank (int): The rank.
            create_empty_proj (bool, optional): Whether to create empty projection. Defaults to False.
        """

        super().__init__()

        self.config = base_mlp.config
        self.hidden_size = base_mlp.hidden_size
        self.intermediate_size = base_mlp.intermediate_size
        self.act_fn = base_mlp.act_fn

        if create_empty_proj:
            self.gate_proj = self._create_empty_proj(
                head_dim=self.hidden_size,
                rank=rank,
                tail_dim=self.intermediate_size,
            )
            self.gate_bias = 0

            self.up_proj = self._create_empty_proj(
                head_dim=self.hidden_size,
                rank=rank,
                tail_dim=self.intermediate_size,
            )
            self.up_bias = 0

            self.down_proj = self._create_empty_proj(
                head_dim=self.intermediate_size,
                rank=rank,
                tail_dim=self.hidden_size,
            )
            self.down_bias = 0
        else:
            self.gate_proj = self._decompose_ffn(
                ffn_weight=base_mlp.gate_proj.weight,
                rank=rank,
            )
            self.gate_bias = base_mlp.gate_proj.bias

            self.up_proj = self._decompose_ffn(
                ffn_weight=base_mlp.up_proj.weight,
                rank=rank,
            )
            self.up_bias = base_mlp.up_proj.bias

            self.down_proj = self._decompose_ffn(
                ffn_weight=base_mlp.down_proj.weight,
                rank=rank,
            )
            self.down_bias = base_mlp.down_proj.bias

            # Release the linear layers of the base MLP.
            base_mlp.release_linears()

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """ The forward method for MoSLELlamaMLP class.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[ torch.Tensor, torch.Tensor, ]: The output tensor and the routing logits.
        """

        gate_output = x @ self.gate_proj['b']()
        gate_output = gate_output @ self.gate_proj['a']().T

        if self.gate_bias is not None:
            gate_output = gate_output + self.gate_bias

        up_output = x @ self.up_proj['b']()
        up_output = up_output @ self.up_proj['a']().T

        if self.up_bias is not None:
            up_output = up_output + self.up_bias

        output = self.act_fn(gate_output) * up_output

        down_output = output @ self.down_proj['b']()
        down_output = down_output @ self.down_proj['a']().T

        if self.down_bias is not None:
            down_output = down_output + self.down_bias

        return down_output

    @classmethod
    def _decompose_ffn(
        cls: Type[Self],
        ffn_weight: torch.Tensor,
        rank: int,
    ) -> Dict[
        str,
        MoSLEParameter,
    ]:
        """ Decompose the FFN weight into two parts.

        Args:
            cls (Type[Self]): The class.
            ffn_weight (torch.Tensor): The FFN weight.
            rank (int): The rank.

        Returns:
            Dict[ str, MoSLEParameter, ]: The two parts of the FFN weight.
        """

        with torch.no_grad():
            a, c, b = svd_lowrank(
                A=ffn_weight,
                q=rank,
            )
            b *= c

        a_param = MoSLEParameter(
            data=a,
            requires_grad=True,
        )
        b_param = MoSLEParameter(
            data=b,
            requires_grad=True,
        )

        return nn.ModuleDict({
            'a': a_param,
            'b': b_param,
        })

    @classmethod
    def _create_empty_proj(
        cls: Type[Self],
        head_dim: int,
        rank: int,
        tail_dim: int,
    ) -> Dict[
        str,
        MoSLEParameter,
    ]:
        a = torch.zeros(size=(tail_dim, rank))
        a_param = MoSLEParameter(
            data=a,
            requires_grad=True,
        )

        b = torch.zeros(size=(head_dim, rank))
        b_param = MoSLEParameter(
            data=b,
            requires_grad=True,
        )

        return nn.ModuleDict({
            'a': a_param,
            'b': b_param,
        })


class MoSLEMoE(nn.Module):
    def __init__(
        self,
        base_mlp: LlamaMLP,
        mosle_config: MoSLEConfig,
        rank: int,
    ) -> None:
        """ The constructor method for MoSLEMoE class.

        Args:
            base_mlp (LlamaMLP): The base MLP.
            mosle_config (MoSLEConfig): The mosle configuration.
            rank (int): The rank.
        """

        super().__init__()

        # The MoSLE configuration.
        self.mosle_config = mosle_config
        self.ex_num = self.mosle_config.ex_num
        self.jitter_noise = self.mosle_config.jitter_noise
        self.selected_ex_num = self.mosle_config.selected_ex_num

        # Create the necessary components for MoSLE.
        self.router = nn.Linear(
            in_features=base_mlp.hidden_size,
            out_features=self.mosle_config.ex_num,
        )

        base_expert = MoSLELlamaMLPV2(
            base_mlp=base_mlp,
            mosle_config=self.mosle_config,
            rank=rank,
        )

        base_expert_state_dict = base_expert.state_dict()

        self.experts = nn.ModuleList(
            modules=[
                MoSLELlamaMLPV2(
                    base_mlp=base_mlp,
                    mosle_config=self.mosle_config,
                    rank=rank,
                    create_empty_proj=True,
                ) for _ in range(self.mosle_config.ex_num)
            ]
        )

        for expert in self.experts:
            expert.load_state_dict(state_dict=base_expert_state_dict)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, sequence_length, hidden_size = x.shape

        # Add jitter noise.
        if self.training and self.jitter_noise > 0:
            x *= torch.empty_like(input=x).uniform_(
                from_=(1.0 - self.jitter_noise),
                to=(1.0 + self.jitter_noise),
            )

        # New shape of x: (batch_size * sequence_length, hidden_size)
        x = x.view(-1, hidden_size)

        # Shape of routing_logits: (batch_size * sequence_length, ex_num)
        routing_logits = self.router(input=x)

        # Do softmax on ex_num dimension.
        routing_weights = F.softmax(
            input=routing_logits,
            dim=1,
            dtype=torch.float,
        )

        # Select the top-k experts.
        # Shape of new routing_weights: (batch_size * sequence_length, selected_ex_num)
        # Shape of selected_experts: (batch_size * sequence_length, selected_ex_num)
        routing_weights, selected_experts = torch.topk(
            input=routing_weights,
            k=self.selected_ex_num,
            dim=-1,
        )

        # Do softmax on selected_ex_num dimension.
        routing_weights /= routing_weights.sum(
            dim=-1,
            keepdim=True,
        )

        routing_weights.to(dtype=x.dtype)

        # Do not know why the parameter name is `input` in `F.one_hot` not `tensor`.
        # Shape of experts_mask: (batch_size * sequence_length, selected_ex_num, ex_num)
        experts_mask = F.one_hot(
            input=selected_experts,
            num_classes=self.ex_num,
        )

        # Shape of new experts_mask: (ex_num, selected_ex_num, batch_size * sequence_length)
        # The meaning of new experts_mask (i, j, k) is that the i-th expert is the k-th token's j-th selected expert.
        experts_mask = experts_mask.permute(2, 1, 0)

        # Shape of y: (batch_size * sequence_length, hidden_size)
        y = torch.zeros(
            size=x.shape,
            dtype=x.dtype,
            device=x.device,
        )

        # There is an example for the following code.
        # Suppose:
        # - batch_size * sequence_length is 5.
        # - selected_ex_num is 2.
        # And:
        # - experts_mask[100] is: tensor([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
        # - top_ex_idx and token_idx will be tensor([0, 1]) and tensor([0, 4]).
        # That mean the 0-th token's 0-th selected expert and the 4-th token's 1-th selected expert are the 100-th expert.
        for ex_idx in range(self.ex_num):
            top_ex_idx, token_idx = torch.where(experts_mask[ex_idx])

            # current_x = x[None, token_idx].reshape(-1, hidden_size)
            current_x = x[token_idx]

            output = self.experts[ex_idx](x=current_x)

            # Multiply the routing weights.
            output *= routing_weights[token_idx, top_ex_idx, None]

            y.index_add_(
                dim=0,
                index=token_idx,
                source=output.to(dtype=x.dtype),
            )

            # Normalize the output with selected_ex_num.
            y /= self.selected_ex_num

        # Reshape y to the original shape.
        y = y.reshape(shape=(batch_size, sequence_length, hidden_size))

        return (
            y,
            routing_logits,
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MoSLELlamaDecoderLayer(nn.Module):
    """ The MoSLELlamaDecoderLayer class. (yxhong-tw added)
    """

    def __init__(
        self,
        base_decoder_layer: LlamaDecoderLayer,
        mosle_config: MoSLEConfig,
    ) -> None:
        """ The constructor method for MoSLELlamaDecoderLayer class.

        Args:
            base_decoder_layer (LlamaDecoderLayer): The base decoder layer.
            mosle_config (MoSLEConfig): The mosle configuration.
        """

        super().__init__()

        self.hidden_size = base_decoder_layer.hidden_size

        self.self_attn = base_decoder_layer.self_attn

        self.mlp = MoSLELlamaMLP(
            base_mlp=base_decoder_layer.mlp,
            mosle_config=mosle_config,
        )

        self.input_layernorm = base_decoder_layer.input_layernorm
        self.post_attention_layernorm = \
            base_decoder_layer.post_attention_layernorm

        self.ex_num = mosle_config.ex_num
        self.similarity_calculator = SimilarityCalculator()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings will become mandatory in v4.46.
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor,
        Optional[
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
        ],
    ]:
        """ The forward method for MoSLELlamaDecoderLayer class.

        Args:
            hidden_states (torch.Tensor): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (Optional[torch.Tensor], optional): Attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1, query_sequence_length, key_sequence_length)` if default attention is used. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): The position of the input tokens in the sequence. Defaults to None.
            past_key_value (Optional[Cache], optional): Cached past key and value projection states. Defaults to None.
            output_attentions (Optional[bool], optional): Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. Defaults to False.
            output_router_logits (Optional[bool], optional): Whether or not to return the router logits. Defaults to False.
            use_cache (Optional[bool], optional): If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`). Defaults to False.
            cache_position (Optional[torch.LongTensor], optional): Indices depicting the position of the input sequence tokens in the sequence. Defaults to None.
            position_embeddings (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`, with `head_dim` being the embedding dimension of each attention head. Defaults to None.
            kwargs (Optional[Dict], optional): Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code into the model. Defaults to None.

        Returns:
            Tuple[ torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]], ]: The outputs.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Compute the self-attention.
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, routing_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (routing_logits,)

        return outputs

    def calculate_exs_similarity_in_layer(
        self,
        similarity_type: str,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        List[Dict[str, Union[int, torch.Tensor]]],
    ]:
        """ Calculate the similarities between the experts.

        Args:
            similarity_type (str): The similarity type.

        Returns:
            Tuple[ torch.Tensor, List[Dict[str, Union[int, torch.Tensor]]], ]: The mean similarity in layer and the similarities.
        """

        # TODO: Check why a, b's dim should be different for cosine similarity.
        if (similarity_type == 'cosine') and ('dim' not in kwargs):
            a_kwargs = kwargs.copy()
            a_kwargs['dim'] = 0
            b_kwargs = kwargs
        else:
            a_kwargs = kwargs
            b_kwargs = kwargs

        similarities = []
        for i in range(self.ex_num):
            for j in range(i + 1, self.ex_num):
                if i == j:
                    continue

                similarities.append({
                    'i': i,
                    'j': j,
                    'gate_similarity_a': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.gate_proj_exs[i]['a'](),
                        x2=self.mlp.gate_proj_exs[j]['a'](),
                        **a_kwargs,
                    ),
                    'gate_similarity_b': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.gate_proj_exs[i]['b'](),
                        x2=self.mlp.gate_proj_exs[j]['b'](),
                        **b_kwargs,
                    ),
                    'up_similarity_a': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.up_proj_exs[i]['a'](),
                        x2=self.mlp.up_proj_exs[j]['a'](),
                        **a_kwargs,
                    ),
                    'up_similarity_b': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.up_proj_exs[i]['b'](),
                        x2=self.mlp.up_proj_exs[j]['b'](),
                        **b_kwargs,
                    ),
                    'down_similarity_a': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.down_proj_exs[i]['a'](),
                        x2=self.mlp.down_proj_exs[j]['a'](),
                        **a_kwargs,
                    ),
                    'down_similarity_b': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.down_proj_exs[i]['b'](),
                        x2=self.mlp.down_proj_exs[j]['b'](),
                        **b_kwargs,
                    ),
                })

        layer_mean_similarity = 0
        for similarity in similarities:
            layer_mean_similarity += (
                similarity['gate_similarity_a'].mean() + \
                similarity['gate_similarity_b'].mean() + \
                similarity['up_similarity_a'].mean() + \
                similarity['up_similarity_b'].mean() + \
                similarity['down_similarity_a'].mean() + \
                similarity['down_similarity_b'].mean()
            ) / 6

        layer_mean_similarity /= len(similarities)

        return (
            layer_mean_similarity,
            similarities,
        )


class MoSLELlamaDecoderLayerV2(nn.Module):
    """ The MoSLELlamaDecoderLayerV2 class. (yxhong-tw added)
    """

    def __init__(
        self,
        base_decoder_layer: LlamaDecoderLayer,
        mosle_config: MoSLEConfig,
        layer_idx: int,
    ) -> None:
        """ The constructor method for MoSLELlamaDecoderLayerV2 class.

        Args:
            base_decoder_layer (LlamaDecoderLayer): The base decoder layer.
            mosle_config (MoSLEConfig): The mosle configuration.
            layer_idx (int): The layer index.
        """

        super().__init__()

        mlp = base_decoder_layer.mlp

        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size
        self.self_attn = base_decoder_layer.self_attn

        rank = self._calculate_rank(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            ex_params_ratio=mosle_config.ex_params_ratio,
        )
        if layer_idx == 0:
            logger.info(msg=f'The rank is {rank}.')

        self.moe = MoSLEMoE(
            base_mlp=mlp,
            mosle_config=mosle_config,
            rank=rank,
        )

        self.input_layernorm = base_decoder_layer.input_layernorm
        self.post_attention_layernorm = \
            base_decoder_layer.post_attention_layernorm

        self.similarity_calculator = SimilarityCalculator()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings will become mandatory in v4.46.
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor,
        Optional[
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
        ],
    ]:
        """ The forward method for MoSLELlamaDecoderLayer class.

        Args:
            hidden_states (torch.Tensor): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (Optional[torch.Tensor], optional): Attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1, query_sequence_length, key_sequence_length)` if default attention is used. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): The position of the input tokens in the sequence. Defaults to None.
            past_key_value (Optional[Cache], optional): Cached past key and value projection states. Defaults to None.
            output_attentions (Optional[bool], optional): Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. Defaults to False.
            output_router_logits (Optional[bool], optional): Whether or not to return the router logits. Defaults to False.
            use_cache (Optional[bool], optional): If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`). Defaults to False.
            cache_position (Optional[torch.LongTensor], optional): Indices depicting the position of the input sequence tokens in the sequence. Defaults to None.
            position_embeddings (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`, with `head_dim` being the embedding dimension of each attention head. Defaults to None.
            kwargs (Optional[Dict], optional): Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code into the model. Defaults to None.

        Returns:
            Tuple[ torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]], ]: The outputs.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Compute the self-attention.
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, routing_logits = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (routing_logits,)

        return outputs

    def calculate_exs_similarity_in_layer(
        self,
        similarity_type: str,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        List[Dict[str, Union[int, torch.Tensor]]],
    ]:
        """ Calculate the similarities between the experts.

        Args:
            similarity_type (str): The similarity type.

        Returns:
            Tuple[ torch.Tensor, List[Dict[str, Union[int, torch.Tensor]]], ]: The mean similarity in layer and the similarities.
        """

        # TODO: Check why a, b's dim should be different for cosine similarity.
        if (similarity_type == 'cosine') and ('dim' not in kwargs):
            a_kwargs = kwargs.copy()
            a_kwargs['dim'] = 0
            b_kwargs = kwargs
        else:
            a_kwargs = kwargs
            b_kwargs = kwargs

        similarities = []
        for i in range(self.ex_num):
            for j in range(i + 1, self.ex_num):
                if i == j:
                    continue

                similarities.append({
                    'i': i,
                    'j': j,
                    'gate_similarity_a': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.gate_proj_exs[i]['a'](),
                        x2=self.mlp.gate_proj_exs[j]['a'](),
                        **a_kwargs,
                    ),
                    'gate_similarity_b': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.gate_proj_exs[i]['b'](),
                        x2=self.mlp.gate_proj_exs[j]['b'](),
                        **b_kwargs,
                    ),
                    'up_similarity_a': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.up_proj_exs[i]['a'](),
                        x2=self.mlp.up_proj_exs[j]['a'](),
                        **a_kwargs,
                    ),
                    'up_similarity_b': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.up_proj_exs[i]['b'](),
                        x2=self.mlp.up_proj_exs[j]['b'](),
                        **b_kwargs,
                    ),
                    'down_similarity_a': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.down_proj_exs[i]['a'](),
                        x2=self.mlp.down_proj_exs[j]['a'](),
                        **a_kwargs,
                    ),
                    'down_similarity_b': self.similarity_calculator(
                        similarity_type=similarity_type,
                        x1=self.mlp.down_proj_exs[i]['b'](),
                        x2=self.mlp.down_proj_exs[j]['b'](),
                        **b_kwargs,
                    ),
                })

        layer_mean_similarity = 0
        for similarity in similarities:
            layer_mean_similarity += (
                similarity['gate_similarity_a'].mean() + \
                similarity['gate_similarity_b'].mean() + \
                similarity['up_similarity_a'].mean() + \
                similarity['up_similarity_b'].mean() + \
                similarity['down_similarity_a'].mean() + \
                similarity['down_similarity_b'].mean()
            ) / 6

        layer_mean_similarity /= len(similarities)

        return (
            layer_mean_similarity,
            similarities,
        )

    @classmethod
    def _calculate_rank(
        cls: Type[Self],
        hidden_size: int,
        intermediate_size: int,
        ex_params_ratio: float,
    ) -> int:
        base_model_params = hidden_size * intermediate_size
        ex_params = base_model_params * ex_params_ratio

        rank = ex_params / (hidden_size + intermediate_size)

        return int(rank)


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        if getattr(config, "pretraining_tp", 1) != 1:
            logger.warn("`pretraining_tp` is deprecated, please use `model.tensor_parallel` instead.")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class MoSLELlamaModel(LlamaPreTrainedModel):
    """ The MoSLELlamaModel class. (yxhong-tw added)
    """

    def __init__(
        self,
        base_model: LlamaModel,
        config: LlamaConfig,
        mosle_config: MoSLEConfig,
    ):
        """ The constructor method for the MoSLELlamaModel class.

        Args:
            base_model (LlamaModel): The base model.
            config (LlamaConfig): The llama configuration.
            mosle_config (MoSLEConfig): The mosle configuration.
        """

        super().__init__(config=config)

        self.padding_idx = base_model.padding_idx
        self.vocab_size = base_model.vocab_size
        self.embed_tokens = base_model.embed_tokens

        self.layers = nn.ModuleList(
            [
                MoSLELlamaDecoderLayer(
                    base_decoder_layer=layer,
                    mosle_config=mosle_config,
                )
                for layer in base_model.layers
            ]
        )

        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb

        self.gradient_checkpointing = base_model.gradient_checkpointing
        if getattr(config, 'pretraining_tp', 1) != 1:
            logger.warn(
                msg='`pretraining_tp` is deprecated, please use `model.tensor_parallel` instead.'
            )

        self.mosle_config = mosle_config

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None \
            else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None \
                else self.config.output_hidden_states
        )

        output_router_logits = (
            output_router_logits if output_router_logits is not None \
                else self.mosle_config.output_router_logits
        )

        use_cache = use_cache if use_cache is not None \
            else self.config.use_cache

        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values
                )
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() \
                if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = \
                    layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ] if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    @staticmethod
    def calculate_load_balance_loss(
        routing_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
        ex_num: Optional[int] = None,
        selected_ex_num: int = 2,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, int]:
        if routing_logits is None:
            return 0

        concatenated_routing_logits = None
        if isinstance(routing_logits, tuple):
            compute_device = routing_logits[0].device
            concatenated_routing_logits = torch.cat(
                tensors=[
                    layer_routing_logits.to(device=compute_device) \
                        for layer_routing_logits in routing_logits
                ],
                dim=0,
            )

        routing_weights = F.softmax(
            input=concatenated_routing_logits \
                if concatenated_routing_logits is not None else routing_logits,
            dim=-1,
        )

        _, selected_experts = torch.topk(
            input=routing_weights,
            k=selected_ex_num,
            dim=-1,
        )

        # Do not know why the parameter name is `input` in `F.one_hot` not `tensor`.
        expert_mask = F.one_hot(
            input=selected_experts,
            num_classes=ex_num,
        )

        if attention_mask is None:
            # Compute the percentage of tokens routed to each experts.
            tokens_per_expert = torch.mean(
                input=expert_mask.float(),
                dim=0,
            )

            # Compute the average probability of routing to these experts.
            router_prob_per_expert = torch.mean(
                input=routing_weights,
                dim=0,
            )
        else:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_routing_logits.shape[0] // \
                (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask.
            expert_attention_mask = (
                attention_mask[None, :, :, None, None].expand(
                    size=(
                        num_hidden_layers,
                        batch_size,
                        sequence_length,
                        selected_ex_num, ex_num,
                    )
                ).reshape(
                    -1,
                    selected_ex_num,
                    ex_num,
                ).to(device=compute_device)
            )

            # Compute the percentage of tokens routed to each experts.
            tokens_per_expert = (
                torch.sum(
                    input=expert_mask.float() * expert_attention_mask,
                    dim=0,
                ) / torch.sum(
                    input=expert_attention_mask,
                    dim=0,
                )
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert.
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None].expand(
                    size=(
                        num_hidden_layers,
                        batch_size,
                        sequence_length,
                        ex_num,
                    )
                ).reshape(
                    -1,
                    ex_num,
                ).to(device=compute_device)
            )

            # Compute the average probability of routing to these experts.
            router_prob_per_expert = (
                torch.sum(
                    input=routing_weights * router_per_expert_attention_mask,
                    dim=0,
                ) / torch.sum(
                    input=router_per_expert_attention_mask,
                    dim=0,
                )
            )

        overall_loss = torch.sum(
            input=tokens_per_expert * router_prob_per_expert.unsqueeze(0)
        )

        return overall_loss * ex_num

    def calculate_exs_similarity_in_model(
        self,
        similarity_type: str = 'cosine',
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[int, int, int], Dict[str, Union[int, torch.Tensor]]],
        List[torch.Tensor],
        torch.Tensor,
    ]:
        """ Calculate layers mean similarity.

        Args:
            similarity_type (str, optional): The similarity type. Defaults to 'cosine'.

        Returns:
            Tuple[ Dict[Tuple[int, int, int], Dict[str, Union[int, torch.Tensor]]], List[torch.Tensor], torch.Tensor, ]: The similarities, layer mean similarities, and model mean similarity.
        """

        all_similarities = {}
        layer_mean_similarities = []
        model_mean_similarity = 0

        for layer_index, layer in enumerate(iterable=self.layers):
            layer_mean_similarity, similarities = \
                layer.calculate_exs_similarity_in_layer(
                    similarity_type=similarity_type,
                    **kwargs,
                )

            layer_mean_similarities.append(layer_mean_similarity)

            for similarity in similarities:
                similarity['l'] = layer_index
                all_similarities[(
                    similarity['l'],
                    similarity['i'],
                    similarity['j'],
                )] = similarity

            model_mean_similarity += layer_mean_similarity

        model_mean_similarity /= len(self.layers)

        return (
            all_similarities,
            layer_mean_similarities,
            model_mean_similarity,
        )


class MoSLELlamaModelV2(LlamaPreTrainedModel):
    """ The MoSLELlamaModelV2 class. (yxhong-tw added)
    """

    def __init__(
        self,
        base_model: LlamaModel,
        config: LlamaConfig,
        mosle_config: MoSLEConfig,
    ):
        """ The constructor method for the MoSLELlamaModelV2 class.

        Args:
            base_model (LlamaModel): The base model.
            config (LlamaConfig): The llama configuration.
            mosle_config (MoSLEConfig): The mosle configuration.
        """

        super().__init__(config=config)

        self.padding_idx = base_model.padding_idx
        self.vocab_size = base_model.vocab_size
        self.embed_tokens = base_model.embed_tokens

        self.layers = nn.ModuleList()
        for layer_idx, layer in enumerate(iterable=base_model.layers):
            if layer_idx < 3 or \
                    layer_idx > len(base_model.layers) - 4 or \
                        layer_idx % 2 == 1:
            # if True:
                self.layers.append(layer)
            else:
                print(f'layer {layer_idx} is replaced by MoSLELlamaDecoderLayerV2.')
                self.layers.append(
                    MoSLELlamaDecoderLayerV2(
                        base_decoder_layer=layer,
                        mosle_config=mosle_config,
                        layer_idx=layer_idx,
                    )
                )

        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb

        self.gradient_checkpointing = base_model.gradient_checkpointing
        if getattr(config, 'pretraining_tp', 1) != 1:
            logger.warn(
                msg='`pretraining_tp` is deprecated, please use `model.tensor_parallel` instead.'
            )

        self.mosle_config = mosle_config

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None \
            else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None \
                else self.config.output_hidden_states
        )

        output_router_logits = (
            output_router_logits if output_router_logits is not None \
                else self.mosle_config.output_router_logits
        )

        use_cache = use_cache if use_cache is not None \
            else self.config.use_cache

        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values
                )
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() \
                if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = \
                    layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and isinstance(
                decoder_layer,
                MoSLELlamaDecoderLayerV2,
            ):
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ] if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    @staticmethod
    def calculate_load_balance_loss(
        routing_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
        ex_num: Optional[int] = None,
        selected_ex_num: int = 2,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, int]:
        if routing_logits is None:
            return 0

        concatenated_routing_logits = None
        if isinstance(routing_logits, tuple):
            compute_device = routing_logits[0].device
            concatenated_routing_logits = torch.cat(
                tensors=[
                    layer_routing_logits.to(device=compute_device) \
                        for layer_routing_logits in routing_logits
                ],
                dim=0,
            )

        routing_weights = F.softmax(
            input=concatenated_routing_logits \
                if concatenated_routing_logits is not None else routing_logits,
            dim=-1,
        )

        _, selected_experts = torch.topk(
            input=routing_weights,
            k=selected_ex_num,
            dim=-1,
        )

        # Do not know why the parameter name is `input` in `F.one_hot` not `tensor`.
        expert_mask = F.one_hot(
            input=selected_experts,
            num_classes=ex_num,
        )

        if attention_mask is None:
            # Compute the percentage of tokens routed to each experts.
            tokens_per_expert = torch.mean(
                input=expert_mask.float(),
                dim=0,
            )

            # Compute the average probability of routing to these experts.
            router_prob_per_expert = torch.mean(
                input=routing_weights,
                dim=0,
            )
        else:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_routing_logits.shape[0] // \
                (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask.
            expert_attention_mask = (
                attention_mask[None, :, :, None, None].expand(
                    size=(
                        num_hidden_layers,
                        batch_size,
                        sequence_length,
                        selected_ex_num, ex_num,
                    )
                ).reshape(
                    -1,
                    selected_ex_num,
                    ex_num,
                ).to(device=compute_device)
            )

            # Compute the percentage of tokens routed to each experts.
            tokens_per_expert = (
                torch.sum(
                    input=expert_mask.float() * expert_attention_mask,
                    dim=0,
                ) / torch.sum(
                    input=expert_attention_mask,
                    dim=0,
                )
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert.
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None].expand(
                    size=(
                        num_hidden_layers,
                        batch_size,
                        sequence_length,
                        ex_num,
                    )
                ).reshape(
                    -1,
                    ex_num,
                ).to(device=compute_device)
            )

            # Compute the average probability of routing to these experts.
            router_prob_per_expert = (
                torch.sum(
                    input=routing_weights * router_per_expert_attention_mask,
                    dim=0,
                ) / torch.sum(
                    input=router_per_expert_attention_mask,
                    dim=0,
                )
            )

        overall_loss = torch.sum(
            input=tokens_per_expert * router_prob_per_expert.unsqueeze(0)
        )

        return overall_loss * ex_num

    def calculate_exs_similarity_in_model(
        self,
        similarity_type: str = 'cosine',
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[int, int, int], Dict[str, Union[int, torch.Tensor]]],
        List[torch.Tensor],
        torch.Tensor,
    ]:
        """ Calculate layers mean similarity.

        Args:
            similarity_type (str, optional): The similarity type. Defaults to 'cosine'.

        Returns:
            Tuple[ Dict[Tuple[int, int, int], Dict[str, Union[int, torch.Tensor]]], List[torch.Tensor], torch.Tensor, ]: The similarities, layer mean similarities, and model mean similarity.
        """

        all_similarities = {}
        layer_mean_similarities = []
        model_mean_similarity = 0

        for layer_index, layer in enumerate(iterable=self.layers):
            layer_mean_similarity, similarities = \
                layer.calculate_exs_similarity_in_layer(
                    similarity_type=similarity_type,
                    **kwargs,
                )

            layer_mean_similarities.append(layer_mean_similarity)

            for similarity in similarities:
                similarity['l'] = layer_index
                all_similarities[(
                    similarity['l'],
                    similarity['i'],
                    similarity['j'],
                )] = similarity

            model_mean_similarity += layer_mean_similarity

        model_mean_similarity /= len(self.layers)

        return (
            all_similarities,
            layer_mean_similarities,
            model_mean_similarity,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MoSLELlamaForCausalLM(
    LlamaPreTrainedModel,
    GenerationMixin,
):
    """ The MoSLELlamaForCausalLM class.
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(
        self,
        base_model: LlamaForCausalLM,
        config: LlamaConfig,
        mosle_config: MoSLEConfig,
    ):
        """ The constructor method for MoSLELlamaForCausalLM class.

        Args:
            base_model (LlamaForCausalLM): The base model.
            config (LlamaConfig): The llama configuration.
            mosle_config (MoSLEConfig): The mosle configuration.
        """

        super().__init__(config=config)

        self.model = MoSLELlamaModel(
            base_model=base_model.model,
            config=config,
            mosle_config=mosle_config,
        )
        self.vocab_size = base_model.vocab_size
        self.lm_head = base_model.lm_head

        self.mosle_config = mosle_config

        self.init_similarity_aux_loss = None
        self.freeze_similarity_aux_loss = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_similarities: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, MoSLECausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None \
            else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        output_router_logits = (
            output_router_logits if output_router_logits is not None \
                else self.mosle_config.output_router_logits
        )

        output_similarities = (
            output_similarities if output_similarities is not None \
                else self.mosle_config.output_similarities
        )

        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        routing_logits = None
        router_aux_loss = None
        if output_router_logits:
            routing_logits = outputs.router_logits

            if labels is not None:
                router_aux_loss = self.model.calculate_load_balance_loss(
                    routing_logits=routing_logits,
                    ex_num=self.mosle_config.ex_num,
                    selected_ex_num=self.mosle_config.selected_ex_num,
                    attention_mask=attention_mask,
                )

                loss += self.mosle_config.router_aux_loss_coef * \
                    router_aux_loss.to(device=loss.device)

        similarity_aux_loss = None
        if output_similarities:
            _, _, similarity_aux_loss = \
                self.model.calculate_exs_similarity_in_model(
                    similarity_type=self.mosle_config.similarity_type,
                )

            if self.init_similarity_aux_loss is None and \
                    similarity_aux_loss is not None:
                self.init_similarity_aux_loss = similarity_aux_loss

            if not self.freeze_similarity_aux_loss:
                bound = (self.init_similarity_aux_loss * \
                    self.mosle_config.similarity_lower_bound_ratio)

                if similarity_aux_loss < bound:
                    self.freeze_similarity_aux_loss = True
            else:
                bound = (self.init_similarity_aux_loss * \
                    self.mosle_config.similarity_upper_bound_ratio)

                if similarity_aux_loss > bound:
                    self.freeze_similarity_aux_loss = False

            if labels is not None and \
                    similarity_aux_loss is not None and \
                        not self.freeze_similarity_aux_loss:
                loss += self.mosle_config.similarity_aux_loss_coef * \
                    similarity_aux_loss.to(device=loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]

            if output_router_logits:
                output = (router_aux_loss,) + output

            if output_similarities:
                output = (similarity_aux_loss,) + output

            return (loss,) + output if loss is not None else output

        return MoSLECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            router_aux_loss=router_aux_loss,
            similarity_aux_loss=similarity_aux_loss,
            router_logits=routing_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MoSLELlamaForCausalLMV2(
    LlamaPreTrainedModel,
    GenerationMixin,
):
    """ The MoSLELlamaForCausalLM class.
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(
        self,
        base_model: LlamaForCausalLM,
        config: LlamaConfig,
        mosle_config: MoSLEConfig,
    ):
        """ The constructor method for MoSLELlamaForCausalLM class.

        Args:
            base_model (LlamaForCausalLM): The base model.
            config (LlamaConfig): The llama configuration.
            mosle_config (MoSLEConfig): The mosle configuration.
        """

        super().__init__(config=config)

        self.model = MoSLELlamaModelV2(
            base_model=base_model.model,
            config=config,
            mosle_config=mosle_config,
        )
        self.vocab_size = base_model.vocab_size
        self.lm_head = base_model.lm_head

        self.mosle_config = mosle_config

        self.init_similarity_aux_loss = None
        self.freeze_similarity_aux_loss = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_similarities: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, MoSLECausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None \
            else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        output_router_logits = (
            output_router_logits if output_router_logits is not None \
                else self.mosle_config.output_router_logits
        )

        output_similarities = (
            output_similarities if output_similarities is not None \
                else self.mosle_config.output_similarities
        )

        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        routing_logits = None
        router_aux_loss = None
        if output_router_logits:
            routing_logits = outputs.router_logits

            if labels is not None:
                router_aux_loss = self.model.calculate_load_balance_loss(
                    routing_logits=routing_logits,
                    ex_num=self.mosle_config.ex_num,
                    selected_ex_num=self.mosle_config.selected_ex_num,
                    attention_mask=attention_mask,
                )

                loss += self.mosle_config.router_aux_loss_coef * \
                    router_aux_loss.to(device=loss.device)

        similarity_aux_loss = None
        if output_similarities:
            _, _, similarity_aux_loss = \
                self.model.calculate_exs_similarity_in_model(
                    similarity_type=self.mosle_config.similarity_type,
                )

            if self.init_similarity_aux_loss is None and \
                    similarity_aux_loss is not None:
                self.init_similarity_aux_loss = similarity_aux_loss

            if not self.freeze_similarity_aux_loss:
                bound = (self.init_similarity_aux_loss * \
                    self.mosle_config.similarity_lower_bound_ratio)

                if similarity_aux_loss < bound:
                    self.freeze_similarity_aux_loss = True
            else:
                bound = (self.init_similarity_aux_loss * \
                    self.mosle_config.similarity_upper_bound_ratio)

                if similarity_aux_loss > bound:
                    self.freeze_similarity_aux_loss = False

            if labels is not None and \
                    similarity_aux_loss is not None and \
                        not self.freeze_similarity_aux_loss:
                loss += self.mosle_config.similarity_aux_loss_coef * \
                    similarity_aux_loss.to(device=loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]

            if output_router_logits:
                output = (router_aux_loss,) + output

            if output_similarities:
                output = (similarity_aux_loss,) + output

            return (loss,) + output if loss is not None else output

        return MoSLECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            router_aux_loss=router_aux_loss,
            similarity_aux_loss=similarity_aux_loss,
            router_logits=routing_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
