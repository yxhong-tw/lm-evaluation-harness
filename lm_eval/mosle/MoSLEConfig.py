from dataclasses import (
    dataclass,
    field,
)
from transformers import PretrainedConfig
from typing import Dict


@dataclass
class MoSLEConfig(PretrainedConfig):
    """ The configuration class for the MoSLE model.
    """

    output_router_logits: bool = False
    output_similarities: bool = False

    jitter_noise: float = 0.0

    router_aux_loss_coef: float = 0.125

    similarity_aux_loss_coef: float = 0.125
    similarity_lower_bound_ratio: float = 0.7
    similarity_upper_bound_ratio: float = 0.8
    similarity_type: str = 'cosine'

    # 'ex' means 'expert'.
    ex_num: int = 2
    ex_params_ratio: float = 0.5
    selected_ex_num: int = 1

    # Set the default values for the special tokens.
    special_tokens: Dict[str, Dict[str, str]] = \
        field(default_factory=lambda: {
            # The key and value of 'seq' can be 'sep_token' and '<|sequence|>', respectively (if needed).
            'sep': {
                'key': None,
                'value': None,
            },
        })
