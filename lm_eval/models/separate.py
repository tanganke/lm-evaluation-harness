import copy
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs, find_executable_batch_size
from accelerate.utils import get_max_memory
from fusion_bench.models.separate_io import separate_load
from huggingface_hub import HfApi
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)
from typing_extensions import override

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)

from .huggingface import HFLM

eval_logger = utils.eval_logger


@register_model("separete", "fusion_bench")
class FusionBenchLM(HFLM):

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    @override
    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        self._model = separate_load(pretrained, dtype=dtype, device=self.device)
        self._model.requires_grad_(False)
        self._model.eval()
        self._model.compile()
        