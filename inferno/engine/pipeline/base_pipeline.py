import torch
from torch import nn
from abc import ABC, abstractmethod

class BasePipeline(nn.Module, ABC):
    def __init__(self, config_path: str):
        super().__init__()
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_path)
        self.logger = self.setup_logger()
        self.progress_fn = lambda progress, desc: None
        self.weight_dtype = self.get_weight_dtype(self.config.model.mixed_precision)

    def load_config(self, config_path: str):
        import pyrallis
        if not config_path or not isinstance(config_path, str):
            raise ValueError("Config path must be a valid string.")
        try:
            return pyrallis.load(open(config_path))
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")

    def setup_logger(self):
        import logging
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        return logger

    def get_weight_dtype(self, mixed_precision: str):
        precision_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        weight_dtype = precision_map.get(mixed_precision)
        if weight_dtype is None:
            raise ValueError(f"Invalid mixed precision: {mixed_precision}")
        return weight_dtype

    @abstractmethod
    def build_models(self):
        pass

    @abstractmethod
    def preprocess_inputs(self, inputs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
