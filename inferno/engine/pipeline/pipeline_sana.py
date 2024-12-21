# Copyright 2024 NVIDIA CORPORATION & AFFILIATES & team Inferno
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from pipelines.base_pipeline import BasePipeline
from utils .model_builder import get_vae, get_tokenizer_and_text_encoder, build_model, find_model

class SanaPipeline(BasePipeline):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.image_size = self.config.model.image_size
        self.latent_size = self.image_size // self.config.vae.vae_downsample_rate
        self.max_sequence_length = self.config.text_encoder.model_max_length
        self.flow_shift = self.config.scheduler.flow_shift
        self.guidance_type = self.get_guidance_type("classifier-free_PAG")
        self.base_ratios = self.get_aspect_ratios(self.image_size)
        self.build_models()

    def get_guidance_type(self, guidance_type_name: str):
        return guidance_type_select(
            guidance_type_name, 
            self.config.pag_scale, 
            self.config.model.attn_type
        )

    def build_models(self):
        # Build VAE
        self.vae = get_vae(self.config.vae.vae_type, self.config.vae.vae_pretrained, self.device).to(self.weight_dtype)

        # Build Tokenizer and Text Encoder
        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(
            name=self.config.text_encoder.name, device=self.device
        )

        # Build Sana Model
        pred_sigma = getattr(self.config.scheduler, "pred_sigma", True)
        learn_sigma = getattr(self.config.scheduler, "learn_sigma", True) and pred_sigma
        model_kwargs = {
            "input_size": self.latent_size,
            "config": self.config,
            "caption_channels": self.text_encoder.config.hidden_size,
            "in_channels": self.config.vae.vae_latent_dim,
            "pred_sigma": pred_sigma,
            "learn_sigma": learn_sigma,
            "use_fp32_attention": self.config.model.get("fp32_attention", False),
        }
        self.model = build_model(self.config.model.model_name, **model_kwargs).to(self.weight_dtype)

        # Log model parameters
        self.logger.info(
            f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

    def from_pretrained(self, model_path: str):
        state_dict = find_model(model_path)
        state_dict = state_dict.get("state_dict", state_dict)
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.weight_dtype)
        self.logger.warning(f"Missing keys: {missing}")
        self.logger.warning(f"Unexpected keys: {unexpected}")

    def get_aspect_ratios(self, image_size: int):
        aspect_ratios = {1024: ASPECT_RATIO_1024_TEST, 512: ASPECT_RATIO_512_TEST}
        return aspect_ratios.get(image_size, default_value)

    def build_models(self):
        self.vae = self.build_vae(self.config.vae)
        self.tokenizer, self.text_encoder = self.build_text_encoder(self.config.text_encoder)
        self.model = self.build_sana_model(self.config).to(self.device)

        with torch.no_grad():
            null_token = self.tokenizer(
                "", max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_token.input_ids, null_token.attention_mask)[0]

    def preprocess_inputs(self, inputs):
        return self.tokenizer(inputs, return_tensors="pt").to(self.device)

    def forward(self, input_ids, attention_mask):
        latents = self.vae.encode(input_ids)
        embeddings = self.text_encoder(input_ids, attention_mask)[0]
        outputs = self.model(latents, embeddings)
        return outputs

