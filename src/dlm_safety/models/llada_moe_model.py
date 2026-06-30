"""
LLaDA-MoE (Large Language Diffusion with Mixture-of-Experts) wrapper.

Generation path matches the official model card snippet at
https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct.

Key differences from base LLaDA-Instruct (GSAI-ML/LLaDA-8B-Instruct):
- Mask token ID is 156895 (not 126336); vocab_size is 157184.
- Architecture is LLaDAMoEModelLM with 64 experts × 8 active per token.
- Forward returns ``model(x).logits`` directly (the official snippet relies on
  the standard transformers logits attribute), so no fallback parsing needed.

Otherwise the block-wise low-confidence remasking loop with Gumbel-noise
argmax is structurally identical to LLaDA, so we subclass ``LLaDAModel`` and
only override the mask-id resolution and the forward unwrap.
"""

from typing import Optional

import torch

from .llada_model import LLaDAModel


class LLaDAMoEModel(LLaDAModel):
    """Wrapper for LLaDA-MoE masked diffusion language model."""

    _DEFAULT_MASK_TOKEN_ID = 156895

    def __init__(
        self,
        model_name_or_path: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        device_str: str = "auto",
        sampling_steps: int = 128,
        block_length: int = 32,
        remasking: str = "low_confidence",
        use_chat_template: bool = True,
        **model_kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            device_str=device_str,
            sampling_steps=sampling_steps,
            block_length=block_length,
            remasking=remasking,
            use_chat_template=use_chat_template,
            **model_kwargs,
        )

        # Override mask id: LLaDA-MoE uses 156895 (vs 126336 in LLaDA-Instruct).
        # The tokenizer does not register <mask>, so the parent fallback to
        # 126336 would be wrong here.
        resolved = self.tokenizer.convert_tokens_to_ids("<mask>")
        if resolved is None or resolved == self.tokenizer.unk_token_id:
            resolved = self._DEFAULT_MASK_TOKEN_ID
        self.mask_id = resolved

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            out = self.model(input_ids, attention_mask=attention_mask)
        else:
            out = self.model(input_ids)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, tuple):
            return out[0]
        raise RuntimeError(f"Unexpected model output type: {type(out)}")
