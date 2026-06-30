"""
LLaDA (Large Language Diffusion with mAsking) model wrapper.

Generation path is aligned to the official LLaDA `generate.py` remasking flow.
See: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

Key requirements from official repo:
- Block-wise generation: response is split into blocks; only tokens in the
  current block are eligible for transfer at each step (semi-autoregressive).
- Steps: optimal when sampling_steps ~ gen_length (Appendix B.4/B.6).
- attention_mask is passed to the model.
- Remasking: 'low_confidence' (default) or 'random'.
"""

from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModel

from .base import LanguageModel


def _patch_llada_compat():
    """Patch transformers compatibility for LLaDA remote code models."""
    adjust_name = "_adjust_tied_keys_with_tied_pointers"
    finalize_name = "_finalize_model_loading"
    mark_tied_name = "mark_tied_weights_as_initialized"

    if hasattr(transformers.PreTrainedModel, mark_tied_name):
        orig_mark_tied = getattr(transformers.PreTrainedModel, mark_tied_name)

        def _safe_mark_tied(self, *args, **kwargs):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return orig_mark_tied(self, *args, **kwargs)

        setattr(transformers.PreTrainedModel, mark_tied_name, _safe_mark_tied)

    if hasattr(transformers.PreTrainedModel, adjust_name):
        orig_adjust = getattr(transformers.PreTrainedModel, adjust_name)

        def _safe_adjust(self, *args, **kwargs):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return orig_adjust(self, *args, **kwargs)

        setattr(transformers.PreTrainedModel, adjust_name, _safe_adjust)

    if hasattr(transformers.PreTrainedModel, finalize_name):
        orig_finalize = getattr(transformers.PreTrainedModel, finalize_name)

        def _patched_finalize(model, load_config, loading_info):
            orig_tie = model.tie_weights

            def _safe_tie(*args, **kwargs):
                try:
                    return orig_tie(*args, **kwargs)
                except TypeError:
                    return orig_tie()

            model.tie_weights = _safe_tie
            try:
                return orig_finalize(model, load_config, loading_info)
            finally:
                model.tie_weights = orig_tie

        setattr(
            transformers.PreTrainedModel,
            finalize_name,
            staticmethod(_patched_finalize),
        )


_patch_llada_compat()


class LLaDAModel(LanguageModel):
    """Wrapper for LLaDA masked diffusion language models."""

    def __init__(
        self,
        model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct",
        device_str: str = "auto",
        sampling_steps: int = 64,
        block_length: int = 32,
        remasking: Literal["low_confidence", "random"] = "low_confidence",
        use_chat_template: bool = True,
        **model_kwargs,
    ):
        self._name = model_name_or_path
        self.sampling_steps = sampling_steps
        self.block_length = block_length
        self.remasking = remasking
        self.use_chat_template = use_chat_template

        if device_str == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device_str

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            mask_id = self.tokenizer.convert_tokens_to_ids("<mask>")
        if mask_id is None or mask_id == self.tokenizer.unk_token_id:
            mask_id = 126336
        self.mask_id = mask_id

        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True, **model_kwargs
        ).to(self._device)
        self.model.eval()

        # Some remote LLaDA configs do not define `use_cache`, but the model
        # forward path still tries to read it. Set it explicitly either way.
        self.model.config.use_cache = False

    @property
    def device(self) -> str:
        return self._device

    @property
    def name(self) -> str:
        return self._name

    def get_responses(
        self,
        prompts: list[str],
        batched: bool = False,
        **kwargs,
    ) -> Tuple[list[list[int]], list[str]]:
        max_new_tokens = kwargs.get("max_new_tokens", 64)
        temperature = kwargs.get("temperature", 0.0)
        if batched and len(prompts) > 1:
            return self._generate_batched(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        all_tokens, all_texts = [], []
        for prompt in prompts:
            tokens, text = self._generate_single(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            all_tokens.append(tokens)
            all_texts.append(text)
        return all_tokens, all_texts

    def compute_log_probabilities(
        self,
        prompts: list[str],
        responses: list[list[int]],
        batched: bool = False,
    ) -> list[float]:
        del batched
        log_probs = []
        for prompt, resp_tokens in zip(prompts, responses):
            log_probs.append(self._compute_elbo_single(prompt, resp_tokens))
        return log_probs

    def _format_prompt(self, prompt: str) -> str:
        if (
            self.use_chat_template
            and hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template
        ):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return prompt

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            out = self.model(input_ids, attention_mask=attention_mask)
        else:
            out = self.model(input_ids)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, tuple):
            return out[0]
        raise RuntimeError(f"Unexpected model output type: {type(out)}")

    def _add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Official LLaDA categorical sampling helper."""
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def _get_num_transfer_tokens(
        self,
        mask_index: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Precompute number of tokens to transition each denoising step."""
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = (
            torch.zeros(
                mask_num.size(0),
                steps,
                device=mask_index.device,
                dtype=torch.int64,
            )
            + base
        )
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1
        return num_transfer_tokens

    @torch.no_grad()
    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Tuple[list[int], str]:
        formatted = self._format_prompt(prompt)
        prompt_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        n_prompt = len(prompt_ids)

        # Official LLaDA: gen_length % block_length == 0
        block_len = min(self.block_length, max_new_tokens)
        num_blocks = max(1, (max_new_tokens + block_len - 1) // block_len)
        gen_length = num_blocks * block_len

        input_ids = torch.tensor(
            [prompt_ids + [self.mask_id] * gen_length],
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.ones(
            (1, n_prompt + gen_length),
            dtype=torch.long,
            device=self._device,
        )
        prompt_index = input_ids != self.mask_id

        steps_per_block = max(1, self.sampling_steps // num_blocks)

        for num_block in range(num_blocks):
            start = n_prompt + num_block * block_len
            end = n_prompt + (num_block + 1) * block_len
            block_mask_index = input_ids[:, start:end] == self.mask_id
            num_transfer_tokens = self._get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for step in range(steps_per_block):
                mask_index = input_ids == self.mask_id
                if not mask_index.any():
                    break

                logits = self._forward(input_ids, attention_mask=attention_mask)
                logits_with_noise = self._add_gumbel_noise(
                    logits, temperature=temperature
                )
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if self.remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(
                        p, dim=-1, index=x0.unsqueeze(-1)
                    ).squeeze(-1)
                else:  # random
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]),
                        device=x0.device,
                        dtype=logits.dtype,
                    )

                # Only tokens in current or past blocks can be transferred
                # (official: x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -inf)
                x0_p = x0_p.clone()
                x0_p[:, end:] = -np.inf

                x0 = torch.where(mask_index, x0, input_ids)
                confidence = torch.where(
                    mask_index,
                    x0_p,
                    torch.full_like(x0_p, -np.inf),
                )
                confidence = torch.where(
                    prompt_index,
                    torch.full_like(confidence, -np.inf),
                    confidence,
                )

                k = int(num_transfer_tokens[0, step].item())
                if k > 0:
                    transfer_index = torch.zeros_like(
                        input_ids, dtype=torch.bool, device=self._device
                    )
                    _, select_index = torch.topk(confidence[0], k=k)
                    transfer_index[0, select_index] = True
                    input_ids[transfer_index] = x0[transfer_index]

        response_ids = input_ids[0, n_prompt : n_prompt + max_new_tokens].tolist()
        response_text = self.tokenizer.decode(
            response_ids, skip_special_tokens=True
        )
        return response_ids, response_text

    @torch.no_grad()
    def _generate_batched(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Tuple[list[list[int]], list[str]]:
        """Batched counterpart to _generate_single.

        Left-pads prompts so every row's response block sits at positions
        [max_prompt_len, max_prompt_len + gen_length). Because the initial
        block mask is identical across rows (full block masked), the
        per-step transfer count `k` is uniform, so we can do a single
        batched top-k over confidence.
        """
        B = len(prompts)
        pad_id = self.tokenizer.pad_token_id
        formatted = [self._format_prompt(p) for p in prompts]
        prompt_token_lists = [
            self.tokenizer.encode(f, add_special_tokens=False) for f in formatted
        ]
        max_prompt_len = max(len(t) for t in prompt_token_lists)

        block_len = min(self.block_length, max_new_tokens)
        num_blocks = max(1, (max_new_tokens + block_len - 1) // block_len)
        gen_length = num_blocks * block_len
        L = max_prompt_len + gen_length

        input_ids = torch.full(
            (B, L), pad_id, dtype=torch.long, device=self._device
        )
        attention_mask = torch.zeros(
            (B, L), dtype=torch.long, device=self._device
        )
        for b, toks in enumerate(prompt_token_lists):
            offset = max_prompt_len - len(toks)
            input_ids[b, offset:max_prompt_len] = torch.tensor(
                toks, dtype=torch.long, device=self._device
            )
            attention_mask[b, offset:] = 1
        input_ids[:, max_prompt_len:] = self.mask_id

        # Prompt + pad positions: non-mask, protected from transfer.
        prompt_index = input_ids != self.mask_id

        steps_per_block = max(1, self.sampling_steps // num_blocks)

        for num_block in range(num_blocks):
            start = max_prompt_len + num_block * block_len
            end = max_prompt_len + (num_block + 1) * block_len
            block_mask_index = input_ids[:, start:end] == self.mask_id
            # Uniform across rows at block start, so num_transfer_tokens[:, step]
            # is identical per row → scalar k.
            num_transfer_tokens = self._get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for step in range(steps_per_block):
                mask_index = input_ids == self.mask_id
                if not mask_index.any():
                    break

                logits = self._forward(
                    input_ids, attention_mask=attention_mask
                )
                logits_with_noise = self._add_gumbel_noise(
                    logits, temperature=temperature
                )
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if self.remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(
                        p, dim=-1, index=x0.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]),
                        device=x0.device,
                        dtype=logits.dtype,
                    )

                x0_p = x0_p.clone()
                x0_p[:, end:] = -np.inf

                x0 = torch.where(mask_index, x0, input_ids)
                confidence = torch.where(
                    mask_index, x0_p, torch.full_like(x0_p, -np.inf)
                )
                confidence = torch.where(
                    prompt_index,
                    torch.full_like(confidence, -np.inf),
                    confidence,
                )

                k = int(num_transfer_tokens[0, step].item())
                if k > 0:
                    _, select_index = torch.topk(confidence, k=k, dim=-1)
                    transfer_index = torch.zeros_like(
                        input_ids, dtype=torch.bool
                    )
                    transfer_index.scatter_(1, select_index, True)
                    input_ids = torch.where(transfer_index, x0, input_ids)

        response_block = input_ids[
            :, max_prompt_len : max_prompt_len + max_new_tokens
        ]
        all_tokens = response_block.tolist()
        all_texts = [
            self.tokenizer.decode(toks, skip_special_tokens=True)
            for toks in all_tokens
        ]
        return all_tokens, all_texts

    @torch.no_grad()
    def _compute_elbo_single(
        self, prompt: str, response_tokens: list[int]
    ) -> float:
        formatted = self._format_prompt(prompt)
        prompt_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        full_ids = prompt_ids + response_tokens
        n_prompt = len(prompt_ids)
        n_total = len(full_ids)

        if len(response_tokens) == 0:
            return 0.0

        n_mc = 100
        sampling_eps = 1e-5

        x0 = torch.tensor(full_ids, dtype=torch.long, device=self._device)
        x0_resp = x0[n_prompt:]
        n_resp = x0_resp.shape[0]
        attention_mask = torch.ones(1, n_total, dtype=torch.long, device=self._device)

        total_weighted = 0.0
        for _ in range(n_mc):
            t = (
                torch.rand(1, device=self._device) * (1.0 - sampling_eps)
                + sampling_eps
            ).item()

            to_mask_resp = torch.rand(n_resp, device=self._device) < t
            x_t = x0.clone()
            x_t[n_prompt:] = torch.where(
                to_mask_resp,
                torch.full_like(x0_resp, self.mask_id),
                x0_resp,
            )

            logits = self._forward(x_t.unsqueeze(0), attention_mask=attention_mask)

            resp_logits = logits[0, n_prompt:, :].clone()
            del logits

            resp_logits[:, self.mask_id] = -1e6
            log_probs = torch.log_softmax(resp_logits, dim=-1)
            del resp_logits

            pred_lp = log_probs.gather(1, x0_resp.unsqueeze(1)).squeeze(1)
            del log_probs
            nll = -(pred_lp * to_mask_resp.float()).sum()
            total_weighted += nll.item() / t

        # Return sequence-level ELBO estimate — same O(L) scale as AR log-prob sum
        # so |diff| in the POSIX formula has the same magnitude as AR models.
        return -(total_weighted / n_mc)
