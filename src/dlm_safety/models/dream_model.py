"""
Dream masked diffusion language model wrapper.

Uses the official diffusion_generate() API from Dream-org/Dream-v0-Instruct-7B.
See: https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
     https://github.com/HKUNLP/Dream

Key differences from LLaDA:
- Calls model.diffusion_generate() (official method) instead of a custom Python loop
- No block_length parameter; Dream denoises the full sequence at once
- steps parameter (default 512) replaces LLaDA's sampling_steps
- alg parameter controls remasking strategy: "origin", "maskgit_plus", "topk_margin", "entropy"
- MASK token is <|mask|> (id=151666), different from LLaDA's <mask> (id=126336)
- Uses the same chat template format as Qwen2 (<|im_start|>/<|im_end|>)
"""

import re
import warnings
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .base import LanguageModel


class DreamModel(LanguageModel):
    """Wrapper for Dream masked diffusion language models.

    Official model: Dream-org/Dream-v0-Instruct-7B
    Official repo:  https://github.com/HKUNLP/Dream
    """

    # <|mask|> token ID per Dream's generation_config.json
    _DEFAULT_MASK_TOKEN_ID = 151666

    # Minimum denoising iterations per generated token. Dream denoises the whole
    # answer block at once, so a fixed `steps` that is small relative to
    # `max_new_tokens` leaves answer positions masked -> they decode to empty and
    # surface as bare "." / truncated labels. We floor `steps` at this density to
    # make under-denoising impossible to configure silently. The value matches
    # LLaDA's setting (64 steps over a 32-token block = 2 steps/token), keeping
    # the two diffusion models comparable on denoising budget.
    _STEPS_PER_NEW_TOKEN = 2

    # A decoded response that is empty or only punctuation/underscores is the
    # signature of an under-denoised block (no real label token survived).
    _PUNCT_ONLY = re.compile(r"^[\s.,;:!?_'\"-]*$")

    def __init__(
        self,
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B",
        device_str: str = "auto",
        steps: int = 512,
        alg: Literal["origin", "maskgit_plus", "topk_margin", "entropy"] = "entropy",
        alg_temp: Optional[float] = None,
        use_chat_template: bool = True,
        **model_kwargs,
    ):
        self._name = model_name_or_path
        self.steps = steps
        self.alg = alg
        self.alg_temp = alg_temp
        self.use_chat_template = use_chat_template
        self._warned_steps = False
        self._degenerate_count = 0

        if device_str == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device_str

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad so all sequences in a batch end at the same position,
        # making generated_ids[i, n_prompt:] a valid response slice for every i.
        self.tokenizer.padding_side = "left"

        # Resolve mask token: try tokenizer first, fall back to known ID
        mask_id = self.tokenizer.convert_tokens_to_ids("<|mask|>")
        if mask_id is None or mask_id == self.tokenizer.unk_token_id:
            mask_id = self._DEFAULT_MASK_TOKEN_ID
        self.mask_id = mask_id

        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True, **model_kwargs
        ).to(self._device)
        self.model.eval()
        self.model.config.use_cache = False

    @property
    def device(self) -> str:
        return self._device

    @property
    def name(self) -> str:
        return self._name

    def _effective_steps(self, max_new_tokens: int) -> int:
        """Denoising steps to actually use, floored at the per-token density.

        Guarantees ``steps >= _STEPS_PER_NEW_TOKEN * max_new_tokens`` so the
        answer block is always fully denoisable. Only ever raises the configured
        value (never lowers it), so well-configured runs are unchanged and stay
        comparable; a too-small ``sampling_steps`` is corrected with a warning
        instead of silently producing bare "." outputs.
        """
        floor = self._STEPS_PER_NEW_TOKEN * max_new_tokens
        if self.steps < floor:
            if not self._warned_steps:
                warnings.warn(
                    f"DreamModel: sampling_steps={self.steps} is below the safe "
                    f"floor of {floor} ({self._STEPS_PER_NEW_TOKEN}x "
                    f"max_new_tokens={max_new_tokens}); raising to {floor} to "
                    f"avoid under-denoising. Set sampling_steps>={floor} in the "
                    f"config to silence this.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._warned_steps = True
            return floor
        return self.steps

    def _flag_degenerate(
        self, response_ids: list[list[int]], texts: list[str]
    ) -> None:
        """Warn (loudly, with counts) on under-denoised responses.

        Two signatures: raw response ids still containing the mask token
        (denoising did not finish) or a decoded response that is empty or
        punctuation-only. Surfaces silent corruption that the parser would
        otherwise bury under a fallback label.
        """
        n_mask = sum(1 for ids in response_ids if self.mask_id in ids)
        n_empty = sum(1 for t in texts if self._PUNCT_ONLY.match(t or ""))
        n_bad = max(n_mask, n_empty)
        if n_bad:
            self._degenerate_count += n_bad
            warnings.warn(
                f"DreamModel: {n_mask}/{len(texts)} responses still contain mask "
                f"tokens and {n_empty}/{len(texts)} decoded to empty/punctuation "
                f"only (under-denoising). steps={self.steps}. Increase "
                f"sampling_steps or reduce batch_size.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _format_prompt(self, prompt: str) -> str:
        if self.use_chat_template:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            # Fallback: hardcoded Qwen2/Dream chat template (used when tokenizer
            # chat_template attr is missing or None in some transformers versions).
            return (
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
        return prompt

    @torch.no_grad()
    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Tuple[list[int], str]:
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        ).to(self._device)
        n_prompt = inputs["input_ids"].shape[1]

        # Official Dream generation API: model.diffusion_generate()
        # temperature=0.0 → deterministic (greedy); alg controls remasking strategy
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            steps=self._effective_steps(max_new_tokens),
            temperature=temperature,
            alg=self.alg,
        )
        if self.alg_temp is not None:
            gen_kwargs["alg_temp"] = self.alg_temp

        output = self.model.diffusion_generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        if hasattr(output, "sequences"):
            generated_ids = output.sequences
        else:
            generated_ids = output

        response_ids = generated_ids[0, n_prompt:].tolist()
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        self._flag_degenerate([response_ids], [response_text])
        return response_ids, response_text

    @torch.no_grad()
    def _generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Tuple[list[list[int]], list[str]]:
        formatted = [self._format_prompt(p) for p in prompts]
        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self._device)
        n_prompt = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            steps=self._effective_steps(max_new_tokens),
            temperature=temperature,
            alg=self.alg,
        )
        if self.alg_temp is not None:
            gen_kwargs["alg_temp"] = self.alg_temp

        output = self.model.diffusion_generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        generated_ids = output.sequences if hasattr(output, "sequences") else output

        all_tokens, all_texts = [], []
        for i in range(len(prompts)):
            resp_ids = generated_ids[i, n_prompt:].tolist()
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            all_tokens.append(resp_ids)
            all_texts.append(text)
        self._flag_degenerate(all_tokens, all_texts)
        return all_tokens, all_texts

    def get_responses(
        self,
        prompts: list[str],
        batched: bool = False,
        **kwargs,
    ) -> Tuple[list[list[int]], list[str]]:
        max_new_tokens = kwargs.get("max_new_tokens", 64)
        temperature = kwargs.get("temperature", 0.0)
        if batched and len(prompts) > 1:
            return self._generate_batch(prompts, max_new_tokens=max_new_tokens, temperature=temperature)
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

    @torch.no_grad()
    def _compute_elbo_single(
        self, prompt: str, response_tokens: list[int]
    ) -> float:
        """ELBO approximation via Monte Carlo masking (same as LLaDA).

        Dream uses the same masked diffusion objective so model(x_t) returns
        logits in the same format and the ELBO estimate is directly comparable.
        """
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

            out = self.model(x_t.unsqueeze(0), attention_mask=attention_mask)
            if isinstance(out, torch.Tensor):
                logits = out
            elif hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, tuple):
                logits = out[0]
            else:
                raise RuntimeError(f"Unexpected model output type: {type(out)}")

            resp_logits = logits[0, n_prompt:, :].clone()
            del logits
            resp_logits[:, self.mask_id] = -1e6
            log_probs = torch.log_softmax(resp_logits, dim=-1)
            del resp_logits

            pred_lp = log_probs.gather(1, x0_resp.unsqueeze(1)).squeeze(1)
            del log_probs
            nll = -(pred_lp * to_mask_resp.float()).sum()
            total_weighted += nll.item() / t

        return -(total_weighted / n_mc)
