"""
Autoregressive model wrapper.

Thin wrapper around the POSIX library's HFModel, adapted to our
LanguageModel interface.  Works with any HuggingFace causal LM:
  - openai-community/gpt2
  - EleutherAI/pythia-*
  - meta-llama/Llama-*
  - etc.
"""

import copy
from typing import Tuple

import torch
from torch.nn.utils import rnn
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import LanguageModel

IGNORE_INDEX = -100


class ARModel(LanguageModel):
    """Wrapper for autoregressive (causal) language models."""

    def __init__(
        self,
        model_name_or_path: str,
        device_str: str = "auto",
        use_chat_template: bool = True,
        **model_kwargs,
    ):
        """
        Args:
            model_name_or_path: HuggingFace model ID or local path.
            device_str: 'cpu', 'cuda', 'cuda:0', or 'auto'.
            **model_kwargs: Extra kwargs for AutoModelForCausalLM.from_pretrained
                            (e.g. torch_dtype=torch.float16).
        """
        self._name = model_name_or_path
        self.use_chat_template = use_chat_template

        # Resolve device
        if device_str == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device_str

        trust_remote_code = model_kwargs.pop("trust_remote_code", False)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        ).to(self._device)
        self.model.eval()

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            elif self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # --- LanguageModel interface ---

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
        """Generate continuations for each prompt."""
        if batched:
            return self._get_responses_batched(prompts, **kwargs)

        all_tokens, all_texts = [], []
        for prompt in prompts:
            tokens, text = self._get_response_single(prompt, **kwargs)
            all_tokens.append(tokens)
            all_texts.append(text)
        return all_tokens, all_texts

    def compute_log_probabilities(
        self,
        prompts: list[str],
        responses: list[list[int]],
        batched: bool = False,
    ) -> list[float]:
        """Compute log P(response | prompt) using forward-pass logits."""
        if batched:
            return self._compute_log_probs_batched(prompts, responses)

        log_probs = []
        for prompt, resp_tokens in zip(prompts, responses):
            lp = self._compute_log_prob_single(prompt, resp_tokens)
            log_probs.append(lp)
        return log_probs

    # --- Internal methods ---

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

    @torch.no_grad()
    def _get_response_single(
        self, prompt: str, **kwargs
    ) -> Tuple[list[int], str]:
        formatted = self._format_prompt(prompt)
        tokenized = self.tokenizer(formatted, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self._device)
        attention_mask = tokenized.attention_mask.to(self._device)
        n_prompt = input_ids.shape[1]

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        response_tokens = output[0][n_prompt:]
        response_text = self.tokenizer.decode(
            response_tokens, skip_special_tokens=True
        )
        return response_tokens.tolist(), response_text

    @torch.no_grad()
    def _get_responses_batched(
        self, prompts: list[str], **kwargs
    ) -> Tuple[list[list[int]], list[str]]:
        formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
        tokenized = self.tokenizer(formatted_prompts)
        input_ids, attention_mask = tuple(
            [
                torch.tensor(feat[::-1])
                for feat in tokenized[key]
            ]
            for key in ["input_ids", "attention_mask"]
        )
        input_ids = rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        attention_mask = rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0,
        ).flip(dims=[1])

        n_padded = input_ids.shape[1]
        generated = self.model.generate(
            input_ids=input_ids.to(self._device),
            attention_mask=attention_mask.to(self._device),
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        generated = generated[:, n_padded:]
        texts = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )
        return generated.tolist(), texts

    @torch.no_grad()
    def _compute_log_prob_single(
        self, prompt: str, response_tokens: list[int]
    ) -> float:
        prompt_tokens = self.tokenizer(self._format_prompt(prompt)).input_ids
        full_tokens = prompt_tokens + response_tokens
        n_prompt = len(prompt_tokens)

        input_ids = torch.tensor([full_tokens]).to(self._device)
        logits = self.model(input_ids).logits

        # Extract logits corresponding to response positions
        response_logits = logits[:, n_prompt - 1 : len(full_tokens) - 1, :]
        log_probs = torch.log_softmax(response_logits, dim=-1)

        total_log_prob = 0.0
        for k, token_id in enumerate(response_tokens):
            total_log_prob += log_probs[0, k, token_id].item()
        return total_log_prob

    @torch.no_grad()
    def _compute_log_probs_batched(
        self, prompts: list[str], responses: list[list[int]]
    ) -> list[float]:
        formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
        tokenized_prompts = self.tokenizer(formatted_prompts)
        tokenized_responses = {
            "input_ids": responses,
            "attention_mask": [[1] * len(r) for r in responses],
        }
        # Concatenate prompt + response
        combined = {
            key: [p + r for p, r in zip(
                tokenized_prompts[key], tokenized_responses[key]
            )]
            for key in ["input_ids", "attention_mask"]
        }
        # Build labels: mask out prompt positions
        all_labels = copy.deepcopy(combined["input_ids"])
        prompt_lengths = [
            len(toks) for toks in tokenized_prompts["input_ids"]
        ]
        for labels, pl in zip(all_labels, prompt_lengths):
            labels[:pl] = [IGNORE_INDEX] * pl
        combined["labels"] = all_labels

        # Left-pad and create tensors
        input_ids, attention_mask, labels = tuple(
            [torch.tensor(feat[::-1]) for feat in combined[key]]
            for key in ["input_ids", "attention_mask", "labels"]
        )
        input_ids = rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        attention_mask = rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0,
        ).flip(dims=[1])
        labels = rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX,
        ).flip(dims=[1])

        batch = dict(
            input_ids=input_ids.to(self._device),
            attention_mask=attention_mask.to(self._device),
            labels=labels.to(self._device),
        )
        outputs = self.model(**batch)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)

        final_log_probs = []
        for i in range(log_probs.shape[0]):
            total_lp = 0.0
            valid_positions = (labels[i] != IGNORE_INDEX).nonzero()
            if len(valid_positions) == 0:
                final_log_probs.append(0.0)
                continue
            start_idx = valid_positions[0]
            resp_tokens = input_ids[i][start_idx:]
            resp_log_probs = log_probs[i][start_idx - 1 : -1]
            for j, token in enumerate(resp_tokens):
                total_lp += resp_log_probs[j, token].item()
            final_log_probs.append(total_lp)
        return final_log_probs
