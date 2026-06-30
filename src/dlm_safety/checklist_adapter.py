"""
CheckList predictor adapter: model under test + scorer -> (preds, probs).

Scorer can be LLMJudge (small LLM) or SentimentPipelineScorer (HF pipeline).
  task prompt -> model generates -> scorer maps response to label/probs.

For SQuAD, the model response IS the prediction (no judge needed).

Supports batched inference via batch_size parameter for GPU throughput.
"""

import random
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .models.base import LanguageModel
from .llm_judge import LLMJudge, SentimentPipelineScorer
from .utils.squad_utils import extract_squad_span


SENTIMENT_TASK_PROMPT = """You are a strict sentiment classifier.
Read the text and decide whether the sentiment is negative, neutral, or positive.
Reply with exactly one lowercase label: negative, neutral, or positive.
Do not explain your answer.
Do not repeat or quote the text.
If the sentiment is mixed, ambiguous, or unclear, reply with neutral.

{text}"""

QQP_TASK_PROMPT = """Determine whether these two questions are duplicates. Reply with exactly one label: duplicate or not_duplicate.

Question 1: {question1}
Question 2: {question2}"""

SQUAD_TASK_PROMPT = """Read the passage and answer the question.
Output exactly one answer span copied verbatim from the passage.
Do not explain.
Do not answer in a full sentence.
Do not repeat the question.
Do not add any words that are not in the passage.
Do not use quotes or prefixes like "Answer:".
If the answer is a short phrase, output only that phrase.

Passage: {passage}
Question: {question}
Answer:"""

MNLI_TASK_PROMPT = """Classify the logical relationship between the premise and the hypothesis.
Reply with exactly one label: entailment, neutral, or contradiction.
Do not explain your answer.
Do not output any other words or punctuation.

Premise: {premise}
Hypothesis: {hypothesis}
Label:"""

PAWS_TASK_PROMPT = """Determine whether these two sentences are paraphrases of each other.
Reply with exactly one label: duplicate or not_duplicate.
Do not explain your answer.

Sentence 1: {sentence1}
Sentence 2: {sentence2}
Label:"""

# ANLI uses the same NLI prompt template as MNLI; kept separate so the task
# string round-trips cleanly through configs and result naming.
ANLI_TASK_PROMPT = """Classify the logical relationship between the premise and the hypothesis.
Reply with exactly one label: entailment, neutral, or contradiction.
Do not explain your answer.
Do not output any other words or punctuation.

Premise: {premise}
Hypothesis: {hypothesis}
Label:"""

# WildGuardMix: prompt-only safety classification. The model is asked whether
# it would refuse the request. Single binary label (refuse / comply) so the
# downstream judge can compute γ/α on the safety axis.
WILDGUARD_TASK_PROMPT = """{prompt}"""

# XSTest: same input shape as WildGuard (single prompt), but the labels are
# the over-refusal taxonomy (refuse / comply). Kept separate for naming.
XSTEST_TASK_PROMPT = """{prompt}"""


class LMPredictorForCheckList:
    """Wrap a generative LanguageModel + scorer for CheckList."""

    def __init__(
        self,
        model: LanguageModel,
        judge: Union[LLMJudge, SentimentPipelineScorer, None],
        task: str = "sentiment",
        task_prompt_template: Optional[str] = None,
        max_new_tokens: int = 64,
        batch_size: int = 1,
        max_examples_per_test: Optional[int] = None,
    ):
        self.model = model
        self.judge = judge
        self.task = task
        self.task_prompt_template = task_prompt_template or self._default_task_prompt(task)
        if self.task_prompt_template is None:
            raise ValueError(f"Unknown task {task!r}; provide task_prompt_template.")
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, batch_size)
        self.max_examples_per_test = max_examples_per_test
        self.trace_log: list[list[dict]] = []
        self.stats_log: list[dict] = []

    @staticmethod
    def _default_task_prompt(task: str) -> Optional[str]:
        if task == "sentiment":
            return SENTIMENT_TASK_PROMPT
        if task == "qqp":
            return QQP_TASK_PROMPT
        if task == "squad":
            return SQUAD_TASK_PROMPT
        if task == "mnli":
            return MNLI_TASK_PROMPT
        if task == "paws":
            return PAWS_TASK_PROMPT
        if task == "anli":
            return ANLI_TASK_PROMPT
        if task in ("wildguard", "wildguardmix"):
            return WILDGUARD_TASK_PROMPT
        if task == "xstest":
            return XSTEST_TASK_PROMPT
        return None

    @staticmethod
    def _serialize_input(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            q1 = item.get("question1", item.get("q1"))
            q2 = item.get("question2", item.get("q2"))
            if q1 is not None and q2 is not None:
                return f"Q1: {q1}\nQ2: {q2}"
            passage = item.get("passage")
            question = item.get("question")
            if passage is not None and question is not None:
                return f"Passage: {passage}\nQuestion: {question}"
            premise = item.get("premise")
            hypothesis = item.get("hypothesis")
            if premise is not None and hypothesis is not None:
                return f"Premise: {premise}\nHypothesis: {hypothesis}"
            s1 = item.get("sentence1")
            s2 = item.get("sentence2")
            if s1 is not None and s2 is not None:
                return f"Sentence 1: {s1}\nSentence 2: {s2}"
            return str(item)
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return f"Q1: {item[0]}\nQ2: {item[1]}"
        return str(item)

    def _format_prompt(self, item: Any) -> str:
        text = self._serialize_input(item)
        if self.task == "qqp":
            if isinstance(item, (tuple, list)) and len(item) == 2:
                return self.task_prompt_template.format(
                    text=text,
                    question1=item[0],
                    question2=item[1],
                )
            if isinstance(item, dict):
                q1 = item.get("question1", item.get("q1"))
                q2 = item.get("question2", item.get("q2"))
                if q1 is not None and q2 is not None:
                    return self.task_prompt_template.format(
                        text=text,
                        question1=q1,
                        question2=q2,
                    )
        if self.task == "squad" and isinstance(item, dict):
            passage = item.get("passage", "")
            question = item.get("question", "")
            return self.task_prompt_template.format(
                text=text,
                passage=passage,
                question=question,
            )
        if self.task == "mnli":
            if isinstance(item, (tuple, list)) and len(item) == 2:
                return self.task_prompt_template.format(
                    text=text, premise=item[0], hypothesis=item[1],
                )
            if isinstance(item, dict):
                return self.task_prompt_template.format(
                    text=text,
                    premise=item.get("premise", ""),
                    hypothesis=item.get("hypothesis", ""),
                )
        if self.task == "paws":
            if isinstance(item, (tuple, list)) and len(item) == 2:
                return self.task_prompt_template.format(
                    text=text, sentence1=item[0], sentence2=item[1],
                )
            if isinstance(item, dict):
                return self.task_prompt_template.format(
                    text=text,
                    sentence1=item.get("sentence1", ""),
                    sentence2=item.get("sentence2", ""),
                )
        if self.task == "anli":
            if isinstance(item, (tuple, list)) and len(item) == 2:
                return self.task_prompt_template.format(
                    text=text, premise=item[0], hypothesis=item[1],
                )
            if isinstance(item, dict):
                return self.task_prompt_template.format(
                    text=text,
                    premise=item.get("premise", ""),
                    hypothesis=item.get("hypothesis", ""),
                )
        if self.task in ("wildguard", "wildguardmix", "xstest"):
            # Single-prompt safety tasks: the input string IS the prompt.
            prompt_str = text if isinstance(text, str) else str(text)
            return self.task_prompt_template.format(text=prompt_str, prompt=prompt_str)
        return self.task_prompt_template.format(text=text)

    def _generate_responses(self, prompts: list[str]) -> list[str]:
        """Generate model responses, using batched inference when batch_size > 1."""
        use_batched = self.batch_size > 1
        if not use_batched:
            all_responses: list[str] = []
            for prompt in tqdm(prompts, desc="generating", unit="ex", dynamic_ncols=True):
                _, resp = self.model.get_responses(
                    [prompt],
                    batched=False,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
                all_responses.extend(resp)
            return all_responses

        all_responses = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="generating batches", unit="batch", dynamic_ncols=True):
            chunk = prompts[start : start + self.batch_size]
            _, chunk_responses = self.model.get_responses(
                chunk,
                batched=True,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            all_responses.extend(chunk_responses)
        return all_responses

    def predict(self, data: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Run model then judge on each input. Return (preds, probs) for CheckList."""
        if self.max_examples_per_test and len(data) > self.max_examples_per_test:
            data = random.sample(data, self.max_examples_per_test)
        print(f"\nPredicting {len(data)} examples", flush=True)
        prompts = [self._format_prompt(item) for item in data]
        responses = self._generate_responses(prompts)
        judge_inputs = [self._serialize_input(item) for item in data]
        preds, probs = self.judge.evaluate(judge_inputs, responses)
        judge_trace = getattr(self.judge, "last_trace", [])
        judge_stats = getattr(self.judge, "last_stats", {})
        batch_trace = []
        for idx, (text, prompt, response) in enumerate(zip(judge_inputs, prompts, responses)):
            judge_info = judge_trace[idx] if idx < len(judge_trace) else {}
            batch_trace.append(
                {
                    "input_text": text,
                    "prompt": prompt,
                    "model_response": response,
                    "predicted_label_idx": int(preds[idx]),
                    "predicted_label": judge_info.get("predicted_label"),
                    "probabilities": probs[idx].tolist(),
                    "judge_output": judge_info.get("judge_output"),
                    "parse_mode": judge_info.get("parse_mode"),
                    "repair_attempted": judge_info.get("repair_attempted"),
                    "repair_output": judge_info.get("repair_output"),
                    "used_judge_fallback": judge_info.get("used_judge_fallback"),
                }
            )
        self.trace_log.append(batch_trace)
        self.stats_log.append(judge_stats)
        return preds, probs

    def predict_proba(self, data: List[Any]) -> np.ndarray:
        """Return only class probabilities for CheckList's softmax wrapper."""
        _preds, probs = self.predict(data)
        return probs

    def predict_squad(self, data: List[Any]) -> Tuple[list, list]:
        """SQuAD predict: model response IS the prediction (no judge).

        Returns (preds, confs) where preds are answer strings and confs are
        floats (1.0 for all since we have no real confidence from generation).
        CheckList's Expect.eq() will compare preds to expected answer labels.
        """
        prompts = [self._format_prompt(item) for item in data]
        responses = self._generate_responses(prompts)
        preds = []
        confs = []
        batch_trace = []
        for idx, (item, prompt, response) in enumerate(zip(data, prompts, responses)):
            passage = ""
            question = ""
            if isinstance(item, dict):
                passage = item.get("passage", "") or ""
                question = item.get("question", "") or ""
            answer, extraction_trace = extract_squad_span(
                response, passage, question=question
            )
            extraction_trace = extraction_trace or {
                "extraction_mode": "unknown",
                "stripped_cue": None,
            }
            response_changed = answer != (response or "").strip()
            final_mode = extraction_trace.get("extraction_mode", "unknown")
            preds.append(answer)
            confs.append(1.0)
            batch_trace.append(
                {
                    "input_text": self._serialize_input(item),
                    "prompt": prompt,
                    "model_response": response,
                    "predicted_label": answer,
                    "predicted_label_idx": None,
                    "probabilities": None,
                    "judge_output": None,
                    # Keep parse_mode stable for downstream code; detailed extraction
                    # behavior is logged via extraction_trace/normalization fields.
                    "parse_mode": "direct_generation",
                    "extraction_trace": extraction_trace,
                    "raw_response_before_repair": response,
                    "normalization": {
                        "response_changed": response_changed,
                        "final_mode": final_mode,
                    },
                }
            )
        self.trace_log.append(batch_trace)
        self.stats_log.append({"total_examples": len(data)})
        return preds, confs

    def reset_trace_log(self) -> None:
        self.trace_log = []
        self.stats_log = []
