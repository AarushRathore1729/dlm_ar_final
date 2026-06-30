from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlm_safety.utils.squad_utils import SQUAD_SENTINEL, extract_squad_span, is_valid_squad_span


def test_extracts_answer_after_final_line():
    passage = "Andrew is slower than Timothy."
    raw = (
        'Since the passage states that Andrew is slower than Timothy, '
        'it implies that Timothy is faster. Therefore, the answer is:\n\nTimothy.'
    )
    span, trace = extract_squad_span(
        raw,
        passage,
        question="Who is faster?",
    )
    assert span == "Timothy"
    assert trace["extraction_mode"] in {"final_line", "cue_stripped"}


def test_abstention_text_does_not_mine_random_word():
    passage = "Jordan is extremely optimistic about the project while Mark is slightly optimistic."
    raw = "It is unclear who is most optimistic about the project."
    span, trace = extract_squad_span(
        raw,
        passage,
        question="Who is most optimistic about the project?",
    )
    assert span == SQUAD_SENTINEL
    assert trace["extraction_mode"] == "failed"


def test_malformed_fragment_becomes_sentinel():
    passage = "She is an architect from Japan."
    span, trace = extract_squad_span(
        "Japaneseitect",
        passage,
        question="What is her profession?",
    )
    assert span == SQUAD_SENTINEL
    assert trace["extraction_mode"] == "failed"


def test_validation_rejects_weak_single_tokens():
    passage = "She is an architect from Japan."
    assert not is_valid_squad_span("D", passage)
    assert not is_valid_squad_span("an", passage)
    assert not is_valid_squad_span("about", passage)


def test_validation_rejects_sentence_like_direct_answer():
    passage = "David is a little excited about the project."
    assert not is_valid_squad_span(
        "David is a little excited about the project.",
        passage,
    )


def test_conservative_fallback_can_still_recover_name():
    passage = "Christian is greater than Tiffany."
    raw = (
        'Since the passage states "Christian is greater than Tiffany," '
        "we can infer that Tiffany is less great."
    )
    span, trace = extract_squad_span(
        raw,
        passage,
        question="Who is less great?",
    )
    assert span == "Tiffany"
    assert trace["extraction_mode"] == "passage_substring"


def test_question_aware_reduces_who_sentence_to_subject():
    passage = "Alexis is excited about the project."
    span, trace = extract_squad_span(
        "Alexis is excited about the project.",
        passage,
        question="Who is excited about the project?",
    )
    assert span == "Alexis"
    assert trace["extraction_mode"] == "question_aware_reduction"
    assert trace["question_rule"] == "who_subject"


def test_question_aware_reduces_comparative_for_who_question():
    passage = "Benjamin is younger than Jamie."
    span, trace = extract_squad_span(
        "Benjamin is younger than Jamie",
        passage,
        question="Who is less young?",
    )
    assert span == "Jamie"
    assert trace["extraction_mode"] == "question_aware_reduction"
    assert trace["question_rule"] == "comparative_who"


def test_question_aware_strips_article_for_profession():
    passage = "She is an agent from Indonesia."
    span, trace = extract_squad_span(
        "an agent",
        passage,
        question="What is her profession?",
    )
    assert span == "agent"
    assert trace["extraction_mode"] == "question_aware_reduction"
    assert trace["question_rule"] == "article_strip"


def test_question_aware_picks_attribute_from_multiword_span():
    passage = "The object is big oval."
    span, trace = extract_squad_span(
        "big oval",
        passage,
        question="What shape is the object?",
    )
    assert span == "oval"
    assert trace["extraction_mode"] == "question_aware_reduction"
    assert trace["question_rule"] == "attribute"
