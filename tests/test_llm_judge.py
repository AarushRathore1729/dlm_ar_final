from dlm_safety.llm_judge import _parse_direct_label_only


def test_parse_direct_label_at_start_of_same_line_explanation():
    label, mode = _parse_direct_label_only(
        "Not_duplicate. The questions ask about different subjects.",
        ("not_duplicate", "duplicate"),
    )

    assert label == 0
    assert mode == "direct_response_leading_label"


def test_do_not_parse_label_mentioned_later_in_explanation():
    label, mode = _parse_direct_label_only(
        "The correct classification is duplicate.",
        ("not_duplicate", "duplicate"),
    )

    assert label is None
    assert mode is None
