"""Unit tests for DreamModel's under-denoising guards (no GPU / no model load).

These exercise the pure logic added to prevent the bare-"." failure from
recurring: the per-token step floor and the degenerate-output detector. We build
a bare DreamModel via __new__ to avoid loading the 7B checkpoint.
"""

import warnings

import pytest

from dlm_safety.models.dream_model import DreamModel


def _bare_model(steps: int) -> DreamModel:
    m = DreamModel.__new__(DreamModel)
    m.steps = steps
    m.mask_id = DreamModel._DEFAULT_MASK_TOKEN_ID
    m._warned_steps = False
    m._degenerate_count = 0
    return m


def test_step_floor_raises_low_steps():
    m = _bare_model(steps=8)
    with pytest.warns(RuntimeWarning):
        eff = m._effective_steps(max_new_tokens=16)
    # 2 * 16 = 32 is the floor
    assert eff == 32


def test_step_floor_warns_only_once():
    m = _bare_model(steps=8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m._effective_steps(16)
        m._effective_steps(16)
    assert sum(issubclass(x.category, RuntimeWarning) for x in w) == 1


def test_step_floor_leaves_sufficient_steps_unchanged():
    # Well-configured run (matches LLaDA density) must be untouched -> comparable.
    m = _bare_model(steps=64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        assert m._effective_steps(max_new_tokens=16) == 64
        assert m._effective_steps(max_new_tokens=32) == 64  # exactly at floor


def test_flag_degenerate_on_leftover_mask():
    m = _bare_model(steps=64)
    ids = [[1, 2, 3], [4, m.mask_id, 6]]  # second row never finished denoising
    with pytest.warns(RuntimeWarning):
        m._flag_degenerate(ids, ["duplicate", "dup"])
    assert m._degenerate_count == 1


def test_flag_degenerate_on_punct_only_text():
    m = _bare_model(steps=64)
    with pytest.warns(RuntimeWarning):
        m._flag_degenerate([[1], [2]], [".", "._duplicate"][:1] + ["."])
    assert m._degenerate_count >= 1


def test_no_warning_on_clean_batch():
    m = _bare_model(steps=64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m._flag_degenerate([[1, 2], [3, 4]], ["duplicate", "not_duplicate"])
    assert m._degenerate_count == 0
