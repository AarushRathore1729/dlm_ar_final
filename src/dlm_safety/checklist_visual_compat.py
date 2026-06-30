"""Compatibility helpers for CheckList visualization APIs.

This module adds a lightweight adapter so callers can use:

    suite = attach_visual_api(suite)
    suite.visual.suite(...)
    suite.visual.test(...)

Internally these map to original CheckList APIs:
- suite.visual_summary_table(...)
- suite.visual_summary_by_test(testname)

If widget rendering is unavailable (common outside classic notebook), the
adapter falls back to suite.summary() instead of failing hard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VisualCompat:
    suite_obj: Any

    def suite(self, types: list[str] | None = None, capabilities: list[str] | None = None):
        """Compatibility alias for suite-level visual summary.

        Preferred target API in original CheckList: suite.visual_summary_table().
        """
        try:
            return self.suite_obj.visual_summary_table(types=types, capabilities=capabilities)
        except Exception as exc:
            print(
                "visual_summary_table() is unavailable in this environment "
                f"({type(exc).__name__}: {exc}). Falling back to textual summary."
            )
            try:
                self.suite_obj.summary(types=types, capabilities=capabilities)
            except Exception as summary_exc:
                print(
                    "No visualization/summary can be shown yet "
                    f"({type(summary_exc).__name__}: {summary_exc}). "
                    "Run `suite.run(...)` or `suite.run_from_file(...)` first."
                )
            return None

    def test(self, test_name: str):
        """Compatibility alias for single-test visual summary.

        Preferred target API in original CheckList: suite.visual_summary_by_test(test_name).
        """
        try:
            return self.suite_obj.visual_summary_by_test(test_name)
        except Exception as exc:
            print(
                "visual_summary_by_test() is unavailable in this environment "
                f"({type(exc).__name__}: {exc}). Falling back to textual summary."
            )
            tests = getattr(self.suite_obj, "tests", {})
            if test_name in tests:
                try:
                    tests[test_name].summary()
                except Exception as summary_exc:
                    print(
                        "No visualization/summary can be shown yet "
                        f"({type(summary_exc).__name__}: {summary_exc}). "
                        "Run the test first."
                    )
                return None
            raise


def attach_visual_api(suite: Any):
    """Attach `suite.visual` compatibility API and return the same suite object."""
    setattr(suite, "visual", VisualCompat(suite))
    return suite
