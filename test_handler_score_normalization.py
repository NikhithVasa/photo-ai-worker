import ast
import json
import math
from pathlib import Path
import re
import unittest
from typing import Any, Dict, List, Optional


def _load_score_functions():
    """Load pure score helpers without executing handler module initialization."""
    handler_path = Path(__file__).with_name("handler.py")
    tree = ast.parse(handler_path.read_text(encoding="utf-8"), filename=str(handler_path))
    wanted = {"clamp_score", "safe_int", "normalize_qwen_data"}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    found = {node.name for node in definitions}
    if found != wanted:
        missing = ", ".join(sorted(wanted - found))
        raise RuntimeError(f"Could not load score helpers from handler.py: {missing}")

    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "json": json,
        "math": math,
        "re": re,
    }
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(handler_path), "exec"), namespace)
    return namespace["clamp_score"], namespace["normalize_qwen_data"]


clamp_score, normalize_qwen_data = _load_score_functions()


class ClampScoreTests(unittest.TestCase):
    def test_missing_invalid_and_non_finite_scores_are_unknown(self):
        cases = {
            "missing": None,
            "true is not one": True,
            "false is not zero": False,
            "empty string": "",
            "invalid string": "not-a-score",
            "invalid collection": [],
            "nan": float("nan"),
            "positive infinity": float("inf"),
            "negative infinity": float("-inf"),
        }

        for name, value in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(clamp_score(value))

    def test_zero_rounding_and_bounds_preserve_score_semantics(self):
        cases = {
            "explicit zero": (0, 0),
            "round down": (4.4, 4),
            "round up numeric string": ("4.6", 5),
            "clamp below range": (-1, 0),
            "clamp above range": (11, 10),
        }

        for name, (value, expected) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(clamp_score(value), expected)


class NormalizeQwenDataScoreTests(unittest.TestCase):
    def test_missing_quality_scores_remain_null_and_search_as_unknown(self):
        normalized = normalize_qwen_data({"people": {"Person 1": {}}})

        self.assertEqual(
            {
                "background_quality": None,
                "frame_clarity": None,
                "album_worthy_score": None,
                "print_worthy_score": None,
            },
            {
                key: normalized["quality"][key]
                for key in (
                    "background_quality",
                    "frame_clarity",
                    "album_worthy_score",
                    "print_worthy_score",
                )
            },
        )
        self.assertEqual(
            {
                "photo_quality_score": None,
                "personal_photo_quality_score": None,
                "person_cover_score": None,
            },
            {
                key: normalized["people_map"]["Person 1"][key]
                for key in (
                    "photo_quality_score",
                    "personal_photo_quality_score",
                    "person_cover_score",
                )
            },
        )

        search_text = normalized["photo"]["search_text"]
        for fragment in (
            "background_quality=unknown",
            "frame_clarity=unknown",
            "album_worthy_score=unknown",
            "print_worthy_score=unknown",
            "cover_score=unknown",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, search_text)

        serialized = json.loads(json.dumps(normalized))
        self.assertIsNone(serialized["quality"]["background_quality"])
        self.assertIsNone(serialized["people_map"]["Person 1"]["photo_quality_score"])

    def test_explicit_zero_scores_remain_zero(self):
        normalized = normalize_qwen_data(
            {
                "background_quality": 0,
                "frame_clarity": 0,
                "album_worthy_score": 0,
                "print_worthy_score": 0,
                "people": {
                    "Person 1": {
                        "photo_quality_score": 0,
                        "person_cover_score": 0,
                    }
                },
            }
        )

        self.assertEqual(normalized["quality"]["background_quality"], 0)
        self.assertEqual(normalized["quality"]["frame_clarity"], 0)
        self.assertEqual(normalized["quality"]["album_worthy_score"], 0)
        self.assertEqual(normalized["quality"]["print_worthy_score"], 0)
        person = normalized["people_map"]["Person 1"]
        self.assertEqual(person["photo_quality_score"], 0)
        self.assertEqual(person["personal_photo_quality_score"], 0)
        self.assertEqual(person["person_cover_score"], 0)

        search_text = normalized["photo"]["search_text"]
        self.assertIn("background_quality=0/10", search_text)
        self.assertIn("frame_clarity=0/10", search_text)
        self.assertIn("album_worthy_score=0/10", search_text)
        self.assertIn("print_worthy_score=0/10", search_text)
        self.assertIn("cover_score=0/10", search_text)


if __name__ == "__main__":
    unittest.main()
