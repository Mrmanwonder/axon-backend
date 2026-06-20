import unittest

from functions.services.mark_scheme_layout_service import MarkSchemeLayoutService


class MarkSchemeLayoutServiceTest(unittest.TestCase):
    def setUp(self):
        self.service = MarkSchemeLayoutService()

    def test_segments_using_layout_lines(self):
        pages = [
            {
                "page_number": 1,
                "width": 1000,
                "height": 1500,
                "lines": [
                    {
                        "page_number": 1,
                        "text": "1 (a) momentum = mass x velocity",
                        "x0": 80,
                        "y0": 120,
                        "x1": 500,
                        "y1": 140,
                    },
                    {
                        "page_number": 1,
                        "text": "(b) resultant force is zero",
                        "x0": 80,
                        "y0": 160,
                        "x1": 480,
                        "y1": 180,
                    },
                    {
                        "page_number": 1,
                        "text": "2 (a) energy is conserved",
                        "x0": 80,
                        "y0": 260,
                        "x1": 420,
                        "y1": 280,
                    },
                ],
            }
        ]
        question = {
            "item_id": "1:1",
            "question_number": 1,
            "question_text": "Explain why momentum is conserved.",
            "page_number": 1,
            "parts": ["b"],
            "spatial_metadata": {
                "normalized_bounds": {"x": 0.08, "y": 0.1, "width": 0.3, "height": 0.1}
            },
        }

        snippet = self.service.segment_question(
            mark_scheme_text="",
            page_layouts=pages,
            question=question,
        )

        self.assertIn("resultant force is zero", snippet)
        self.assertNotIn("energy is conserved", snippet)

    def test_falls_back_to_text_boundaries(self):
        question = {
            "item_id": "1:3",
            "question_number": 3,
            "question_text": "State the definition of momentum.",
            "page_number": 1,
            "parts": [],
            "spatial_metadata": {},
        }
        text = "\n".join(
            [
                "2 (a) old answer",
                "3 momentum = mass x velocity",
                "allow p = mv",
                "4 another answer",
            ]
        )

        snippet = self.service.segment_question(
            mark_scheme_text=text,
            page_layouts=[],
            question=question,
        )

        self.assertIn("momentum = mass x velocity", snippet)
        self.assertNotIn("another answer", snippet)


if __name__ == "__main__":
    unittest.main()
