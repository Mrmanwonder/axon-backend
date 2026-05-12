import unittest

from functions.services.grading_service import HandwritingGradingGateway


class GradingServiceHeuristicTest(unittest.TestCase):
    def setUp(self):
        self.gateway = HandwritingGradingGateway.__new__(HandwritingGradingGateway)

    def test_state_depth_requires_short_answer(self):
        depth = self.gateway._analyze_command_word_depth(
            command_word="state",
            student_answer="momentum is defined as mass times velocity",
        )

        self.assertFalse(depth["depth_satisfied"])
        self.assertGreaterEqual(depth["word_count"], 6)

    def test_explain_depth_requires_causal_language(self):
        depth = self.gateway._analyze_command_word_depth(
            command_word="explain",
            student_answer="Momentum is conserved because the resultant force is zero.",
        )

        self.assertTrue(depth["depth_satisfied"])
        self.assertIn("because", depth["observed_features"]["causal_markers"])

    def test_semantic_match_lexical_fallback_finds_relevant_points(self):
        result = self.gateway._lexical_semantic_match(
            student_answer="Momentum stays constant because the resultant force is zero.",
            points=[
                "resultant force is zero",
                "momentum is mass x velocity",
                "energy is conserved",
            ],
        )

        self.assertGreater(result["overall_score"], 0.2)
        self.assertIn("resultant force is zero", result["matched_points"])


if __name__ == "__main__":
    unittest.main()
