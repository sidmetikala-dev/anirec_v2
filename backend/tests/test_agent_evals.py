"""Verify evaluation assertions without calling an LLM."""
import json
import unittest

from anirec.agent_service import AgentAnswer
from evals.run_agent_evals import CASES_PATH, check_case, fixtures


class AgentEvalTests(unittest.TestCase):
    def setUp(self):
        self.tools, _ = fixtures()
        self.cases = {c["id"]: c for c in json.loads(CASES_PATH.read_text())}

    def answer(self, ids, error=""):
        return AgentAnswer.model_validate({"error":error, "recommendations":[
            {"anime_id":i,"explanation":"Fixture explanation"} for i in ids]})

    def test_bare_username_detects_invented_filter(self):
        failures = check_case(self.cases["bare_username"], self.answer([20]),
            [{"name":"get_recs_with_username_tool","arguments":{}},
             {"name":"filter_anime","arguments":{"max_episodes":20}}], self.tools.anime_data)
        self.assertIn("Unexpected tool: filter_anime", failures)

    def test_ranking_and_episode_rules_pass_for_valid_results(self):
        ids = [44511,37999,37450,5940]
        calls = [{"name":"filter_anime", "arguments":{"max_episodes":19},
                  "result":{str(i):1-index*.1 for index,i in enumerate(ids)}}]
        self.assertEqual(check_case(self.cases["under_20"], self.answer(ids), calls, self.tools.anime_data), [])

    def test_zero_episode_counts_fail(self):
        failures = check_case(self.cases["under_20"], self.answer([21]),
            [{"name":"filter_anime","arguments":{"max_episodes":19},"result":{"21":1}}], self.tools.anime_data)
        self.assertIn("Final episode constraint violated", failures)

    def test_missing_user_requires_error_and_no_results(self):
        calls = [{"name":"get_recs_with_username_tool","arguments":{}}]
        self.assertEqual(check_case(self.cases["missing_user"], self.answer([], "User not found"), calls, self.tools.anime_data), [])
        self.assertTrue(check_case(self.cases["missing_user"], self.answer([20]), calls, self.tools.anime_data))

    def test_wrong_final_order_is_detected(self):
        failures = check_case(self.cases["bare_username"], self.answer([1535,20]),
            [{"name":"get_recs_with_username_tool","arguments":{},"result":{"20":1,"1535":.9}}], self.tools.anime_data)
        self.assertTrue(any("backend top five" in failure for failure in failures))

    def test_cases_have_unique_ids(self):
        cases = json.loads(CASES_PATH.read_text())
        self.assertEqual(len(cases), 15)
        self.assertEqual(len(cases), len(self.cases))


if __name__ == "__main__":
    unittest.main()
