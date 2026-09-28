import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prime_moment_attention.deepseek_671b_oracle import DeepSeek671BOracle, get_api_key

class TestDeepSeek671BOracle(unittest.TestCase):
    def test_unconfigured_state(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("os.path.exists", return_value=False):
            oracle = DeepSeek671BOracle()
            self.assertFalse(oracle.is_available())
            self.assertIsNone(oracle.query_derivation("Solve 2+2"))

    def test_openrouter_mocked_response(self):
        mock_payload = {
            "choices": [
                {
                    "message": {
                        "content": "4",
                        "reasoning_content": "2 + 2 equals 4 by basic arithmetic."
                    }
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-mock-test-key"}), \
             patch("urllib.request.urlopen", return_value=mock_response):
            oracle = DeepSeek671BOracle()
            self.assertTrue(oracle.is_available())
            self.assertEqual(oracle.provider, "openrouter")
            
            res = oracle.query_derivation("What is 2+2?")
            self.assertIsNotNone(res)
            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["content"], "4")
            self.assertEqual(res["reasoning_trace"], "2 + 2 equals 4 by basic arithmetic.")
            self.assertIn("<think>\n2 + 2 equals 4 by basic arithmetic.\n</think>", res["full_output"])

    def test_think_tag_in_content_extraction(self):
        mock_payload = {
            "choices": [
                {
                    "message": {
                        "content": "<think>\nLet x = 5.\n</think>\nTherefore x = 5."
                    }
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-mock-deepseek-key"}), \
             patch("urllib.request.urlopen", return_value=mock_response):
            oracle = DeepSeek671BOracle()
            self.assertTrue(oracle.is_available())
            self.assertEqual(oracle.provider, "deepseek")
            
            res = oracle.query_derivation("Find x.")
            self.assertIsNotNone(res)
            self.assertEqual(res["reasoning_trace"], "Let x = 5.")
            self.assertEqual(res["content"], "Therefore x = 5.")

if __name__ == "__main__":
    unittest.main()
