import unittest
from unittest.mock import patch, MagicMock
import os
import json
from exprouter.router import Router, ProviderExhaustedError, MalformedResponseError, BudgetExceededError

class TestRouter(unittest.TestCase):
    def setUp(self):
        # Setup a dummy config for testing
        self.test_config = [
            {"name": "anthropic", "priority": 1, "rpm": 2, "rpd": 10},
            {"name": "openai", "priority": 2, "rpm": 5, "rpd": 10}
        ]
        
    @patch("exprouter.router.load_router_config")
    @patch("exprouter.providers.anthropic.call")
    def test_provider_selection_and_success(self, mock_anthropic_call, mock_load_config):
        mock_load_config.return_value = self.test_config
        mock_anthropic_call.return_value = ("Success", 10, 5)
        
        router = Router()
        # First call should hit Anthropic
        resp = router.complete("claude-3-5-sonnet", [{"role": "user", "content": "hi"}])
        self.assertEqual(resp, "Success")
        self.assertEqual(mock_anthropic_call.call_count, 1)

    @patch("exprouter.router.PROVIDER_MODELS", {"test-model": {"anthropic": "a", "openai": "o"}})
    @patch("exprouter.router.load_router_config")
    @patch("exprouter.providers.anthropic.call")
    @patch("exprouter.providers.openai.call")
    def test_rate_limiting_fallback(self, mock_openai_call, mock_anthropic_call, mock_load_config):
        mock_load_config.return_value = self.test_config
        mock_anthropic_call.return_value = ("Anthropic Response", 10, 5)
        mock_openai_call.return_value = ("OpenAI Response", 10, 5)
        
        router = Router()
        # rpm is 2 for Anthropic
        router.complete("test-model", [{"role": "user", "content": "1"}])
        router.complete("test-model", [{"role": "user", "content": "2"}])
        self.assertEqual(mock_anthropic_call.call_count, 2)
        
        # 3rd call should hit OpenAI because Anthropic is rate-limited (rpm=2)
        resp = router.complete("test-model", [{"role": "user", "content": "3"}])
        self.assertEqual(resp, "OpenAI Response")
        self.assertEqual(mock_openai_call.call_count, 1)

    @patch("exprouter.router.load_router_config")
    @patch("exprouter.providers.anthropic.call")
    def test_schema_enforcement_retry(self, mock_anthropic_call, mock_load_config):
        mock_load_config.return_value = self.test_config
        # First call fails schema, second succeeds
        mock_anthropic_call.side_effect = [
            ("Thinking...", 10, 5),
            ("Reason: test, Choice: A", 10, 5)
        ]
        
        router = Router()
        resp = router.complete(
            "claude-3-5-sonnet", 
            [{"role": "user", "content": "hi"}], 
            required_fields=["Reason:", "Choice:"]
        )
        self.assertIn("Choice: A", resp)
        self.assertEqual(mock_anthropic_call.call_count, 2)

    @patch("exprouter.router.load_router_config")
    @patch("exprouter.providers.anthropic.call")
    @patch("exprouter.providers.openai.call")
    def test_all_providers_exhausted(self, mock_openai_call, mock_anthropic_call, mock_load_config):
        mock_load_config.return_value = self.test_config
        # Simulate errors from all providers
        mock_anthropic_call.side_effect = Exception("429 Rate Limit")
        mock_openai_call.side_effect = Exception("500 Internal Server Error")
        
        router = Router()
        with self.assertRaises(ProviderExhaustedError):
            router.complete("gpt-4o", [{"role": "user", "content": "hi"}])

    @patch("exprouter.router.load_router_config")
    @patch("exprouter.providers.anthropic.call")
    def test_budget_exceeded(self, mock_anthropic_call, mock_load_config):
        mock_load_config.return_value = self.test_config
        mock_anthropic_call.return_value = ("Success", 1000, 1000) # $ cost will be high
        
        # Set a very low budget
        router = Router(budget=0.00001)
        # First call should succeed if it's the first one, but wait
        # The budget check is BEFORE the call.
        # Let's set budget to 0.0 to trigger it.
        router.budget = 0.0
        with self.assertRaises(BudgetExceededError) as cm:
            router.complete("claude-3-5-sonnet", [{"role": "user", "content": "hi"}])
        self.assertIn("Budget exceeded", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
