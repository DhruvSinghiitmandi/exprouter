import pytest
import requests
import json
from unittest.mock import patch
from exprouter.router import Router

@pytest.mark.live
class TestProviderDeepSeek:

    def test_key_validation(self, require_key):
        api_key = require_key("DEEPSEEK_API_KEY")
        resp = requests.get(
            "https://api.deepseek.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        if resp.status_code == 401:
            pytest.fail("DEEPSEEK_API_KEY is invalid.")
        assert resp.status_code == 200

    def test_prisoners_format(self, require_key, prisoners_system, prisoners_prompt):
        require_key("DEEPSEEK_API_KEY")
        router = Router()
        response = router.complete(
            model="deepseek-chat",
            messages=prisoners_prompt,
            system=prisoners_system,
            required_fields=["Reason:", "Choice:"]
        )
        assert "Reason:" in response
        assert "Choice:" in response

    def test_reasoning_bleed(self, require_key, prisoners_system, prisoners_prompt):
        require_key("DEEPSEEK_API_KEY")
        router = Router()
        # Verify R1 reasoning bleed catches
        with patch("exprouter.providers.deepseek.call", wraps=__import__("exprouter").providers.deepseek.call) as mock_call:
            response = router.complete(
                model="deepseek-r1",
                messages=prisoners_prompt,
                system=prisoners_system,
                required_fields=["Reason:", "Choice:"]
            )
            assert "Choice: U" in response or "Choice: C" in response or "Choice: D" in response
            assert mock_call.call_count <= 2

    def test_cost_accuracy(self, require_key, prisoners_system, prisoners_prompt):
        require_key("DEEPSEEK_API_KEY")
        router = Router()
        router.complete("deepseek-chat", prisoners_prompt, system=prisoners_system)
        with open(router.log_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        last_entry = lines[-1]
        
        assert last_entry["cost"] > 0
        assert last_entry["cost"] < 0.001

    @patch("exprouter.providers.deepseek.call")
    def test_fallback_deepseek_to_openrouter(self, mock_deepseek_call, require_key):
        require_key("OPENROUTER_API_KEY") # To hit fallback successfully
        
        mock_deepseek_call.side_effect = Exception("500")
        router = Router()
        resp = router.complete("deepseek-r1", [{"role": "user", "content": "hello"}])
        assert resp
        
        with open(router.log_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        error_entry = lines[-2]
        success_entry = lines[-1]
        assert error_entry["provider"] == "deepseek"
        assert success_entry["provider"] == "openrouter"
        assert success_entry["model"] == "deepseek-r1"
