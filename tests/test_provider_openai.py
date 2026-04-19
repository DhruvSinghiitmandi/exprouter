import pytest
import openai
import json
from unittest.mock import patch
from exprouter.router import Router

@pytest.mark.live
class TestProviderOpenAI:
    
    def test_key_validation(self, require_key):
        api_key = require_key("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
        try:
            client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
        except openai.AuthenticationError:
            pytest.fail("OPENAI_API_KEY is invalid.")
            
    def test_prisoners_format(self, require_key, prisoners_system, prisoners_prompt):
        require_key("OPENAI_API_KEY")
        router = Router()
        response = router.complete(
            model="gpt-4o",
            messages=prisoners_prompt,
            system=prisoners_system,
            required_fields=["Reason:", "Choice:"]
        )
        assert "Reason:" in response
        assert "Choice:" in response
        assert "Choice: U" in response or "Choice: C" in response or "Choice: D" in response

    def test_seed_determinism(self, require_key, prisoners_system, prisoners_prompt):
        require_key("OPENAI_API_KEY")
        router = Router()
        # Seed test - GPT-4o actually returns deterministic results with same seed
        resp1 = router.complete("gpt-4o", prisoners_prompt, system=prisoners_system, seed=42)
        resp2 = router.complete("gpt-4o", prisoners_prompt, system=prisoners_system, seed=42)
        assert resp1 == resp2

    @patch("exprouter.providers.openai.call")
    def test_fallback_openai_to_openrouter(self, mock_openai_call, require_key):
        require_key("OPENROUTER_API_KEY")  # Need to test fallback properly. Let it be live for openrouter
        
        # Test routing openai (fallback triggered by 429 string or mock error)
        # Configure the router specifically for this 
        mock_openai_call.side_effect = Exception("429 RateLimit")
        router = Router()
        
        # gpt-4o should be hit, but openai fails, so openrouter intercepts. 
        resp = router.complete("gpt-4o", [{"role": "user", "content": "Say 'hello from openrouter'."}])
        assert resp
        
        # We can confirm this using log since it ran live on OpenRouter
        with open(router.log_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        # Should record an error line, then an openrouter success line
        error_entry = lines[-2]
        success_entry = lines[-1]
        assert error_entry["type"] == "error"
        assert error_entry["provider"] == "openai"
        assert success_entry["provider"] == "openrouter"
