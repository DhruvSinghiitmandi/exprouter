import pytest
import anthropic
import json
from exprouter.router import Router

@pytest.mark.live
class TestProviderAnthropic:
    
    def test_key_validation(self, require_key):
        api_key = require_key("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        try:
            client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
        except anthropic.AuthenticationError:
            pytest.fail("ANTHROPIC_API_KEY is invalid.")
            
    def test_prisoners_format(self, require_key, prisoners_system, prisoners_prompt):
        require_key("ANTHROPIC_API_KEY")
        router = Router()
        response = router.complete(
            model="claude-3-5-sonnet",
            messages=prisoners_prompt,
            system=prisoners_system,
            required_fields=["Reason:", "Choice:"]
        )
        assert "Reason:" in response
        assert "Choice:" in response
        assert "Choice: U" in response or "Choice: C" in response or "Choice: D" in response
        print(f"Raw Response: {response}")

    def test_seed_reproducibility(self, require_key, prisoners_system, prisoners_prompt):
        require_key("ANTHROPIC_API_KEY")
        router = Router()
        prompt = [{"role": "user", "content": "Tell me a random word."}]
        router.complete("claude-3-5-sonnet", prompt, seed=42)
        router.complete("claude-3-5-sonnet", prompt, seed=42)
        
        # Read the last two lines of the log and assert hashes are identical
        with open(router.log_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        entry1_hash = lines[-2]["prompt_hash"]
        entry2_hash = lines[-1]["prompt_hash"]
        assert entry1_hash == entry2_hash

    def test_rate_limit_increments(self, require_key, prisoners_system, prisoners_prompt):
        require_key("ANTHROPIC_API_KEY")
        router = Router()
        router.complete("claude-3-5-sonnet", [{"role": "user", "content": "hi"}])
        assert router._counts["anthropic"]["rpm"] >= 1
