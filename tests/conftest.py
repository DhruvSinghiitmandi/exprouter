import os
import pytest
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def require_key():
    """Returns a factory function to check if a key exists, skipping the test otherwise."""
    def _require_key(env_var: str):
        key = os.environ.get(env_var)
        if not key:
            pytest.skip(f"{env_var} not set — skipping live test")
        return key
    return _require_key

@pytest.fixture
def prisoners_system():
    return "You are a participant in the prisoner's dilemma."

@pytest.fixture
def prisoners_prompt():
    return [{"role": "user", "content": "Will you Cooperate (C) or Defect (D)? Provide your Choice: and Reason: strictly."}]

def validate_keys():
    """Prints a summary table of the active provider keys at test start."""
    keys = {
        "ANTHROPIC_API_KEY": "Anthropic",
        "OPENAI_API_KEY": "OpenAI",
        "DEEPSEEK_API_KEY": "DeepSeek",
        "OPENROUTER_API_KEY": "OpenRouter",
        "DASHSCOPE_API_KEY": "DashScope"
    }
    print("\n--- Active API Keys Summary ---")
    for var, provider in keys.items():
        status = "✅ ACTIVE" if os.environ.get(var) else "❌ MISSING"
        print(f"{provider:<12} | {var:<20} | {status}")
    print("-------------------------------\n")

def pytest_sessionstart(session):
    validate_keys()
