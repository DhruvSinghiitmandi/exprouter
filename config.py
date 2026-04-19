import json
import os

# Maps canonical names to provider-specific names
PROVIDER_MODELS = {
    "claude-3-5-sonnet": {"anthropic": "claude-3-5-sonnet-20241022"},
    "gpt-4o":            {"openai": "gpt-4o-2024-11-20", "openrouter": "openai/gpt-4o"},
    "deepseek-r1":       {"openrouter": "deepseek/deepseek-r1"},
    "gemini-pro":        {"gemini": "gemini-1.5-pro"},
}

# (Input cost per 1K, Output cost per 1K)
COST_PER_1K = {
    "claude-3-5-sonnet": (0.003, 0.015),
    "gpt-4o":            (0.0025, 0.010),
    "deepseek-r1":       (0.0001, 0.0002),
    "gemini-pro":        (0.00125, 0.00375),
}

def load_router_config(config_path="router_config.json"):
    """Loads router configuration and returns a list of provider settings sorted by priority."""
    if not os.path.exists(config_path):
        return []
        
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    
    # Return list of dicts with provider name included, sorted by priority ascending
    providers = []
    for name, settings in config.items():
        provider_dict = settings.copy()
        provider_dict["name"] = name
        providers.append(provider_dict)
    
    return sorted(providers, key=lambda x: x.get("priority", 999))
