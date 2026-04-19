import json
import hashlib
import importlib
from datetime import datetime, timedelta
from .config import PROVIDER_MODELS, COST_PER_1K, load_router_config

class ProviderExhaustedError(Exception):
    pass

class MalformedResponseError(Exception):
    def __init__(self, message, raw_response=None):
        super().__init__(message)
        self.raw_response = raw_response

class BudgetExceededError(Exception):
    pass

class Router:
    def __init__(self, log_path="routers.jsonl", config_path="router_config.json", budget=None):
        self.log_path = log_path
        self.config_path = config_path
        self.budget = budget
        
        self.providers = load_router_config(config_path)
        # Initialize counts and spend
        self._counts = {p["name"]: {"rpm": 0, "rpd": 0} for p in self.providers}
        self._total_spend = 0.0
        
        # Initialize reset times: rpm resets in 1 min, rpd resets at midnight
        now = datetime.utcnow()
        self._reset_times = {
            p["name"]: {
                "rpm": now + timedelta(minutes=1),
                "rpd": (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            }
            for p in self.providers
        }

    def _maybe_reset(self, provider_name):
        """Lazy check-on-use reset of rate limit counters."""
        now = datetime.utcnow()
        resets = self._reset_times[provider_name]
        
        if now >= resets["rpm"]:
            self._counts[provider_name]["rpm"] = 0
            self._reset_times[provider_name]["rpm"] = now + timedelta(minutes=1)
            
        if now >= resets["rpd"]:
            self._counts[provider_name]["rpd"] = 0
            self._reset_times[provider_name]["rpd"] = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def _check_rate_limit(self, provider_name):
        """Checks if a provider is within its RPM and RPD limits."""
        if provider_name not in self._counts:
            return False
            
        self._maybe_reset(provider_name)
        counts = self._counts[provider_name]
        
        # Find settings for this provider
        settings = next((p for p in self.providers if p["name"] == provider_name), {})
        
        if counts["rpm"] >= settings.get("rpm", float('inf')):
            return False
        if counts["rpd"] >= settings.get("rpd", float('inf')):
            return False
        return True

    def _select_provider(self, model, excluded=None):
        """
        Selects the highest priority provider that supports the model and is within rate limits.
        Excluded is a set of provider names to skip.
        """
        if excluded is None:
            excluded = set()
            
        supported_mapping = PROVIDER_MODELS.get(model, {})
        
        for p_info in self.providers:
            name = p_info["name"]
            if name in supported_mapping and name not in excluded:
                if self._check_rate_limit(name):
                    return name, supported_mapping[name]
                    
        raise ProviderExhaustedError(f"No available providers for model {model} (excluded: {excluded})")

    def _log(self, data):
        """Appends a line to the JSONL log file."""
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def complete(self, model, messages, system="", temperature=0.7, max_tokens=1024, seed=None, required_fields=None):
        """
        Main routing function. Handles retries, logging, and schema enforcement.
        """
        if self.budget is not None and self._total_spend >= self.budget:
            raise BudgetExceededError(f"Budget exceeded. Current spend: ${self._total_spend:.4f}, Budget: ${self.budget:.4f}")

        excluded = set()
        while True:
            try:
                provider_name, provider_model = self._select_provider(model, excluded)
            except ProviderExhaustedError:
                raise

            try:
                # Dynamic call to provider module
                # Note: Assuming 'exprouter' is the package name
                provider_module = importlib.import_module(f".providers.{provider_name}", package="exprouter")
                
                text, in_tok, out_tok = provider_module.call(
                    model_id=provider_model,
                    messages=messages,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed
                )
                
                # Check schema if required
                if required_fields:
                    missing = self._enforce_schema(text, required_fields)
                    if missing:
                        # Correction retry logic
                        correction_prompt = f"Your previous response was missing required fields: {', '.join(missing)}. Please provide the full response again, ensuring all fields are included exactly."
                        new_messages = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": correction_prompt}]
                        
                        text, in_tok2, out_tok2 = provider_module.call(
                            model_id=provider_model,
                            messages=new_messages,
                            system=system,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            seed=seed
                        )
                        in_tok += in_tok2
                        out_tok += out_tok2
                        
                        # Check again
                        missing = self._enforce_schema(text, required_fields)
                        if missing:
                            raise MalformedResponseError(f"Second attempt failed to include fields: {missing}", text)

                # Update state
                self._counts[provider_name]["rpm"] += 1
                self._counts[provider_name]["rpd"] += 1
                
                # Calculate cost
                rate_in, rate_out = COST_PER_1K.get(model, (0, 0))
                cost = (in_tok / 1000.0 * rate_in) + (out_tok / 1000.0 * rate_out)
                self._total_spend += cost
                
                # Hashing exactly as specified: sha256 of json.dumps of {"system": s, "messages": m} sorted keys
                prompt_content = {"system": system, "messages": messages}
                prompt_hash = hashlib.sha256(json.dumps(prompt_content, sort_keys=True).encode()).hexdigest()[:16]
                
                log_entry = {
                    "ts": datetime.utcnow().isoformat(),
                    "prompt_hash": prompt_hash,
                    "provider": provider_name,
                    "model": model,
                    "cost": cost,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "response": text
                }
                self._log(log_entry)
                
                return text

            except (MalformedResponseError, ProviderExhaustedError) as e:
                # These are terminal or should be re-raised
                raise e
            except Exception as e:
                # Log error and fallback
                self._log({
                    "ts": datetime.utcnow().isoformat(),
                    "provider": provider_name,
                    "error": str(e),
                    "type": "error"
                })
                excluded.add(provider_name)
                continue

    def _enforce_schema(self, response, required_fields):
        """Simple substring check for required fields."""
        if not required_fields:
            return []
        missing = [f for f in required_fields if f not in response]
        return missing

    def spend_report(self):
        """Returns the total spend accumulated in this session and by provider."""
        return {
            "total_spend": self._total_spend,
            "counts": self._counts
        }