import click
import json
import os
# Use absolute imports or relative imports depending on how it's run
# If run as 'python3 -m exprouter.cli', relative works.
from .router import Router

@click.group()
def cli():
    """exprouter - A resilient LLM router."""
    pass

@cli.command()
@click.option("--model", required=True, help="Canonical model name (e.g., gpt-4o)")
@click.option("--prompt", required=True, help="User prompt string")
@click.option("--system", default="", help="System prompt string")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature")
@click.option("--max-tokens", default=1024, type=int, help="Max tokens to generate")
@click.option("--seed", type=int, default=None, help="Random seed for generation")
@click.option("--budget", type=float, default=None, help="Budget limit for this call")
def complete(model, prompt, system, temperature, max_tokens, seed, budget):
    """Executes a completion request via the router."""
    router = Router(budget=budget)
    messages = [{"role": "user", "content": prompt}]
    try:
        response = router.complete(
            model=model,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        )
        click.echo(response)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command(name="log-show")
@click.option("--path", default="routers.jsonl", help="Path to log file")
def log_show(path):
    """Displays formatted logs from routers.jsonl."""
    if not os.path.exists(path):
        click.echo(f"Log file {path} not found.")
        return
    
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "error":
                    ts = data.get('timestamp', 'N/A')
                    provider = data.get('provider', 'N/A')
                    error = data.get('error', 'Unknown Error')
                    click.echo(f"{ts} | ERROR | {provider} | {error}")
                else:
                    ts = data.get("timestamp", "")
                    provider = data.get("provider", "")
                    model = data.get("model", "")
                    cost = data.get("cost", 0.0)
                    resp = data.get("response", "").replace("\n", " ")[:80]
                    click.echo(f"{ts} | {provider} | {model} | ${cost:.4f} | {resp}")
            except (json.JSONDecodeError, KeyError):
                # Print non-JSON or malformed lines as-is
                click.echo(line)

if __name__ == "__main__":
    cli()
