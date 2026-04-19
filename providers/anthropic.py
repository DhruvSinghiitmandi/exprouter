import os
import anthropic

def call(model_id, messages, system, temperature, max_tokens, seed):
    """
    Calls Anthropic API. Note: Anthropic SDK does not support a 'seed' parameter directly.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        temperature=temperature
    )
    
    text = response.content[0].text
    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    
    return text, in_tok, out_tok
