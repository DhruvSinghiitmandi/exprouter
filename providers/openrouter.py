import os
from openai import OpenAI

def call(model_id, messages, system, temperature, max_tokens, seed):
    """
    Calls OpenRouter API using the OpenAI client.
    Requires OPENROUTER_API_KEY.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    # Merge system prompt into messages if present
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)
    
    # OpenRouter specific headers and parameters
    extra_headers = {"HTTP-Referer": "https://github.com/exprouter"}
    extra_body = {}
    if seed is not None:
        extra_body["seed"] = seed
    
    response = client.chat.completions.create(
        model=model_id,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=extra_headers,
        extra_body=extra_body
    )
    
    text = response.choices[0].message.content
    in_tok = response.usage.prompt_tokens
    out_tok = response.usage.completion_tokens
    
    return text, in_tok, out_tok
