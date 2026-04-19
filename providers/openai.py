import os
from openai import OpenAI

def call(model_id, messages, system, temperature, max_tokens, seed):
    """
    Calls OpenAI API.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Merge system prompt into messages if present
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)
    
    response = client.chat.completions.create(
        model=model_id,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed
    )
    
    text = response.choices[0].message.content
    in_tok = response.usage.prompt_tokens
    out_tok = response.usage.completion_tokens
    
    return text, in_tok, out_tok
