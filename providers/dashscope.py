import os
from openai import OpenAI

def call(model_id, messages, system, temperature, max_tokens, seed):
    """
    Calls DashScope API (Alibaba) using the generic OpenAI client.
    Requires DASHSCOPE_API_KEY.
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key
    )
    
    # Merge system prompt into messages if present
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)
    
    # Optional parameters for advanced DashScope usage.
    # Enable Qwen reasoning features if required.
    extra_body = {}
    if "reasoning" in model_id.lower() or "thinking" in model_id.lower():
        extra_body["enable_thinking"] = True

    kwargs = {
        "model": model_id,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    if seed is not None:
        kwargs["seed"] = seed

    response = client.chat.completions.create(**kwargs)
    
    text = response.choices[0].message.content
    in_tok = response.usage.prompt_tokens
    out_tok = response.usage.completion_tokens
    
    return text, in_tok, out_tok
