import os
import google.generativeai as genai

def call(model_id, messages, system, temperature, max_tokens, seed):
    """
    Calls Gemini API using google-generativeai.
    System prompt goes into system_instruction.
    Messages are converted to Gemini's role/parts format.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    # Map roles: 'user' -> 'user', 'assistant' -> 'model'
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [msg["content"]]})
    
    model = genai.GenerativeModel(
        model_name=model_id,
        system_instruction=system if system else None
    )
    
    # Configure generation
    gen_config = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    
    # Pass seed if supported by the SDK version, silently ignore if not
    if seed is not None:
        gen_config["seed"] = seed

    try:
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(**gen_config)
        )
        
        text = response.text
        in_tok = response.usage_metadata.prompt_token_count
        out_tok = response.usage_metadata.candidates_token_count
        
        return text, in_tok, out_tok
    except Exception as e:
        # Re-raise to let the router handle it
        raise e
