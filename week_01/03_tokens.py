# tokens
# common reusable words - entire internet is scanned and common words are stored 
# tokens are split and understood like playing = play + ing, similar works for unknown words
# s + ou + ra + jit = sourajit

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("API KEY NOT FOUND")

client = Groq(api_key=GROQ_API_KEY)

prompt1 = "what is tokens for LLM?"
prompt2 = "write a love letter for Sohalina"
prompt3 = "Heyy"

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=50
    )
    usage = response.usage
    print(
        f"Prompt: {prompt} --- input_tokens: {usage.prompt_tokens} "
        f"--- completion_tokens: {usage.completion_tokens} "
        f"--- total_tokens: {usage.total_tokens} --- Finish Reason: {response.choices[0].finish_reason}"
    )

# max_tokens is used to limit no. of tokens will be using
