# call an llm

# messages are sent as an array to the llm
# response.choices[0].message.content this returns only the answer of the llm
# the llm also returns token details like used in prompt, time needed

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API KEY NOT FOUND!")

client = Groq(api_key=API_KEY)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "what does Sourajit name means?",
        }
    ]
)

print(response)

print("*"*100)

answer = response.choices[0].message.content
print(answer)